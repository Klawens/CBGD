import os
from typing import Dict

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

from models.diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from models.unet import UNet
from utils.scheduler import GradualWarmupScheduler
from utils.dataset import PairedDataset
from utils.helper_funcs import sp_ch_3, sp_ch_1, PairRandomCrop, expand_channel
from models.losses import SmoothLossL, SSIM

# fp16
from torch.cuda.amp import autocast, GradScaler


def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])

    # dataset
    tf = transforms.Compose([
    transforms.Resize([modelConfig["img_size"], modelConfig["img_size"]]),
    transforms.ToTensor(),
    ])

    paired_dataset = PairedDataset(
        root_dir_low=modelConfig["data_dir"]+'low',
        root_dir_high=modelConfig["data_dir"]+'high',
        transform=tf
        )
    paired_dataloader = DataLoader(
        paired_dataset, batch_size=modelConfig["batch_size"], shuffle=False, num_workers=2, drop_last=True, pin_memory=True)

    paired_dataset_R = PairedDataset(
        root_dir_low=modelConfig["data_dir"]+'decom_gts/low_r',
        root_dir_high=modelConfig["data_dir"]+'decom_gts/high_r',
        transform=tf
        )
    paired_dataloader_R = DataLoader(
        paired_dataset_R, batch_size=modelConfig["batch_size"], shuffle=False, num_workers=2, drop_last=True, pin_memory=True)

    paired_dataset_L = PairedDataset(
        root_dir_low=modelConfig["data_dir"]+'decom_gts/low_i',
        root_dir_high=modelConfig["data_dir"]+'decom_gts/high_i',
        transform=tf
        )
    paired_dataloader_L = DataLoader(
        paired_dataset_L, batch_size=modelConfig["batch_size"], shuffle=False, num_workers=2, drop_last=True, pin_memory=True)


    # model setup
    net_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"], num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)

    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device))

    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
    sampler = GaussianDiffusionSampler(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
    
    ckp_path = os.path.join(modelConfig["save_weight_dir"], modelConfig["resume_from"])
    start_epoch = 0
    if os.path.isfile(ckp_path):
        checkpoint = torch.load(ckp_path, map_location=device)
        net_model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        print("resuming...")

    scaler = GradScaler()

    # start training
    for e in range(start_epoch, modelConfig["epoch"]):
        with tqdm(paired_dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for paired_images, paired_R, paired_L in zip(tqdmDataLoader, paired_dataloader_R, paired_dataloader_L):
                # TRAIN
                optimizer.zero_grad()
                # fp16
                with autocast():
                    # get low
                    low_ori = paired_images[0][:, :3, :, :].to(device)
                    # get high
                    high_ori = paired_images[0][:, 3:, :, :].to(device)
                    # get low reflectance
                    low_R = paired_R[0][:, :3, :, :].to(device)
                    # get low illumination
                    low_L = paired_L[0][:, :3, :, :].to(device)
                    # get high reflectance
                    high_R = paired_R[0][:, 3:, :, :].to(device)
                    # get high illumination
                    high_L = paired_L[0][:, 3:, :, :].to(device)
                    # crop input
                    low_ori, high_ori, low_R, low_L, high_R, high_L = PairRandomCrop(low_ori, high_ori, low_R, low_L, high_R, high_L, modelConfig["crop_size"])
                    # concat input
                    trainer_input = torch.cat((high_R, high_L[:, :1, :, :]), dim=1) # [b, 4, h, w]
                    sampler_input = torch.cat((low_R, low_L[:, :1, :, :]), dim=1) # [b, 4, h, w]

                    # Multi-channel Diffusion
                    diff_loss, x_t = trainer(trainer_input)
                    # L_diff
                    diff_loss = diff_loss.sum() / 1000.
                    # fwd_image = sp_ch_3(x_t) * expand_channel(sp_ch_1(x_t))

                    # Reverse Guidance
                    recon = sampler(sampler_input)
                    rec_image = sp_ch_3(recon) * expand_channel(sp_ch_1(recon))
                    # L_guid
                    loss_color = SSIM().ssim_loss(high_R, sp_ch_3(recon))
                    loss_light = SmoothLossL().smooth(sp_ch_1(recon), high_R)
                    recon_loss = SSIM().ssim_loss(high_ori, rec_image)
                    
                    # total loss
                    w1, w2, w3, w4 = 0.4, 3, 3, 5
                    loss = w1 * diff_loss + w2 * loss_color + w3 * loss_light + w4 * recon_loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                
                # SAVE IMAGES
                '''
                your code here
                '''

                # screen logger
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss": loss.item(),
                    "diffusion loss": w1 * diff_loss.item(),
                    "color loss": w2 * loss_color.item(),
                    "light loss": w3 * loss_light.item(),
                    "recon loss": w4 * recon_loss.item(),
                    "img shape": x_t.shape[-1],
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()

        # save weights
        ckp = {
            'epoch': e,
            'state_dict': net_model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }
        if not os.path.exists(modelConfig["save_weight_dir"]):
            os.makedirs(modelConfig["save_weight_dir"])
        torch.save(ckp, os.path.join(
            modelConfig["save_weight_dir"], 'ckpt_' + str(e) + "_.pt"))
        latest_ckp_link = os.path.join(modelConfig["save_weight_dir"], 'latest.pt')
        if os.path.islink(latest_ckp_link):
            os.remove(latest_ckp_link)
        os.symlink('ckpt_' + str(e) + "_.pt", latest_ckp_link)
        

def main(model_config = None):
    modelConfig = {
        "epoch": 150,
        "batch_size": 1,
        "T": 100,
        "channel": 64,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 5e-6,
        "multiplier": 2.,
        "beta_1": 2.5e-5,
        "beta_T": 0.002,
        "img_size": 512,
        "crop_size": 144,
        "grad_clip": 1.,
        "device": "cuda:0",
        "training_load_weight": None,
        "data_dir": "/home/lsc/LOL/",
        "save_weight_dir": "./checkpoints/",
        "test_load_weight": "latest.pt",
        "sampled_dir": "./results/",
        "nrow": 3,
        "resume_from": "latest.pt"
        }
    if model_config is not None:
        modelConfig = model_config
    
    train(modelConfig)


if __name__ == '__main__':
    main()
