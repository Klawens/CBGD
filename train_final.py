import os
from typing import Dict

import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from unet import UNet
from utils.scheduler import GradualWarmupScheduler
from utils.dataset import CustomDataset
from utils.helper_funcs import extract_illumination_map_batch, sp_ch_3, sp_ch_1, add_text_to_image, PairRandomCrop, expand_channel
from losses import SmoothLossL, SmoothLossR, SSIM, PSNR

# fp16
from torch.cuda.amp import autocast, GradScaler


def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    # dataset
    dataset_high = CustomDataset(
        root_dir=modelConfig["data_dir"]+'high',
        transform=transforms.Compose([
            transforms.Resize([modelConfig["img_size"], modelConfig["img_size"]]),
            transforms.ToTensor(),
            ])
        )
    dataloader_high = DataLoader(
        dataset_high, batch_size=modelConfig["batch_size"], shuffle=False, num_workers=2, drop_last=True, pin_memory=True)
    
    dataset_low = CustomDataset(
        root_dir=modelConfig["data_dir"]+'low',
        transform=transforms.Compose([
            transforms.Resize([modelConfig["img_size"], modelConfig["img_size"]]),
            transforms.ToTensor(),
            ])
        )
    dataloader_low = DataLoader(
        dataset_low, batch_size=modelConfig["batch_size"], shuffle=False, num_workers=2, drop_last=True, pin_memory=True)
    
    dataset_low_r = CustomDataset(
        root_dir=modelConfig["data_dir"]+'decom_gts/low_r',
        transform=transforms.Compose([
            transforms.Resize([modelConfig["img_size"], modelConfig["img_size"]]),
            transforms.ToTensor(),
            ])
        )
    dataloader_low_r = DataLoader(
        dataset_low_r, batch_size=modelConfig["batch_size"], shuffle=False, num_workers=2, drop_last=True, pin_memory=True)
    
    dataset_low_i = CustomDataset(
        root_dir=modelConfig["data_dir"]+'decom_gts/low_i',
        transform=transforms.Compose([
            transforms.Resize([modelConfig["img_size"], modelConfig["img_size"]]),
            transforms.ToTensor(),
            ])
        )
    dataloader_low_i = DataLoader(
        dataset_low_i, batch_size=modelConfig["batch_size"], shuffle=False, num_workers=2, drop_last=True, pin_memory=True)
    
    dataset_high_r = CustomDataset(
        root_dir=modelConfig["data_dir"]+'decom_gts/high_r',
        transform=transforms.Compose([
            transforms.Resize([modelConfig["img_size"], modelConfig["img_size"]]),
            transforms.ToTensor(),
            ])
        )
    dataloader_high_r = DataLoader(
        dataset_high_r, batch_size=modelConfig["batch_size"], shuffle=False, num_workers=2, drop_last=True, pin_memory=True)
    
    dataset_high_i = CustomDataset(
        root_dir=modelConfig["data_dir"]+'decom_gts/high_i',
        transform=transforms.Compose([
            transforms.Resize([modelConfig["img_size"], modelConfig["img_size"]]),
            transforms.ToTensor(),
            ])
        )
    dataloader_high_i = DataLoader(
        dataset_high_i, batch_size=modelConfig["batch_size"], shuffle=False, num_workers=2, drop_last=True, pin_memory=True)

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
        n = 0
        with tqdm(dataloader_high, dynamic_ncols=True) as tqdmDataLoader:
            for high_images, low_images, low_r, low_i, high_r, high_i in zip(tqdmDataLoader, dataloader_low, dataloader_low_r, dataloader_low_i, dataloader_high_r, dataloader_high_i):
                n += 1
                # TRAIN
                optimizer.zero_grad()
                # fp16
                with autocast():
                    # get high
                    high_ori = high_images.to(device) # [b, 3, h, w]
                    # get low
                    low_ori = low_images.to(device) # [b, 3, h, w]
                    # get high reflectance
                    high_r = high_r.to(device) # [b, 3, h, w]
                    # get high illumination
                    high_i = high_i.to(device) # [b, 3, h, w]
                    # get low reflectance
                    low_r = low_r.to(device) # [b, 3, h, w]
                    # get low illumination
                    low_i = low_i.to(device) # [b, 3, h, w]
                    # x_0, low_ori = PairRandomCrop(x_0, low_ori, modelConfig["crop_size"])
                    # print(low_i[:, 0, :, :].shape, high_i.shape)
                    trainer_input = torch.cat((high_r, high_i[:, :1, :, :]), dim=1) # [b, 4, h, w]
                    sampler_input = torch.cat((low_r, low_i[:, :1, :, :]), dim=1) # [b, 4, h, w]

                    diff_loss, x_t = trainer(trainer_input)
                    diff_loss = diff_loss.sum() / 1000.
                    fwd_image = sp_ch_3(x_t) * expand_channel(sp_ch_1(x_t))

                    recon = sampler(sampler_input)
                    rec_image = sp_ch_3(recon) * expand_channel(sp_ch_1(recon))
                    rec_image_1 = high_r * expand_channel(sp_ch_1(recon))

                    # loss_color = F.l1_loss(high_r, sp_ch_3(recon))
                    # loss_color = PSNR().psnr_loss(high_r, sp_ch_3(recon))
                    loss_color = SSIM().ssim_loss(high_r, sp_ch_3(recon))
                    loss_light = SmoothLossL().smooth(sp_ch_1(recon), high_r)
                    # l1, ssim, psnr
                    # recon_loss = F.l1_loss(high_ori, rec_image)# + F.l1_loss(high_ori, rec_image_1)
                    recon_loss = SSIM().ssim_loss(high_ori, rec_image) + 0.5 * SSIM().ssim_loss(high_ori, rec_image_1)
                    # recon_loss = PSNR().psnr_loss(high_ori, rec_image)
                    
                    # total loss
                    w1, w2, w3, w4 = 1, 5, 3, 5
                    loss = w1 * diff_loss + w2 * loss_color + w3 * loss_light + w4 * recon_loss
                scaler.scale(loss).backward()
                # loss.backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                scaler.step(optimizer)
                # optimizer.step()
                scaler.update()
                
                # SAVE IMAGES
                # r_r = add_text_to_image(sp_ch_3(recon), 'rvrs_ref')  # reverse reflectance
                # r_i = add_text_to_image(sp_ch_1(recon), 'rvrs_ilu')  # reverse illumination
                # high_ori = add_text_to_image(high_ori.cpu(), 'high')  # high original image
                # low_ori = add_text_to_image(low_ori.cpu(), 'low')  # low original image
                # rec_image = add_text_to_image(rec_image.cpu(), 'recon')  # reconstructed image
                # # low_r = add_text_to_image(low_ori/low_illumination.cpu(), 'low_ref')  # low reflectance
                # low_r = add_text_to_image(sp_ch_3(low_r), 'in_ref')  # low reflectance
                # low_illumination = add_text_to_image(low_i.cpu(), 'in_ilu')  # low illumination
                # gt_reflectance = add_text_to_image(high_r.cpu(), 'gt_ref')  # ground truth reflectance
                # gt_illumination = add_text_to_image(high_i.cpu(), 'gt_ilu')  # ground truth illumination
                # fwd_image = add_text_to_image(fwd_image.cpu(), 'fwd_nois')  # forward image
                # concat images
                # v = torch.cat((high_ori.cpu(), low_ori.cpu(), rec_image.cpu(), r_r.cpu(), r_i.cpu(), rec_image.cpu(), low_r.cpu(), low_illumination.cpu(), low_ori.cpu(), gt_reflectance.cpu(), gt_illumination.cpu(), fwd_image.cpu()), dim=0)
                # create path
                if os.path.exists(modelConfig["sampled_dir"]+'epoch_{}/'.format(e)) is False:
                    os.makedirs(modelConfig["sampled_dir"]+'epoch_{}/'.format(e))
                # save images
                # save_image(v, os.path.join(
                #     modelConfig["sampled_dir"]+'epoch_{}/'.format(e), "{}_{}.png".format(e, n)), nrow=modelConfig["nrow"])
                save_image(sp_ch_3(recon), os.path.join(
                    modelConfig["sampled_dir"]+'epoch_{}/'.format(e), "{}_{}_rvrs_ref.png".format(e, n)), nrow=modelConfig["nrow"])
                save_image(sp_ch_1(recon), os.path.join(
                    modelConfig["sampled_dir"]+'epoch_{}/'.format(e), "{}_{}_rvrs_ilu.png".format(e, n)), nrow=modelConfig["nrow"])
                save_image(high_ori, os.path.join(
                    modelConfig["sampled_dir"]+'epoch_{}/'.format(e), "{}_{}_high.png".format(e, n)), nrow=modelConfig["nrow"])
                save_image(low_ori, os.path.join(
                    modelConfig["sampled_dir"]+'epoch_{}/'.format(e), "{}_{}_low.png".format(e, n)), nrow=modelConfig["nrow"])
                save_image(rec_image, os.path.join(
                    modelConfig["sampled_dir"]+'epoch_{}/'.format(e), "{}_{}_recon.png".format(e, n)), nrow=modelConfig["nrow"])
                save_image(sp_ch_3(low_r), os.path.join(
                    modelConfig["sampled_dir"]+'epoch_{}/'.format(e), "{}_{}_in_ref.png".format(e, n)), nrow=modelConfig["nrow"])
                save_image(low_i, os.path.join(
                    modelConfig["sampled_dir"]+'epoch_{}/'.format(e), "{}_{}_in_ilu.png".format(e, n)), nrow=modelConfig["nrow"])
                save_image(high_r, os.path.join(
                    modelConfig["sampled_dir"]+'epoch_{}/'.format(e), "{}_{}_gt_ref.png".format(e, n)), nrow=modelConfig["nrow"])
                save_image(high_i, os.path.join(
                    modelConfig["sampled_dir"]+'epoch_{}/'.format(e), "{}_{}_gt_ilu.png".format(e, n)), nrow=modelConfig["nrow"])
                save_image(fwd_image, os.path.join(
                    modelConfig["sampled_dir"]+'epoch_{}/'.format(e), "{}_{}_fwd_nois.png".format(e, n)), nrow=modelConfig["nrow"])

                with open("ssimx3_100e_144_39t_log.txt", "a") as log_file:
                    log_file.write("epoch: {}, ".format(e))
                    log_file.write("loss: {}, ".format(round(loss.item(), 3)))
                    log_file.write("diffusion loss: {}, ".format(round(w1 * diff_loss.item(), 3)))
                    log_file.write("color loss: {}, ".format(round(w2 * loss_color.item(), 3)))
                    log_file.write("light loss: {}, ".format(round(w3 * loss_light.item(), 3)))
                    log_file.write("recon loss: {}, ".format(round(w4 * recon_loss.item(), 3)))
                    log_file.write("LR: {}, \n".format(optimizer.state_dict()['param_groups'][0]["lr"]))
                    # log_file.write("\n")
                
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
        torch.save(ckp, os.path.join(
            modelConfig["save_weight_dir"], 'ckpt_' + str(e) + "_.pt"))
        latest_ckp_link = os.path.join(modelConfig["save_weight_dir"], 'latest.pth')
        if os.path.islink(latest_ckp_link):
            os.remove(latest_ckp_link)
        os.symlink('ckpt_' + str(e) + "_.pt", latest_ckp_link)
        

def main(model_config = None):
    modelConfig = {
        "epoch": 100,
        "batch_size": 1,
        "T": 39,
        "channel": 64,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 2.5e-6,
        "multiplier": 2.,
        "beta_1": 2.5e-5,
        "beta_T": 0.002,
        "img_size": 144,
        "crop_size": 96,
        "grad_clip": 1.,
        "device": "cuda:0",
        "training_load_weight": None,
        "data_dir": "/home/lsc/LOL/",
        "save_weight_dir": "./checkpoints/",
        "test_load_weight": "ckpt_199_.pt",
        "sampled_dir": "./results/",
        "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
        "sampledImgName": "SampledNoGuidenceImgs.png",
        "nrow": 3,
        "resume_from": "latest.pth"
        }
    if model_config is not None:
        modelConfig = model_config
    
    train(modelConfig)


if __name__ == '__main__':
    main()
