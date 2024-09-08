import os
from typing import Dict
import argparse

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import PIL

from model.diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer, DDIMSampler
# from model.backbone.unet import UNet
from model.backbone.sr3 import SR3UNet
from utils.scheduler import GradualWarmupScheduler
from utils.dataset import PairedDataset, TestDataset
from utils.helper_funcs import sp_ch_3, sp_ch_1, PairRandomCrop, expand_channel
from model.losses import SmoothLossL, SSIMLoss, PSNRLoss
from utils.simple_decom import DecomNet
from utils.metrics import PSNR as PSNR_
from utils.metrics import SSIM as SSIM_
# fp16
from torch.cuda.amp import autocast, GradScaler

# evaluation metrics
ssim_eval = SSIM_().ssim_metric
psnr_eval = PSNR_().psnr_metric

ssim_loss = SSIMLoss()
psnr_loss = PSNRLoss()

def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])

    # train dataset transform
    if modelConfig["img_size"] == 0:
        tf = transforms.Compose([transforms.ToTensor()])    # original size & crop
    else:
        tf = transforms.Compose([
            transforms.Resize([modelConfig["img_size"], modelConfig["img_size"]]),  # resize & crop
            transforms.ToTensor()])
    # evaluation transform
    eval_tf = transforms.Compose([
        transforms.Resize([modelConfig["img_size"], modelConfig["img_size"]]),
        transforms.ToTensor()])
    
    # train image dataset
    paired_dataset = PairedDataset(
        root_dir_low=modelConfig["image_dir"] + modelConfig['train_path'] + 'low',
        root_dir_high=modelConfig["image_dir"] + modelConfig['train_path'] + 'high',
        transform=tf)
    paired_dataloader = DataLoader(
        paired_dataset, batch_size=modelConfig["batch_size"], shuffle=False, num_workers=8, drop_last=True, pin_memory=True)

    # components dataset
    # reflectance dataset
    paired_dataset_R = PairedDataset(
        root_dir_low=modelConfig["decoms_dir"] + modelConfig['train_path'] + 'low_r',
        root_dir_high=modelConfig["decoms_dir"] + modelConfig['train_path'] + 'high_r',
        transform=tf)
    paired_dataloader_R = DataLoader(
        paired_dataset_R, batch_size=modelConfig["batch_size"], shuffle=False, num_workers=8, drop_last=True, pin_memory=True)
    # illumination dataset
    paired_dataset_L = PairedDataset(
        root_dir_low=modelConfig["decoms_dir"] + modelConfig['train_path'] + 'low_i',
        root_dir_high=modelConfig["decoms_dir"] + modelConfig['train_path'] + 'high_i',
        transform=tf)
    paired_dataloader_L = DataLoader(
        paired_dataset_L, batch_size=modelConfig["batch_size"], shuffle=False, num_workers=8, drop_last=True, pin_memory=True)
    
    # eval dataset
    eval_dataset = TestDataset(
        root_dir=modelConfig["image_dir"] + modelConfig['test_path'] + 'low',
        transform=eval_tf)
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=True, pin_memory=True)

    # model setup
    # net_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"], num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)

    net_model = SR3UNet(inner_channel=modelConfig['channel']).to(device)

    net_model = torch.compile(net_model)

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
        net_model, modelConfig["beta"][0], modelConfig["beta"][1], modelConfig["T"]).to(device)
    trainer = torch.compile(trainer)
    # sampler = GaussianDiffusionSampler(
    #     net_model, modelConfig["beta"][0], modelConfig["beta"][1], modelConfig["T"]).to(device)
    sampler = DDIMSampler(
        net_model, modelConfig["beta"], modelConfig["T"], modelConfig["steps"], modelConfig["method"], modelConfig["eta"], modelConfig["only_return_x_0"], modelConfig["interval"]).to(device)
    sampler = torch.compile(sampler)

    # DecomNet
    Decom = DecomNet().to(device)
    decom_model = torch.load(modelConfig["decom_model"])
    Decom.load_state_dict(decom_model)
    Decom.eval()
    Decom = torch.compile(Decom)
    
    ckp_path = os.path.join(modelConfig["save_weight_dir"], modelConfig["resume_from"])
    start_epoch = 0
    if os.path.isfile(ckp_path):
        checkpoint = torch.load(ckp_path, map_location=device)
        net_model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        print("resuming from {}...".format(start_epoch))

    scaler = GradScaler()

    # start training
    for e in range(start_epoch, modelConfig["epoch"]):
        n = 0
        with tqdm(paired_dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for paired_images, paired_R, paired_L in zip(tqdmDataLoader, paired_dataloader_R, paired_dataloader_L):
                n += 1
                # TRAIN
                optimizer.zero_grad()
                # fp16
                with autocast():
                    # get low
                    low_ori = paired_images[0][:, :3, :, :].to(device)
                    # get high
                    high_ori = paired_images[0][:, 3:, :, :].to(device)
                    # get low reflectance
                    low_r = paired_R[0][:, :3, :, :].to(device)
                    # get low illumination
                    low_i = paired_L[0][:, :3, :, :].to(device)
                    # get high reflectance
                    high_r = paired_R[0][:, 3:, :, :].to(device)
                    # get high illumination
                    high_i = paired_L[0][:, 3:, :, :].to(device)
                    # crop        
                    high_ori , low_ori, high_r, high_i, low_r, low_i = PairRandomCrop(high_ori, low_ori, high_r, high_i, low_r, low_i, modelConfig["crop_size"])
                    # print(low_i[:, 0, :, :].shape, high_i.shape)
                    trainer_input = torch.cat((high_r, high_i[:, :1, :, :]), dim=1) # [b, 4, h, w]
                    sampler_input = torch.cat((low_r, low_i[:, :1, :, :]), dim=1) # [b, 4, h, w]

                    diff_loss, x_t = trainer(trainer_input)
                    diff_loss = diff_loss.sum() / 1000.
                    # fwd_image = sp_ch_3(x_t) * expand_channel(sp_ch_1(x_t))

                    recon = sampler(sampler_input)
                    rec_image = sp_ch_3(recon) * expand_channel(sp_ch_1(recon))

                    loss_color = ssim_loss(high_r, sp_ch_3(recon))

                    loss_light = SmoothLossL().smooth(sp_ch_1(recon), high_r)

                    recon_loss = ssim_loss(rec_image, high_ori) + (2 * psnr_loss(rec_image, high_ori))

                    # total loss
                    w1, w2, w3, w4 = modelConfig['loss_weights']
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
                # create path
                if e % 5 == 0 or e == modelConfig["epoch"] - 1:
                    if os.path.exists(modelConfig["sampled_dir"]+'epoch_{}/'.format(e)) is False:
                        os.makedirs(modelConfig["sampled_dir"]+'epoch_{}/'.format(e))
                    # save images
                    # save_image(v, os.path.join(
                    #     modelConfig["sampled_dir"]+'epoch_{}/'.format(e), "{}_{}.png".format(e, n)), nrow=modelConfig["nrow"])
                    # save_image(sp_ch_3(recon), os.path.join(
                    #     modelConfig["sampled_dir"]+'epoch_{}/'.format(e), "{}_{}_rvrs_ref.png".format(e, n)), nrow=modelConfig["nrow"])
                    # save_image(sp_ch_1(recon), os.path.join(
                    #     modelConfig["sampled_dir"]+'epoch_{}/'.format(e), "{}_{}_rvrs_ilu.png".format(e, n)), nrow=modelConfig["nrow"])
                    save_image(high_ori, os.path.join(
                        modelConfig["sampled_dir"]+'epoch_{}/'.format(e), "{}_{}_high.png".format(e, n)), nrow=modelConfig["nrow"])
                    # save_image(low_ori, os.path.join(
                    #     modelConfig["sampled_dir"]+'epoch_{}/'.format(e), "{}_{}_low.png".format(e, n)), nrow=modelConfig["nrow"])
                    save_image(rec_image, os.path.join(
                        modelConfig["sampled_dir"]+'epoch_{}/'.format(e), "{}_{}_recon.png".format(e, n)), nrow=modelConfig["nrow"])
                    # save_image(sp_ch_3(low_r), os.path.join(
                    #     modelConfig["sampled_dir"]+'epoch_{}/'.format(e), "{}_{}_in_ref.png".format(e, n)), nrow=modelConfig["nrow"])
                    # save_image(low_i, os.path.join(
                    #     modelConfig["sampled_dir"]+'epoch_{}/'.format(e), "{}_{}_in_ilu.png".format(e, n)), nrow=modelConfig["nrow"])
                    # save_image(high_r, os.path.join(
                    #     modelConfig["sampled_dir"]+'epoch_{}/'.format(e), "{}_{}_gt_ref.png".format(e, n)), nrow=modelConfig["nrow"])
                    # save_image(high_i, os.path.join(
                    #     modelConfig["sampled_dir"]+'epoch_{}/'.format(e), "{}_{}_gt_ilu.png".format(e, n)), nrow=modelConfig["nrow"])
                    # save_image(fwd_image, os.path.join(
                    #     modelConfig["sampled_dir"]+'epoch_{}/'.format(e), "{}_{}_fwd_nois.png".format(e, n)), nrow=modelConfig["nrow"])

                with open("log.txt", "a") as log_file:
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
                    "diff": w1 * diff_loss.item(),
                    "R": w2 * loss_color.item(),
                    "L": w3 * loss_light.item(),
                    "rec": w4 * recon_loss.item(),
                    "lr": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        
        # evaluate
        if e > 150:
            if e % 2 == 0 or e == modelConfig["epoch"] - 1:
                net_model.eval()
                with torch.no_grad():
                    ssim_score, psnr_score = [], []
                    ev = 0
                    for i, eval_img in enumerate(tqdm(eval_dataloader)):
                        img = eval_img[0].to(device)
                        R_l, I_l = Decom(img)
                        X_T = torch.cat((R_l, I_l), dim=1).to(device)
                        X_0 = sampler(X_T)
                        recon_img = sp_ch_3(X_0) * sp_ch_1(X_0).repeat(1, 3, 1, 1)
                        if os.path.exists(modelConfig["sampled_dir"]+'test_epoch_{}/'.format(e)) is False:
                            os.makedirs(modelConfig["sampled_dir"]+'test_epoch_{}/'.format(e))
                        save_image(recon_img, os.path.join(
                            modelConfig["sampled_dir"]+'test_epoch_{}/'.format(e), "{}_{}.png".format(e, ev)), nrow=1)
                        gt = PIL.Image.open(modelConfig['image_dir'] + modelConfig['test_path'] + 'high/' + eval_img[1][ev][0].split('/')[-1])

                        ssim_score.append(ssim_eval(recon_img, gt).cpu())
                        psnr_score.append(psnr_eval(recon_img, gt).cpu())

                        ev += 1
                    ssim, psnr = np.average(ssim_score), np.average(psnr_score)
                    print("SSIM: {}, PSNR: {}".format(ssim, psnr))
                    with open(modelConfig["log_name"], "a") as log_file:
                        log_file.write("epoch: {}, ".format(e))
                        log_file.write("SSIM: {}, ".format(ssim))
                        log_file.write("PSNR: {}, \n".format(psnr))

                net_model.train()
        else:
            if e % 10 == 0 or e == modelConfig["epoch"] - 1:
                net_model.eval()
                with torch.no_grad():
                    ssim_score, psnr_score = [], []
                    ev = 0
                    for i, eval_img in enumerate(tqdm(eval_dataloader)):
                        img = eval_img[0].to(device)
                        R_l, I_l = Decom(img)
                        X_T = torch.cat((R_l, I_l), dim=1).to(device)
                        X_0 = sampler(X_T)
                        recon_img = sp_ch_3(X_0) * sp_ch_1(X_0).repeat(1, 3, 1, 1)
                        if os.path.exists(modelConfig["sampled_dir"]+'test_epoch_{}/'.format(e)) is False:
                            os.makedirs(modelConfig["sampled_dir"]+'test_epoch_{}/'.format(e))
                        save_image(recon_img, os.path.join(
                            modelConfig["sampled_dir"]+'test_epoch_{}/'.format(e), "{}_{}.png".format(e, ev)), nrow=1)
                        gt = PIL.Image.open(modelConfig['image_dir'] + modelConfig['test_path'] + 'high/' + eval_img[1][ev][0].split('/')[-1])

                        ssim_score.append(ssim_eval(recon_img, gt).cpu())
                        psnr_score.append(psnr_eval(recon_img, gt).cpu())

                        ev += 1
                    ssim, psnr = np.average(ssim_score), np.average(psnr_score)
                    print("SSIM: {}, PSNR: {}".format(ssim, psnr))
                    with open(modelConfig["log_name"], "a") as log_file:
                        log_file.write("epoch: {}, ".format(e))
                        log_file.write("SSIM: {}, ".format(ssim))
                        log_file.write("PSNR: {}, \n".format(psnr))

                net_model.train()

        warmUpScheduler.step()
        # save weights
        ckp = {
            'epoch': e,
            'state_dict': net_model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }
        if os.path.exists(modelConfig["save_weight_dir"]) is False:
            os.makedirs(modelConfig["save_weight_dir"])
        torch.save(ckp, os.path.join(
            modelConfig["save_weight_dir"], 'ckpt_' + str(e) + "_.pt"))
        latest_ckp_link = os.path.join(modelConfig["save_weight_dir"], 'latest.pth')
        if os.path.islink(latest_ckp_link):
            os.remove(latest_ckp_link)
        os.symlink('ckpt_' + str(e) + "_.pt", latest_ckp_link)


def main(model_config = None, args = None):
    modelConfig = {
        "epoch": 200,
        "batch_size":1,
        "T": 500,
        "steps": 25,   # actual steps
        "method": "linear",
        "eta": 0.0, # 0.0 for ddim, 1.0 for ddpm
        "only_return_x_0": True,
        "interval": 1,
        "channel": 128, # 128
        "channel_mult": [0.5, 5, 3, 5],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.1,
        "lr": 1e-5,          # 1e-5, # 2e-4 
        "beta": [0.0001, 0.02],    # [0.0001, 0.02]
        "img_size": 512,  # 512. 320 if is SDE
        "crop_size": 144,
        "grad_clip": 1.,
        "multiplier": 2.,
        "device": "cuda:0",
        "training_load_weight": None,
        "decoms_dir": "/home/lsc/data/LLIE_decoms/SDE/indoor/",
        "image_dir": "/home/lsc/data/LLIE/SDE/indoor/",
        "train_path": "train/",
        "test_path": "test/",
        "save_weight_dir": "./checkpoints/",
        "test_load_weight": "latest.pth",
        "loss_weights": [0.5, 5, 3, 5],   # diffusion, color, illu, recon
        "sampled_dir": "./results/",
        "log_name": "eval_log.txt",
        "decom_model": "./utils/9200.tar",  # 205200.tar
        "nrow": 3,
        # "resume_from": "latest.pth"
        "resume_from": args.resume
        }

    if model_config is not None:
        modelConfig = model_config
    
    train(modelConfig)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train LLIE')
    parser.add_argument('--resume', type=str, default="latest.pth", help='model config')    # none or latest.pth or ckpt_XXX_.pt

    args = parser.parse_args()
    
    main(model_config=None, args=args)
