import glob
import os
from typing import Dict

import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from model.decomp_cnn import DecomNet
# from model.decomp_transformer import DecomNet 
from model.diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from model.unet import UNet
from utils.scheduler import GradualWarmupScheduler
from utils.dataset import CustomDataset
import copy
from PIL import Image
import cv2


def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    # load dataset
    dataset_high = CustomDataset(
        root_dir='/home/klawens/LOL/LOL/our485/high',
        transform=transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.Resize([modelConfig["img_size"], modelConfig["img_size"]]),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )
    dataset_low = CustomDataset(
        root_dir='/home/klawens/LOL/LOL/our485/low',
        transform=transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.Resize([modelConfig["img_size"], modelConfig["img_size"]]),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )
    dataloader_high = DataLoader(
        dataset_high, batch_size=modelConfig["batch_size"], shuffle=False, num_workers=2, drop_last=True, pin_memory=True)
    dataloader_low = DataLoader(
        dataset_low, batch_size=modelConfig["batch_size"], shuffle=False, num_workers=2, drop_last=True, pin_memory=True)
    
    for i, j in zip(dataloader_high, dataloader_low):
        print(i.shape, j.shape)
        break
    
    # DecomNet model setup  
    decom_model = DecomNet().to(device)
    if modelConfig["decomp_weight"] is not None:
        decom_model.load_state_dict(torch.load(os.path.join(
            modelConfig["pretrained_weight_dir"], modelConfig["decomp_weight"]), map_location=device))
        print('decomposition model weights loaded.')
    else:
        print('No decomposition model weights loaded.')
    optimizer_decom = torch.optim.AdamW(
        decom_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer_decom, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer_decom, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)

    # Unet model setup
    unet_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    if modelConfig["training_load_weight"] is not None:
        unet_model.load_state_dict(torch.load(os.path.join(
            modelConfig["pretrained_weight_dir"], modelConfig["training_load_weight"]), map_location=device))
        print('diffusion model weights loaded.')
    else:
        print('No diffusion model weights loaded.')
    optimizer = torch.optim.AdamW(
        unet_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        unet_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
    reconstructor = GaussianDiffusionSampler(
        unet_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
    
    # print model parameters
    model_size = 0
    decom_size = 0
    unet_size = 0
    for param in unet_model.parameters():
        unet_size += param.data.nelement()
    for param in decom_model.parameters():
        decom_size += param.data.nelement()
    model_size = unet_size + decom_size
    print('Decomposite params: %.2f M' % (decom_size / 1024 / 1024))
    print('Diffusion params: %.2f M' % (unet_size / 1024 / 1024))
    print('Total params: %.2f M' % (model_size / 1024 / 1024))

    # start training
    for e in range(modelConfig["epoch"]):
        idx = 0
        with tqdm(dataloader_high, dynamic_ncols=True) as tqdmDataLoader:
            # for images, labels in tqdmDataLoader:
            for high_images, low_images in zip(tqdmDataLoader, dataloader_low):
                # train
                optimizer.zero_grad()
                # original high/low images, worked
                x_h = high_images.to(device)
                x_l = low_images.to(device)

                # decomposite
                ref_high, illum_high = decom_model(x_h) # shape: batch, 3(1), h, w
                ref_low, illum_low = decom_model(x_l)
                cv2.imwrite('ref_high.jpg', ref_high.permute(0,2,3,1).detach().cpu().numpy()[0, :, :, :]*255)
                cv2.imwrite('ref_low.jpg', ref_low.permute(0,2,3,1).detach().cpu().numpy()[0, :, :, :]*255)
                cv2.imwrite('illum_high.jpg', illum_high.permute(0,2,3,1).detach().cpu().numpy()[0, :, :, :]*255)
                cv2.imwrite('illum_low.jpg', illum_low.permute(0,2,3,1).detach().cpu().numpy()[0, :, :, :]*255)

                # decomposition loss = reflectance loss of 2 reflectance map, and restoration loss 
                decom_loss = F.mse_loss(ref_low, ref_high, reduction='none').sum() / 1000. + F.l1_loss(ref_high * illum_high, x_h, reduction='none').sum() / 1000. + F.l1_loss(ref_low * illum_low, x_l, reduction='none').sum() / 1000.# + F.l1_loss(ref_low * illum_high, x_h)

                '''
                Gradient Scaling: It's common to scale the loss by a constant factor to control the magnitude of gradients during training. Large gradients can lead to unstable training, so scaling the loss down can help stabilize the optimization process.
                Thus "/1000." makes the gradient desent faster and stable.
                '''
                # diffusion loss is the MSE loss between the predicted noise and the sampled noise
                diff_loss, x_th, pred_noise = trainer(illum_high.repeat(1, 3, 1, 1), idx, high=True)
                diff_loss = diff_loss.sum() / 1000. # Gradient Scaling

                save_image(x_th, os.path.join(
                './middle/diffused/', 'diff.jpg'), nrow=modelConfig["nrow"])

                # reconstruction loss
                rec = reconstructor(x_th)
                save_image(rec, os.path.join(
                    modelConfig["sampled_dir"], modelConfig["sampledImgName"]), nrow=modelConfig["nrow"])
                # total loss = decomposition loss + diffusion loss + reconstruction loss
                rec_loss = F.l1_loss(rec, x_h, reduction='none').sum() / 1000.
                loss = decom_loss + diff_loss + rec_loss
                loss.backward()
                idx += 1
                high = x_th.permute(0, 2, 3, 1).detach().cpu().numpy() * ref_high.permute(0, 2, 3, 1).detach().cpu().numpy() * 255
                cv2.imwrite('high.jpg', high[0])

                torch.nn.utils.clip_grad_norm_(
                    unet_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_h.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        if e % 10 == 0:
            torch.save(decom_model.state_dict(), os.path.join(
                modelConfig["save_weight_dir"], 'ckp_decom_' + str(e) + ".pt"))
            torch.save(unet_model.state_dict(), os.path.join(
                modelConfig["save_weight_dir"], 'ckpt_' + str(e) + ".pt"))


def eval(modelConfig: Dict):
    # load model and evaluate
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
        ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
        # Sampled from standard normal distribution
        noisyImage = torch.randn(
            size=[modelConfig["batch_size"], 3, modelConfig["img_size"], modelConfig["img_size"]], device=device)
        noisyImage = -torch.abs(noisyImage) # black noise
        # saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        saveNoisy = torch.clamp(noisyImage * 0.5 - 0.5, -1, 0)
        save_image(saveNoisy, os.path.join(
            modelConfig["sampled_dir"], modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])
        sampledImgs = sampler(noisyImage)
        # sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        sampledImgs = sampledImgs * 0.5 - 0.5  # [-1 ~ 0]
        save_image(sampledImgs, os.path.join(
            modelConfig["sampled_dir"],  modelConfig["sampledImgName"]), nrow=modelConfig["nrow"])

