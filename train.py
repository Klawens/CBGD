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


def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    # load dataset
    dataset_high = CustomDataset(
        root_dir=modelConfig["data_dir"]+'high',
        transform=transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.Resize([modelConfig["img_size"], modelConfig["img_size"]]),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )
    dataset_low = CustomDataset(
        root_dir=modelConfig["data_dir"]+'low',
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
    if modelConfig["pretrained"]:
        decom_model.load_state_dict(torch.load(os.path.join(
            modelConfig["pretrained_weight_dir"], modelConfig["decom_pre"]), map_location=device))
        print('Pretrained decomposition model weights loaded, start fine-tuning...')
    elif modelConfig["resume"]:
        decom_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["decomp_weight"]), map_location=device))
        print('Decomposition model weights loaded, resume training...')
    else:
        print('No decomposition model weights loaded, start training...')
    optimizer_decom = torch.optim.AdamW(
        decom_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer_decom, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer_decom, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)

    # Unet model setup
    unet_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)

    if modelConfig["pretrained"]:
        decom_model.load_state_dict(torch.load(os.path.join(
            modelConfig["pretrained_weight_dir"], modelConfig["diff_pre"]), map_location=device))
        print('Pretrained diffusion model weights loaded, start fine-tuning...')
    elif modelConfig["resume"]:
        decom_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device))
        print('Diffusion model weights loaded, resume training...')
    else:
        print('No diffusion model weights loaded, start training...')
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
    # torch 2.0 compile booster
    compile_trainer = torch.compile(trainer)
    compile_decom = torch.compile(decom_model)
    
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
                ref_high, illum_high = compile_decom(x_h) # shape: batch, 3(1), h, w
                ref_low, illum_low = compile_decom(x_l)

                '''
                Gradient Scaling: It's common to scale the loss by a constant factor to control the magnitude of gradients during training. Large gradients can lead to unstable training, so scaling the loss down can help stabilize the optimization process.
                Thus "/1000." makes the gradient desent faster and stable.
                '''

                # decomposition loss = reflectance loss of 2 reflectance map, and restoration loss
                ref_loss = F.l1_loss(ref_low, ref_high, reduction='none').sum() / 1000.
                low_loss = F.l1_loss(ref_low * illum_low, x_l, reduction='none').sum() / 1000.
                high_loss = F.l1_loss(ref_high * illum_high, x_h, reduction='none').sum() / 1000.
                low_high_loss = F.l1_loss(ref_low * illum_high, x_h, reduction='none').sum() / 1000.
                decom_loss = ref_loss + low_loss + high_loss + low_high_loss

                # diffusion loss is the MSE loss between the predicted noise and the sampled noise
                # using torch 2.0 compile booster
                diff_loss, x_th, pred_noise = compile_trainer(illum_high.repeat(1, 3, 1, 1), idx, high=True)
                diff_loss = diff_loss.sum() / 1000. # Gradient Scaling


                # reconstruction loss
                rec_illu = reconstructor(x_th, reduce=True)
                rec_illu_loss = F.l1_loss(rec_illu, illum_high, reduction='none').sum() / 1000.

                rec_loss = F.l1_loss(rec_illu * ref_high, x_h, reduction='none').sum() / 1000.
                # total loss = decomposition loss + diffusion loss + reconstruction loss
                loss = decom_loss + diff_loss + rec_illu_loss + rec_loss
                loss.backward()
                idx += 1

                save_image(x_h, os.path.join(
                    modelConfig["sampled_dir"], 'high.jpg'), nrow=modelConfig["nrow"])
                save_image(x_l, os.path.join(
                    modelConfig["sampled_dir"], 'low.jpg'), nrow=modelConfig["nrow"])
                save_image(ref_high, os.path.join(
                    modelConfig["sampled_dir"], 'ref_high.jpg'), nrow=modelConfig["nrow"])
                save_image(ref_low, os.path.join(
                    modelConfig["sampled_dir"], 'ref_low.jpg'), nrow=modelConfig["nrow"])
                save_image(illum_high, os.path.join(
                    modelConfig["sampled_dir"], 'illum_high.jpg'), nrow=modelConfig["nrow"])
                save_image(illum_low, os.path.join(
                    modelConfig["sampled_dir"], 'illum_low.jpg'), nrow=modelConfig["nrow"])
                save_image(x_th, os.path.join(
                './middle/diffused/', 'diffused_high_illum.jpg'), nrow=modelConfig["nrow"])
                save_image(rec_illu, os.path.join(
                    modelConfig["sampled_dir"], modelConfig["sampledImgName"]), nrow=modelConfig["nrow"])                
                save_image(rec_illu * ref_low, os.path.join(
                    modelConfig["sampled_dir"], 'reconstructed.jpg'), nrow=modelConfig["nrow"])

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
            decom_ckp = {
                'epoch': e,
                'state_dict': decom_model.state_dict(),
                'optimizer': optimizer_decom.state_dict(),
            }
            diff_ckp = {
                'epoch': e,
                'state_dict': unet_model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            latest_decom = 'ckp_decom_' + str(e) + '.pt'
            latest_diff = 'ckpt_diff_' + str(e) + '.pt'
            torch.save(decom_ckp, os.path.join(
                modelConfig["save_weight_dir"], latest_decom))
            torch.save(diff_ckp, os.path.join(
                modelConfig["save_weight_dir"], latest_diff))
            os.symlink(os.path.join(modelConfig["save_weight_dir"], latest_decom), os.path.join(
                modelConfig["save_weight_dir"], 'decom_latest.pt'))
            os.symlink(os.path.join(modelConfig["save_weight_dir"], latest_diff), os.path.join(
                modelConfig["save_weight_dir"], 'diff_latest.pt'))


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

