import os
from typing import Dict

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import PIL
import cv2

from model.diffusion import DDIMSampler
# from model.backbone.unet import UNet
from model.backbone.sr3 import SR3UNet as UNet
from utils.dataset import TestDataset
from utils.helper_funcs import sp_ch_3, sp_ch_1
from model.losses import SSIMLoss, PSNRLoss
from utils.simple_decom import DecomNet
from utils.metrics import PSNR as PSNR_
from utils.metrics import SSIM as SSIM_
from utils.metrics import LPIPS
# fp16
from torch.cuda.amp import autocast, GradScaler

# evaluation metrics
ssim_eval = SSIM_().ssim_metric
psnr_eval = PSNR_().psnr_metric
lpips_eval = LPIPS().lpips_metric

ssim_loss = SSIMLoss()
psnr_loss = PSNRLoss()

def eval(modelConfig: Dict):
    device = torch.device(modelConfig["device"])

    # evaluation transform
    eval_tf = transforms.Compose([
        transforms.Resize([modelConfig['img_size'], modelConfig['img_size']]),
        transforms.ToTensor()])
    
    # eval dataset
    eval_dataset = TestDataset(
        root_dir=modelConfig["image_dir"] + modelConfig['test_path'] + 'low',
        transform=eval_tf)
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=True, pin_memory=True)

    # model setup
    # net_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"], num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    net_model = UNet(inner_channel=modelConfig["channel"]).to(device)

    net_model = torch.compile(net_model)

    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig ["training_load_weight"]), map_location=device))

    sampler = DDIMSampler(
        net_model, modelConfig["beta"], modelConfig["T"], modelConfig["steps"], modelConfig["method"], modelConfig["eta"], modelConfig["only_return_x_0"], modelConfig["interval"]).to(device)
    sampler = torch.compile(sampler)

    # DecomNet
    Decom = DecomNet().to(device)
    decom_model = torch.load(modelConfig["decom_model"])
    Decom.load_state_dict(decom_model)
    Decom.eval()
    Decom = torch.compile(Decom)
    
    ckp_path = os.path.join(modelConfig["save_weight_dir"], modelConfig["test_load_weight"])
    if os.path.isfile(ckp_path):
        checkpoint = torch.load(ckp_path, map_location=device)
        net_model.load_state_dict(checkpoint['state_dict'])

    net_model.eval()
    with torch.no_grad():
        ssim_score, psnr_score, lpips_score = [], [], []
        ev = 0
        for i, eval_img in enumerate(tqdm(eval_dataloader)):
            if modelConfig["ensemble"] == True:
                # This test setting is the same as Retinexformer
                img = eval_img[0].to(device)
                img_hflip = torch.flip(img, (-2,))
                img_vflip = torch.flip(img, (-1,))
                img_rotate = torch.rot90(img, dims=(-2, -1))
                # make batch
                img_combo = torch.cat((img, img_hflip, img_vflip, img_rotate), dim=0)
                R_l, I_l = Decom(img_combo)
                X_T = torch.cat((R_l, I_l), dim=1).to(device)
                X_0 = sampler(X_T)
                # separate combo
                x0_ori, x0_hflip, x0_vflip, x0_rotate = torch.split(X_0, img.size(0), dim=0)
                recon_ori = sp_ch_3(x0_ori) * sp_ch_1(x0_ori).repeat(1, 3, 1, 1)
                recon_hflip = sp_ch_3(x0_hflip) * sp_ch_1(x0_hflip).repeat(1, 3, 1, 1)
                recon_vflip = sp_ch_3(x0_vflip) * sp_ch_1(x0_vflip).repeat(1, 3, 1, 1)
                recon_rotate = sp_ch_3(x0_rotate) * sp_ch_1(x0_rotate).repeat(1, 3, 1, 1)
                recon_hflip = torch.flip(recon_hflip, (-2,))
                recon_vflip = torch.flip(recon_vflip, (-1,))
                recon_rotate = torch.rot90(recon_rotate, dims=(-2, -1), k=3)
                
                recon_img = torch.stack([recon_ori, recon_hflip, recon_vflip, recon_rotate])
                recon_img = torch.mean(recon_img, dim=0)

            else:
                img = eval_img[0].to(device)
                R_l, I_l = Decom(img)
                X_T = torch.cat((R_l, I_l), dim=1).to(device)
                X_0 = sampler(X_T)
                recon_img = sp_ch_3(X_0) * sp_ch_1(X_0).repeat(1, 3, 1, 1)

                save_image(sp_ch_3(X_0), modelConfig['sampled_dir']+'sampleR.png')
                save_image(sp_ch_1(X_0), modelConfig['sampled_dir']+'sampleI.png')
            
            gt = cv2.imread(modelConfig['image_dir'] + modelConfig['test_path'] + 'high/' + eval_img[1][ev][0].split('/')[-1])
            
            if modelConfig["gt_mean"] == True:
                recon = torch.clamp(recon_img, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
                gt = gt / 255.0
                mean_recon = cv2.cvtColor(recon.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
                mean_gt = cv2.cvtColor(gt.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
                recon = np.clip(recon * (mean_gt / mean_recon), 0, 1)
                recon_img = torch.from_numpy(recon).permute(2, 0, 1).unsqueeze(0).to(device)

            if not os.path.exists(modelConfig['sampled_dir']):
                os.makedirs(modelConfig['sampled_dir'])
            save_image(recon_img, modelConfig['sampled_dir']+eval_img[1][ev][0].split('/')[-1])
            gt = PIL.Image.open(modelConfig['image_dir'] + modelConfig['test_path'] + 'high/' + eval_img[1][ev][0].split('/')[-1])
            s = ssim_eval(recon_img, gt).cpu()
            p = psnr_eval(recon_img, gt).cpu()
            l = lpips_eval(recon_img, gt)
            print("SSIM: {}, PSNR: {}, LPIPS: {}".format(s, p, l))
            ssim_score.append(s)
            psnr_score.append(p)
            lpips_score.append(l)

            ev += 1
        ssim, psnr, lpips = np.average(ssim_score), np.average(psnr_score), np.average(lpips_score)
        print("Final: SSIM: {}, PSNR: {}, LPIPS: {}".format(ssim, psnr, lpips))


def main(model_config = None):
    modelConfig = {
        "batch_size": 1,
        "T": 300,   # T
        "steps": 25,   # actual steps
        "method": "linear",
        "eta": 0.0, # 0.0 for ddim, 1.0 for ddpm
        "only_return_x_0": True,
        "interval": 1,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.1,
        "beta": [0.0001, 0.02],    # [0.0001, 0.02]
        "img_size": 512,  # 512
        "device": "cuda:0",
        "ensemble": False,  # Retinexformer setting 
        "gt_mean": False,   # common setting
        "training_load_weight": None,
        "image_dir": "./",
        "test_path": "vis/",
        "save_weight_dir": "/home/lsc/data/save_weights/Syn/",
        "test_load_weight": "ckpt_199_.pt",
        "sampled_dir": "./visualization/",
        "decom_model": "./utils/9200.tar",  # 205200.tar
        "nrow": 3,
        }

    if model_config is not None:
        modelConfig = model_config
    
    eval(modelConfig)

if __name__ == '__main__':
    main()
