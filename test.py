import os
from typing import Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from models.diffusion import GaussianDiffusionSampler
from models.unet import UNet
from utils.dataset import TestDataset
from utils.helper_funcs import sp_ch_3, sp_ch_1
from utils.simple_decom import DecomNet


def eval(modelConfig: Dict):
    # load model and evaluate
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"], num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
        ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        print("Diffusion weights loaded.")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
        
        # DecomNet
        Decom = DecomNet()
        decom_model = torch.load(modelConfig["decom_model"])
        Decom.load_state_dict(decom_model)
        print("DecomNet weights loaded.")
        Decom.eval()
        
        test_dataset = TestDataset(
            root_dir='./test/',
            transform=transforms.Compose([
            transforms.Resize([modelConfig["img_size"], modelConfig["img_size"]]),
            transforms.ToTensor(),
            ]))
        test_dataloader = DataLoader(
            test_dataset, batch_size=modelConfig["batch_size"], shuffle=False, num_workers=2, drop_last=True, pin_memory=True)
        # Visual
        for i, test_img in enumerate(tqdm(test_dataloader)):
            R_l, I_l = Decom(test_img[0])
            X_T = torch.cat((R_l, I_l), dim=1).cuda()
            X_0 = sampler(X_T)
            recon_img = sp_ch_3(X_0) * sp_ch_1(X_0).repeat(1, 3, 1, 1)

            if os.path.exists(modelConfig["sampled_dir"]) is False:
                os.makedirs(modelConfig["sampled_dir"])
            # save_image(sp_ch_3(X_0), os.path.join(
            #     modelConfig["sampled_dir"], '{}_reconR.jpg'.format(n)), nrow=modelConfig["nrow"])
            # save_image(sp_ch_1(X_0), os.path.join(
            #     modelConfig["sampled_dir"], '{}_reconL.jpg'.format(n)), nrow=modelConfig["nrow"])
            save_image(recon_img, os.path.join(
                modelConfig["sampled_dir"], '{}_recon.jpg'.format(test_img[1].split('/')[-1].split('.')[0])), nrow=modelConfig["nrow"])

def main(model_config = None):
    modelConfig = {
        "batch_size": 1,
        "T": 100,
        "channel": 64,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 2.5e-6,
        "multiplier": 2.,
        "beta_1": 2.5e-5,
        "beta_T": 0.002,
        "img_size": 512,
        "crop_size": 96,
        "grad_clip": 1.,
        "device": "cuda:0",
        "data_dir": "./",
        "save_weight_dir": "./checkpoints/",
        "test_load_weight": "latest.pt",
        "sampled_dir": "./test_recon/",
        "nrow": 3,
        "decom_model": "./utils/9200.tar"
        }
    if model_config is not None:
        modelConfig = model_config
    eval(modelConfig)


if __name__ == '__main__':
    main()
