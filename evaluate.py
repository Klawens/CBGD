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
from utils.metrics import PSNR, SSIM, PI, FID

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
            size=[modelConfig["batch_size"], 3, 32, 32], device=device)
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        save_image(saveNoisy, os.path.join(
            modelConfig["sampled_dir"], modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])
        sampledImgs = sampler(noisyImage)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        save_image(sampledImgs, os.path.join(
            modelConfig["sampled_dir"],  modelConfig["sampledImgName"]), nrow=modelConfig["nrow"])
        
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
        "test_load_weight": "latest.pt",
        "sampled_dir": "./results/",
        "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
        "sampledImgName": "SampledNoGuidenceImgs.png",
        "nrow": 3,
        "resume_from": "latest.pt"
        }
    if model_config is not None:
        modelConfig = model_config
    
    eval(modelConfig)


if __name__ == '__main__':
    main()