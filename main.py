from train import train, eval
import torch


def main(model_config = None):
    modelConfig = {
        "state": "train", # train or eval
        "epoch": 501,
        "batch_size": 6,
        "T": 50,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 5e-5,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:0",
        "data_dir": "/home/klawens/LOL/",
        "pretrained": False,
        "pretrained_weight_dir": "./pretrained/",
        "decom_pre": "decom_pretrained.pt",
        "diff_pre": "diff_pretrained.pt",
        "resume": False,
        # "training_load_weight": "diff_latest.pt",
        "training_load_weight": None,
        # "decomp_weight": "decom_latest.pt",
        "decomp_weight": None,
        "save_weight_dir": "./checkpoints/",
        "test_load_weight": "ckpt_1000_.pt",
        "sampled_dir": "./SampledImgs/",
        "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
        "sampledImgName": "rec_illum.png",
        "nrow": 3
        }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    main()
