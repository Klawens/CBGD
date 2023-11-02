from train import train, eval


def main(model_config = None):
    modelConfig = {
        "state": "train", # train or eval
        "epoch": 201,
        "batch_size": 1,
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
        "img_size": 64,
        "grad_clip": 1.,
        "device": "cuda:0",
        "pretrained_weight_dir": "./pretrained/",
        # "training_load_weight": "DiffusionWeight.pt",
        "training_load_weight": None,
        # "decomp_weight": "ckp_decom_0.pt",
        "decomp_weight": None,
        "save_weight_dir": "./checkpoints/",
        "test_load_weight": "ckpt_199_.pt",
        "sampled_dir": "./SampledImgs/",
        "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
        "sampledImgName": "SampledNoGuidenceImgs.png",
        "nrow": 2
        }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    main()
