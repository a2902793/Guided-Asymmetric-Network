import argparse, time
import options.options as option

def main():
    start = time.time()

    opt = {
        "name": "2e5_LH6_M6_pretrained-2",
        "use_tb_logger": True,
        "model": "DualSR_pretrain",
        "scale": 4,
        "gpu_ids": [0, 1, 2, 3],
        "datasets": {
            "train": {
                "name": "DIV2K",
                "mode": "LRHR",
                "dataroot_HR": "/home/johnny/Guided-Asymmetric-Network/DIV2K800_train_HR.lmdb",
                "dataroot_LR": "/home/johnny/Guided-Asymmetric-Network/DIV2K800_train_LR.lmdb",
                "subset_file": None,
                "use_shuffle": True,
                "n_workers": 16,
                "batch_size": 28,
                "HR_size": 128,
                "use_flip": True,
                "use_rot": True
            },
            "val": {
                "name": "val_set5_part",
                "mode": "LRHR",
                "dataroot_HR": "/home/johnny/Guided-Asymmetric-Network/Datasets/Set5/GTmod12",
                "dataroot_LR": "/home/johnny/Guided-Asymmetric-Network/Datasets/Set5/LRbicx4"
            }
        },
        "path": {
            "root": "/home/johnny/Guided-Asymmetric-Network/LCC_BasicSR-master"
        },
        "network_G": {
            "which_model_G": "DualSR_Effnet",
            "norm_type": None,
            "mode": "CNA",
            "nf": 64,
            "nb_l_1": 10,
            "nb_l_2": 5,
            "nb_h_1": 5,
            "nb_h_2": 10,
            "nb_e": 2,
            "nb_m": 5,
            "in_nc": 3,
            "out_nc": 3,
            "gc": 32,
            "group": 1,
            "low_layers": 6,
            "high_layers": 6,
            "mask_layers": 6
        },
        "network_D": {
            "which_model_D": "discriminator_vgg_128",
            "norm_type": "batch",
            "act_type": "leakyrelu",
            "mode": "CNA",
            "nf": 64,
            "in_nc": 3
        },
        "train": {
            "lr_G": 5e-5,
            "lr_scheme": "MultiStepLR",
            "lr_steps": [200000, 400000, 600000, 800000],
            "lr_gamma": 0.5,
            "pixel_criterion": "l1",
            "pixel_weight": 1.0,
            "feature_weight": 1,
            "val_freq": 1e3,
            "manual_seed": 0,
            "niter": 2e5
        },

        "logger": {
            "print_freq": 200,
            "save_checkpoint_freq": 1e3
        }
    }
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-opt', type=str, required=False, default='options/train/DualSR_pretrain.json', help='Path to option JSON file.')
    # opt = option.parse(parser.parse_args().opt, is_train=True)
    # opt = option.dict_to_nonedict(opt)  # Convert to NoneDict, which return None for missing key.

    end = time.time()
    print(f'Duration = {end - start} seconds')

    
if __name__ == '__main__':
    main()