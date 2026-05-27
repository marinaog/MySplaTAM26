import argparse
import os
import sys
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

import numpy as np
import torch

from datasets.gradslam_datasets import (load_dataset_config, ICLDataset, ReplicaDataset, ReplicaV2Dataset,
                                        AzureKinectDataset, ScannetDataset, Ai2thorDataset, Record3DDataset,
                                        RealsenseDataset, TUMDataset, ScannetPPDataset, NeRFCaptureDataset,
                                        RawSLAMDataset)
from utils.eval_helpers import eval


def get_dataset(config_dict, basedir, sequence, raw=False, **kwargs):
    if config_dict["dataset_name"].lower() in ["icl"]:
        return ICLDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replica"]:
        return ReplicaDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replicav2"]:
        return ReplicaV2Dataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["azure", "azurekinect"]:
        return AzureKinectDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannet"]:
        return ScannetDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["ai2thor"]:
        return Ai2thorDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["record3d"]:
        return Record3DDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["realsense"]:
        return RealsenseDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["tum"]:
        return TUMDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannetpp"]:
        return ScannetPPDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["nerfcapture"]:
        return NeRFCaptureDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["rawslam"]:
        return RawSLAMDataset(config_dict, basedir, sequence, raw=raw, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {config_dict['dataset_name']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=str,
                        help="Path to experiment directory (must contain config.py and params.npz)")
    parser.add_argument("--eval_every", type=int, default=1,
                        help="Evaluate every nth frame (default: 1 = all frames)")
    parser.add_argument("--save_frames", action="store_true", default=False,
                        help="Save rendered RGB and depth frames")
    args = parser.parse_args()

    exp_dir = args.experiment_dir
    config_path = os.path.join(exp_dir, "config.py")
    params_path = os.path.join(exp_dir, "params.npz")

    if not os.path.exists(config_path):
        print(f"Skipping {exp_dir}: config.py not found.")
        sys.exit(0)
    if not os.path.exists(params_path):
        checkpoints = sorted(
            [f for f in os.listdir(exp_dir) if f.startswith("params") and f.endswith(".npz") and f != "params.npz"],
            key=lambda f: int(f[len("params"):-len(".npz")])
        )
        if not checkpoints:
            print(f"Skipping {exp_dir}: no params.npz or checkpoint found.")
            sys.exit(0)
        params_path = os.path.join(exp_dir, checkpoints[-1])
        print(f"params.npz not found, using latest checkpoint: {checkpoints[-1]}")

    experiment = SourceFileLoader(os.path.basename(config_path), config_path).load_module()
    config = experiment.config

    device = torch.device(config["primary_device"])
    raw = config.get("raw", False)
    use_mlp = config.get("use_mlp", False)

    # Load dataset
    print("Loading dataset ...")
    dataset_config = config["data"]
    if "gradslam_data_cfg" not in dataset_config:
        gradslam_data_cfg = {"dataset_name": dataset_config["dataset_name"]}
    else:
        gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])

    dataset_config.setdefault("ignore_bad", False)
    dataset_config.setdefault("use_train_split", True)

    dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        device=device,
        relative_pose=True,
        ignore_bad=dataset_config["ignore_bad"],
        use_train_split=dataset_config["use_train_split"],
        raw=raw,
    )
    num_frames = dataset_config["num_frames"]
    if num_frames == -1:
        num_frames = len(dataset)

    # Load params
    print("Loading params ...")
    params = dict(np.load(params_path, allow_pickle=True))
    params = {k: torch.tensor(params[k]).cuda().float() for k in params.keys()}

    # Load MLP if present
    variables = {}
    mlp_path = os.path.join(exp_dir, "mlp.pt")
    if use_mlp and os.path.exists(mlp_path):
        from utils.slam_helpers import TinyColorMLP
        variables['color_mlp'] = TinyColorMLP().cuda()
        variables['color_mlp'].load_state_dict(torch.load(mlp_path))
        print("Loaded MLP weights.")

    eval_dir = os.path.join(exp_dir, "eval_rerun")
    print(f"Saving results to: {eval_dir}")

    with torch.no_grad():
        eval(
            dataset, params, num_frames, eval_dir,
            sil_thres=config['mapping']['sil_thres'],
            mapping_iters=config['mapping']['num_iters'],
            add_new_gaussians=config['mapping']['add_new_gaussians'],
            eval_every=5,
            save_frames=args.save_frames,
            raw=raw,
            variables=variables if variables else None,
            save_plots=False,
        )
