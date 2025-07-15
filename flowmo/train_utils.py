import argparse
import contextlib
import copy
import glob
import os
import subprocess
import tempfile
import time

import fsspec
import psutil
import torch
import torch.distributed as dist
from mup import MuReadout, set_base_shapes
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from flowmo import data, models


def get_args_and_unknown():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default="my_experiment")
    parser.add_argument("--resume-from-ckpt", type=str, default="")
    
    # arguments for slurm (unused in code release)
    parser.add_argument("--account", type=str, default="")
    parser.add_argument("--num-tasks-per-node", type=int, default=4)
    parser.add_argument("--slurm-partition", type=str, default="")
    parser.add_argument("--nodelist", type=str, default="")
    parser.add_argument("--exclude", type=str, default="")
    parser.add_argument("--mem-per-task-gb", type=int, default=64)
    parser.add_argument("--cpus-per-task", type=int, default=8)
    parser.add_argument("--eval-num-tasks-per-node", type=int, default=1)
    parser.add_argument("--eval-slurm-partition", type=str, default="")
    parser.add_argument("--eval-nodelist", type=str, default="")
    parser.add_argument("--eval-exclude", type=str, default="")
    parser.add_argument("--eval-mem-per-task-gb", type=int, default=64)
    parser.add_argument("--eval-cpus-per-task", type=int, default=4)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--slurm-qos", type=str, default="batch")
    parser.add_argument("--eval-slurm-qos", type=str, default="batch")
    parser.add_argument(
        "--modes",
        type=str,
        default="train",
        help="could be train or train,eval or eval",
    )
    args, unknown = parser.parse_known_args()
    return args, unknown


def restore_config(config):
    new_config = OmegaConf.load("flowmo/configs/base.yaml")
    config = OmegaConf.merge(new_config, config)
    return config


def get_args_and_config():
    args, unknown = get_args_and_unknown()

    config = OmegaConf.load("flowmo/configs/base.yaml")
    OmegaConf.set_struct(config, True)
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(config, cli)
    return args, config


def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())


def bfloat16_is_available():
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            torch.ones((1,), device="cuda")
        return True
    except Exception as e:
        print("BFloat16 is NOT available!")
        print(e)
        return False


def wrap_dataloader(dataloader):
    while True:
        for batch in dataloader:
            new_batch = {
                k: v.permute(0, 3, 1, 2).to("cuda")
                for (k, v) in batch.items()
                if k
                in [
                    "image",
                ]
            }
            yield new_batch


def load_dataset(config, split, shuffle_val=False):
    match config.data.imagenet_data_source:
        case "local":
            data_source = data.IndexedTarDataset
            shuffle_train = True
            shuffle_val = shuffle_val
        case "wds":
            data_source = data.ImageNetWebDataset
            shuffle_train = False
            shuffle_val = False
    if split == "train":
        dataset = data_source(
            config.data.imagenet_train_tar,
            config.data.imagenet_train_index,
            size=config.data.image_size,
            random_crop=True,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            shuffle=shuffle_train,
            pin_memory=True,
            drop_last=True,
        )
        dataloader.sampler.replacement = True
        return dataloader
    elif split == "val":
        dataset = data_source(
            config.data.imagenet_val_tar,
            config.data.imagenet_val_index,
            size=config.data.image_size,
            random_crop=False,
        )
        return DataLoader(
            dataset,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            shuffle=shuffle_val,
            pin_memory=True,
            drop_last=False,
        )
    else:
        raise NotImplementedError


def memory_usage():
    process = psutil.Process()
    mem = process.memory_info().rss / 1024**3
    return mem


@contextlib.contextmanager
def record_function(name):
    torch.cuda.synchronize()
    tic = time.time()
    yield
    torch.cuda.synchronize()
    toc = time.time()
    print(f"{name}, {toc - tic} seconds")


def soft_init(timeout=None, requeue=False):
    try:
        dist.init_process_group(
            "nccl",
            timeout=timeout,
        )
    except Exception as e:
        print(e)
        if requeue:
            subprocess.run(["scontrol", "requeue", os.environ["SLURM_JOB_ID"]])
            print("Waiting...")
            time.sleep(30)
    dist.barrier()


def build_model(config):
    with tempfile.TemporaryDirectory() as log_dir:
        models.MUP_ENABLED = config.model.enable_mup
        model_partial = models.FlowMo

        shared_kwargs = dict(config=config)
        model = model_partial(
            **shared_kwargs,
            width=config.model.mup_width,
        ).cuda()

        if config.model.enable_mup:
            print("Mup enabled!")
            with torch.device("cpu"):
                base_model = model_partial(
                    **shared_kwargs, width=config.model.mup_width
                )
                delta_model = model_partial(
                    **shared_kwargs,
                    width=config.model.mup_width * 4
                    if config.model.mup_width == 1
                    else 1,
                )
                true_model = model_partial(
                    **shared_kwargs, width=config.model.mup_width
                )

                if torch.distributed.is_initialized():
                    bsh_path = os.path.join(log_dir, f"{dist.get_rank()}.bsh")
                else:
                    bsh_path = os.path.join(log_dir, "0.bsh")
                set_base_shapes(
                    true_model, base_model, delta=delta_model, savefile=bsh_path
                )

            model = set_base_shapes(model, base=bsh_path)

            for module in model.modules():
                if isinstance(module, MuReadout):
                    module.width_mult = lambda: module.weight.infshape.width_mult()
    return model


def load_state_dict(path):
    if path.startswith("gs"):
        fs = fsspec.filesystem("gs")
        with fs.open(path, "rb") as gs_file:
            state_dict = torch.load(gs_file, map_location="cpu")
    else:
        state_dict = torch.load(path, map_location="cpu")
    return state_dict


@torch.no_grad()
def restore_from_ckpt(model, optimizer, path):
    print("Restoring from checkpoint!", path)
    state_dict = load_state_dict(path)
    model.load_state_dict(state_dict["model_state_dict"])

    if "optimizer_state_dict" in state_dict:
        print("Not loading optimizer state dict. TODO: fix memory issue.")
        del state_dict["optimizer_state_dict"]

    total_steps = state_dict["total_steps"]
    return total_steps


def cpu_state_dict(model):
    return {k: v.cpu() for k, v in model.state_dict().items()}


def get_last_checkpoint(config, logdir):
    if config.trainer.gs_checkpoint_bucket:
        fs = fsspec.filesystem("gs")
        all_checkpoints = fs.glob(
            os.path.join(
                config.trainer.gs_checkpoint_bucket, logdir, "checkpoints/*.pth"
            )
        )
        # fsspec strips prefix for some weird reason...
        all_checkpoints = ["gs://" + c for c in all_checkpoints]
        all_checkpoints_sorted = sorted(all_checkpoints)
    else:
        all_checkpoints_sorted = sorted(
            glob.glob(os.path.join(logdir, "checkpoints", "*.pth"))
        )

    if all_checkpoints_sorted:
        return all_checkpoints_sorted[-1]
    else:
        return None


class SimpleEMA:
    @torch.no_grad()
    def __init__(self, model, decay=0.9999, update_freq=10):
        self.model = copy.deepcopy(model)
        self.decay = decay
        self.update_freq = update_freq

    @torch.no_grad()
    def update(self, in_model, step):
        if step % self.update_freq == 0:
            in_state_dict = in_model.state_dict()
            for key, value in self.model.state_dict().items():
                value.copy_(
                    value * (self.decay**self.update_freq)
                    + in_state_dict[key] * (1 - (self.decay**self.update_freq))
                )
