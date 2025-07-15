"""FlowMo train script."""

import contextlib
import glob
import os
import shutil
import time

import fsspec
import lpips
import torch
import torch.distributed as dist
import torch.optim as optim
from mup import MuAdam, MuAdamW
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
import wandb

from flowmo import models, perceptual_loss, train_utils

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

BFLOAT16_IS_AVAILABLE = None


def _get_norm(model, getter):
    return sum(
        (getter(p) ** 2).sum() for p in model.parameters() if p.grad is not None
    ).sqrt()


def train_step(config, model, batch, optimizer, aux_state):
    assert BFLOAT16_IS_AVAILABLE is not None
    dtype = torch.bfloat16 if BFLOAT16_IS_AVAILABLE else torch.float32

    aux = {"loss_dict": {}}
    total_loss = 0

    optimizer.zero_grad()
    b = batch["image"].shape[0]
    chunksize = b // config.opt.n_grad_acc
    batch_chunks = [
        {k: v[i * chunksize : (i + 1) * chunksize] for (k, v) in batch.items()}
        for i in range(config.opt.n_grad_acc)
    ]

    total_loss = 0.0
    assert len(batch_chunks) == config.opt.n_grad_acc
    for i, batch_chunk in enumerate(batch_chunks):
        with (
            contextlib.nullcontext()
            if i == config.opt.n_grad_acc - 1
            else model.no_sync()
        ):
            with torch.autocast(
                "cuda",
                dtype=dtype,
            ):
                loss, aux = models.rf_loss(config, model, batch_chunk, aux_state)
                loss = loss / config.opt.n_grad_acc

            loss.backward()
            total_loss += loss.detach()

    if config.opt.log_norms:
        original_grad_norm = _get_norm(model, getter=lambda p: p.grad)
        aux["loss_dict"]["debug/original_grad_norm"] = original_grad_norm
        aux["loss_dict"]["debug/param_norm"] = _get_norm(model, getter=lambda p: p)

    optimizer.step()
    return total_loss, aux


def main(args, config):
    config = train_utils.restore_config(config)
    print(torch.__version__)
    models.MUP_ENABLED = config.model.enable_mup

    train_utils.soft_init()

    rank = dist.get_rank()
    print(rank)
    dist.barrier()

    log_dir = os.path.join(args.results_dir, args.experiment_name)
    os.makedirs(log_dir, exist_ok=True)

    # Inductor cache should be on same drive as GPU to prevent "torch._inductor.exc.InductorError: CppCompileError: C++ compile error"
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(
        '/', 'tmp', f"{args.experiment_name}_torchinductor_cache_{str(rank)}"
    )

    device = rank % torch.cuda.device_count()
    print(device, torch.cuda.device_count())

    torch.cuda.set_device(device)

    global BFLOAT16_IS_AVAILABLE
    BFLOAT16_IS_AVAILABLE = (
        train_utils.bfloat16_is_available() and config.trainer.enable_bfloat16
    )
    print("Using bfloat16: ", BFLOAT16_IS_AVAILABLE)

    torch.manual_seed(0)

    model = train_utils.build_model(config)

    aux_state = {}

    n_params = sum(p.numel() for p in model.parameters())
    print(f"n_params: {n_params}")

    seed = config.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)

    model = DistributedDataParallel(model, find_unused_parameters=True)

    if config.model.enable_mup:
        if config.opt.weight_decay:
            opt_cls = MuAdamW
        else:
            opt_cls = MuAdam
    else:
        if config.opt.weight_decay:
            opt_cls = optim.AdamW
        else:
            opt_cls = optim.Adam

    encoder_pg = {
        "params": [p for (n, p) in model.named_parameters() if "encoder" in n]
    }
    decoder_pg = {
        "params": [p for (n, p) in model.named_parameters() if "decoder" in n]
    }
    assert set(encoder_pg["params"]).union(set(decoder_pg["params"])) == set(
        model.parameters()
    )

    def build_optimizer(pgs):
        optimizer = opt_cls(
            pgs,
            lr=config.opt.lr,
            weight_decay=config.opt.weight_decay,
            betas=(config.opt.beta1, config.opt.beta2),
        )
        return optimizer

    optimizer = build_optimizer([encoder_pg, decoder_pg])
    rebuilt_optimizer = False

    train_dataloader = train_utils.load_dataset(config, split='train')

    if rank == 0:
        wandb.init(
            project="flowmo",
            name=args.experiment_name,
            config=OmegaConf.to_container(config, resolve=True),
        )

    total_steps = 0

    latest_ckpt = train_utils.get_last_checkpoint(config, log_dir)
    if latest_ckpt:
        total_steps = train_utils.restore_from_ckpt(model.module, optimizer, path=latest_ckpt)
    elif args.resume_from_ckpt:
        total_steps = train_utils.restore_from_ckpt(
            model.module, optimizer, path=args.resume_from_ckpt
        )

    model_ema = train_utils.SimpleEMA(model.module, decay=config.model.ema_decay)

    tic = time.time()
    dl_iter = iter(train_utils.wrap_dataloader(train_dataloader))

    print("Training begins.")
    print(args)
    print(OmegaConf.to_yaml(config))
    if rank == 0:
        OmegaConf.save(config, os.path.join(log_dir, "config.yaml"))

    with torch.no_grad():
        if config.model.fix_initial_norms:
            if config.model.fix_norm_mode == "channel":
                norm_kwargs = dict(axis=1, keepdims=True)
            elif config.model.fix_norm_mode == "l2":
                norm_kwargs = dict()
            else:
                raise NotImplementedError

            initial_norms = {
                k: torch.linalg.norm(v, **norm_kwargs)
                for (k, v) in models.get_weights_to_fix(model)
            }
            print("Norms checksum", sum(v.sum() for v in initial_norms.values()))

    if config.opt.lpips_weight != 0.0:
        if config.opt.lpips_mode == "vgg":
            aux_state["lpips_model"] = (
                lpips.LPIPS(net="vgg").eval().requires_grad_(False).cuda()
            )
        elif config.opt.lpips_mode == "resnet":
            aux_state["lpips_model"] = (
                perceptual_loss.PerceptualLoss().eval().requires_grad_(False).cuda()
            )
        else:
            raise NotImplementedError

    running_losses = {}

    aux_state["dl_iter"] = dl_iter

    while total_steps <= config.trainer.max_steps:
        model.train()
        if config.opt.freeze_encoder or total_steps >= config.opt.freeze_encoder_after:
            if not rebuilt_optimizer:
                print(f"Rebuilding optimizer at step {total_steps}")
                optimizer = build_optimizer([decoder_pg])
                rebuilt_optimizer = True
                model.module.encoder.requires_grad_(False)
                model_ema.decay = config.model.ema_decay

        dl_tic = time.time()
        batch = next(dl_iter)
        dl_toc = time.time()
        if dl_toc - dl_tic > 1.0:
            print(f"Dataloader took {dl_toc - dl_tic} seconds!")
        images = batch["image"]

        aux_state["total_steps"] = total_steps

        loss, aux = train_step(config, model, batch, optimizer, aux_state)
        loss_dict = aux["loss_dict"]

        for k, v in loss_dict.items():
            if k in running_losses:
                running_losses[k] += v
            else:
                running_losses[k] = v

        if config.model.fix_initial_norms:
            for name, weight in models.get_weights_to_fix(model):
                weight.data = (
                    weight
                    / torch.linalg.norm(weight, **norm_kwargs)
                    * initial_norms[name]
                )

        model_ema.update(model.module, step=total_steps)

        total_steps += 1

        if total_steps == 1:
            print("first step done!")
            print(images.min(), images.max(), images.mean())

        # Refresh dataloader
        if total_steps % 10_000 == 0:
            train_dataloader = train_utils.load_dataset(config, split='train')
            dl_iter = iter(train_utils.wrap_dataloader(train_dataloader))

        if total_steps % config.trainer.log_every == 0:
            toc = time.time()
            torch.cuda.synchronize()

            steps_per_sec = config.trainer.log_every / (toc - tic)
            running_losses = {
                k: (l / config.trainer.log_every).item()
                for (k, l) in running_losses.items()
            }
            reserved_gb = torch.cuda.max_memory_reserved() / 1e9
            allocated_gb = torch.cuda.max_memory_allocated() / 1e9

            with torch.no_grad():
                encoder_checksum = sum(
                    p.mean() for p in model.module.encoder.parameters()
                ).item()
                running_losses["encoder_checksum"] = encoder_checksum

            print(
                dict(
                    memory_usage=train_utils.memory_usage(),
                    total_steps=total_steps,
                    steps_per_sec=steps_per_sec,
                    reserved_gb=reserved_gb,
                    allocated_gb=allocated_gb,
                    **running_losses,
                )
            )

            if rank == 0:
                log_data = {**running_losses, "Steps per sec": steps_per_sec}
                wandb.log(log_data, step=total_steps)

            tic = time.time()
            running_losses = dict()

        if rank == 0 and total_steps % config.trainer.checkpoint_every == 0:
            if config.trainer.gs_checkpoint_bucket:
                local_checkpoint_dir = os.path.join(log_dir, "checkpoints")
                os.makedirs(local_checkpoint_dir, exist_ok=True)
                local_checkpoint_path = os.path.join(
                    local_checkpoint_dir, f"{total_steps:08d}.pth"
                )

                # Only save if the checkpoint file doesn’t already exist
                if not os.path.exists(local_checkpoint_path):
                    torch.save(
                        {
                            "total_steps": total_steps,
                            "model_ema_state_dict": train_utils.cpu_state_dict(
                                model_ema.model
                            ),
                            "model_state_dict": train_utils.cpu_state_dict(
                                model.module
                            ),
                        },
                        local_checkpoint_path,
                    )

                gcs_checkpoint_dir = os.path.join(
                    config.trainer.gs_checkpoint_bucket, f"{log_dir}/checkpoints"
                )
                fs = fsspec.filesystem("gs")

                # E.g., gs://my-bucket/path-to-logs/checkpoints/00001000.pth
                gcs_checkpoint_path = (
                    f"{gcs_checkpoint_dir}/{os.path.basename(local_checkpoint_path)}"
                )
                if not fs.exists(gcs_checkpoint_path):
                    # Upload to GCS by streaming from local file to remote
                    with (
                        fs.open(gcs_checkpoint_path, "wb") as gcs_file,
                        open(local_checkpoint_path, "rb") as local_file,
                    ):
                        shutil.copyfileobj(local_file, gcs_file)
                os.remove(local_checkpoint_path)

                gcs_checkpoints = fs.glob(f"{gcs_checkpoint_dir}/*.pth")
                gcs_checkpoints = sorted(gcs_checkpoints)
                # Keep the two newest, plus any multiples of `keep_every`
                for ckpt in gcs_checkpoints[:-2]:
                    ckpt_step = os.path.splitext(os.path.basename(ckpt))[0]
                    if (int(ckpt_step) % config.trainer.keep_every) != 0:
                        fs.rm(ckpt)

                print("after checkpoint save:")
                print(
                    dict(
                        reserved_gb=torch.cuda.max_memory_reserved() / 1e9,
                        allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
                    )
                )
            else:
                checkpoint_dir = os.path.join(log_dir, "checkpoints")
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, "%.8d.pth" % total_steps)

                if not os.path.exists(checkpoint_path):
                    torch.save(
                        {
                            "total_steps": total_steps,
                            "model_ema_state_dict": train_utils.cpu_state_dict(
                                model_ema.model
                            ),
                            "model_state_dict": train_utils.cpu_state_dict(
                                model.module
                            ),
                        },
                        checkpoint_path,
                    )

                # Remove old checkpoints
                for checkpoint in sorted(
                    glob.glob(os.path.join(checkpoint_dir, "*.pth"))
                )[:-2]:
                    ckpt_step, _ = os.path.basename(checkpoint).split(".")
                    if (int(ckpt_step) % config.trainer.keep_every) != 0:
                        os.remove(checkpoint)

                print("after checkpoint save:")
                print(
                    dict(
                        reserved_gb=torch.cuda.max_memory_reserved() / 1e9,
                        allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
                    )
                )
    
    if rank == 0:
        wandb.finish()


if __name__ == "__main__":
    try:
        args, config = train_utils.get_args_and_config()
        main(args, config)
    finally:
        torch.distributed.destroy_process_group()
