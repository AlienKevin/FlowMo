from torch.optim import AdamW
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import wandb
import os
from torch.utils.data import DataLoader
from datasets import Dataset
import json
import random
from omegaconf import OmegaConf
from flowmo import train_utils
import torchvision
from einops import rearrange
import numpy as np
from transformers import LogitsProcessor
import contextlib


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FilterLogitsProcessor(LogitsProcessor):
    def __init__(self, filter_vocab_size: int):
        self.filter_vocab_size = filter_vocab_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores[:, self.filter_vocab_size:] = -float("Inf")
        return scores


class SFTTrainer:
    def __init__(self, rank, world_size, local_rank, entity='kevinxli', project_name='sft', batch_size=256, epochs=100, max_grad_norm=1.0, gradient_accumulation_steps=1):
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(self.device)

        self.ckpt_dir = f"./ckpts/{project_name}"
        if self.rank == 0:
            total_batch_size = batch_size * world_size
            self.logger = wandb.init(entity=entity, project=project_name, name=f'batch_size_{total_batch_size}_epochs_{epochs}')
        else:
            self.logger = None

        self.epochs = epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps

        with open('data/encoded_tokens_flowmo_lo.json', 'r') as f:
            data = json.load(f)

        items = [{'image_name': k, 'tokens': v} for k, v in data.items()]
        
        class_data = {}
        for item in items:
            class_id = item['image_name'].split('_')[0]
            if class_id not in class_data:
                class_data[class_id] = []
            class_data[class_id].append(item)

        unique_classes = sorted(class_data.keys())
        self.class_to_idx = {c: i for i, c in enumerate(unique_classes)}
        
        with open('data/imagenet_class_index.json', 'r') as f:
            imagenet_class_index = json.load(f)
        
        self.class_id_to_name = {}
        for _, info in imagenet_class_index.items():
            self.class_id_to_name[info[0]] = info[1]
        
        train_items = []
        val_items = []

        for class_id, class_items in class_data.items():
            random.shuffle(class_items)
            val_split_idx = int(round(len(class_items) * 0.001))
            
            val_items.extend(class_items[:val_split_idx])
            train_items.extend(class_items[val_split_idx:])

        if self.rank == 0:
            print(f'Train items: {len(train_items)}')
            print(f'Validation items: {len(val_items)}')
        
        train_dataset = Dataset.from_list(train_items)
        val_dataset = Dataset.from_list(val_items)
        
        # Calculate number of unique visual tokens from both datasets
        all_tokens = set()
        for item in train_items + val_items:
            all_tokens.update(item['tokens'])
        
        all_tokens = sorted(list(all_tokens))
        self.num_visual_tokens = len(all_tokens)
        self.num_class_tokens = len(self.class_to_idx)
        
        if self.rank == 0:
            print(f'Number of unique visual tokens: {self.num_visual_tokens}')
            print(f'Number of class tokens: {self.num_class_tokens}')
        
        # Check if the visual tokens are within the expected range
        self.sub_codebook_size = 2 ** 9
        max_token_id = max(all_tokens) if all_tokens else -1
        if max_token_id >= self.sub_codebook_size:
            raise ValueError(f"Max token ID {max_token_id} is out of range for the expected number of visual tokens {self.sub_codebook_size}.")
        
        self.num_visual_tokens = self.sub_codebook_size * 2
        if self.rank == 0:
            print(f'Number of visual tokens set to: {self.num_visual_tokens}')

        def preprocess(batch):
            input_ids_list = []

            for i in range(len(batch['tokens'])):
                class_id = batch['image_name'][i].split('_')[0]
                class_token = self.num_visual_tokens + self.class_to_idx[class_id]
                
                original_tokens = batch['tokens'][i]
                factorized_tokens = []
                for j, token in enumerate(original_tokens):
                    if j % 2 == 0: # even index
                        factorized_tokens.append(token + self.sub_codebook_size)
                    else:
                        factorized_tokens.append(token)
                
                token_ids = [class_token] + factorized_tokens
                input_ids_list.append(token_ids)

            return {
                "input_ids": input_ids_list,
            }

        self.train_dataset = train_dataset.map(preprocess, batched=True, remove_columns=train_dataset.column_names)
        self.val_dataset = val_dataset.map(preprocess, batched=True, remove_columns=val_dataset.column_names)
        self.train_dataset.set_format(type='torch', columns=['input_ids'])
        self.val_dataset.set_format(type='torch', columns=['input_ids'])

        n_ctx = 513

        if self.rank == 0:
            print(f"Setting context length to {n_ctx}")

        model = init_gpt(vocab_size=self.num_visual_tokens+self.num_class_tokens+1, n_ctx=n_ctx, device=self.device)
        if self.world_size > 1:
            self.model = DDP(model, device_ids=[self.rank])
        else:
            self.model = model

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(42)

        if self.world_size > 1:
            train_sampler = DistributedSampler(self.train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
            val_sampler = DistributedSampler(self.val_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)
            shuffle_train = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle_train = True

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, pin_memory=True, worker_init_fn=seed_worker, generator=g, sampler=train_sampler, shuffle=shuffle_train)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, pin_memory=True, worker_init_fn=seed_worker, generator=g, sampler=val_sampler)
        
        if self.rank == 0:
            print(self.train_dataset)
            print(self.val_dataset)

        # Get optimizer
        lr = 1e-4
        self.optimizer = AdamW(self.model.parameters(), lr=lr)

        # Load decoder model for generation
        decoder_model_name = "flowmo_lo"
        decoder_ckpth_iteration = 1325000
        config_path = f'results/{decoder_model_name}/config.yaml'
        self.decoder_config = OmegaConf.load(config_path)
        checkpoint_path = f"results/{decoder_model_name}/checkpoints/{decoder_ckpth_iteration:08d}.pth"
        
        self.decoder_model = train_utils.build_model(self.decoder_config)
        state_dict = torch.load(checkpoint_path, map_location='cuda')
        self.decoder_model.load_state_dict(state_dict['model_ema_state_dict'], strict=False)
        self.decoder_model.eval()
        self.decoder_model.to(self.device)

    def eval(self):
        seed_everything(42)
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating", disable=self.rank != 0):
                input_ids = batch["input_ids"].to(self.device)
                labels = input_ids.clone()

                outputs = self.model(
                    input_ids=input_ids,
                    labels=labels,
                    return_dict=True,
                )
                loss = outputs.loss
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_dataloader)
        if self.world_size > 1:
            loss_tensor = torch.tensor(avg_loss).to(self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()
        
        metrics = {"val/loss": avg_loss}

        if self.rank == 0:
            # Generation
            self.decoder_model.eval()

            num_samples_per_class = 1
            columns = ["class_id", "class_name"] + [f"sample_{i}" for i in range(num_samples_per_class)]
            table = wandb.Table(columns=columns)
            
            val_generation_class_ids = sorted(list(self.class_to_idx.keys()))

            logits_processor = [FilterLogitsProcessor(self.num_visual_tokens)]
            
            model_for_generation = self.model.module if self.world_size > 1 else self.model

            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                class_tokens = [self.num_visual_tokens + self.class_to_idx[class_id] for class_id in val_generation_class_ids]

                for i in tqdm(range(0, len(class_tokens), self.batch_size), desc="Generating images"):
                    batch_class_tokens = class_tokens[i:i+self.batch_size]
                    batch_class_ids = val_generation_class_ids[i:i+self.batch_size]
                    
                    prompts = torch.tensor(batch_class_tokens, device=self.device).unsqueeze(1)
                    attention_mask = torch.ones_like(prompts)
                    
                    generated_sequences = model_for_generation.generate(
                        prompts,
                        attention_mask=attention_mask,
                        max_length=model_for_generation.config.max_position_embeddings,
                        num_return_sequences=num_samples_per_class,
                        do_sample=True,
                        top_k=0,
                        top_p=1.0,
                        temperature=1.0,
                        logits_processor=logits_processor,
                    )
                    
                    visual_tokens = generated_sequences[:, 1:]

                    # De-factorize tokens
                    de_factorized_tokens = visual_tokens.clone()
                    de_factorized_tokens[:, 0::2] -= self.sub_codebook_size

                    code_length = self.decoder_model.code_length
                    
                    # Decode token ids to get the quantized vectors (-1 or 1).
                    # The shape will be (batch_size, num_tokens, quantizer_dim)
                    decoded_code = self.decoder_model.quantizer.decode(de_factorized_tokens)
                    
                    # Reshape to match the structure before the final rearrange in _quantize
                    # The shape of `quantized` in _quantize after quantizer call is (b, d, ...)
                    # so we permute (b, n, d) to (b, d, n)
                    decoded_code = decoded_code.permute(0, 2, 1)

                    # Rearrange back to (b, code_length, context_dim)
                    # 'b d (t fh) -> b t (d fh)'
                    # b = batch size
                    # d = quantizer_dim
                    # t = code_length
                    # fh = (num_tokens / code_length) = (context_dim / quantizer_dim)
                    code = rearrange(
                        decoded_code,
                        'b d (t fh) -> b t (d fh)',
                        t=code_length
                    )

                    total_images_in_batch = visual_tokens.shape[0]
                    reconstructed_images = self.decoder_model.reconstruct(images=torch.zeros(total_images_in_batch, 3, self.decoder_config.data.image_size, self.decoder_config.data.image_size).cuda(), code=code)
                    reconstructed_images = (reconstructed_images + 1.0) / 2.0
                    reconstructed_images.clamp_(0, 1)

                    img_idx = 0
                    for class_id in batch_class_ids:
                        class_name = self.class_id_to_name[class_id]
                        row = [class_id, class_name]
                        for _ in range(num_samples_per_class):
                            image = torchvision.transforms.ToPILImage()(reconstructed_images[img_idx])
                            row.append(wandb.Image(image))
                            img_idx += 1
                        table.add_data(*row)
            
            metrics["val/generations"] = table

        return metrics
    def train(self):
        eval_metrics = self.eval()
        if self.rank == 0:
            self.logger.log(eval_metrics)
        if self.world_size > 1:
            dist.barrier()
        
        for epoch in range(self.epochs):
            self.model.train()
            if self.world_size > 1:
                self.train_dataloader.sampler.set_epoch(epoch)
            epoch_loss = 0
            self.optimizer.zero_grad()
            for idx, batch in enumerate(tqdm(self.train_dataloader, disable=self.rank != 0)):
                is_sync_step = (idx + 1) % self.gradient_accumulation_steps == 0 or (idx + 1) == len(self.train_dataloader)
                use_no_sync = self.world_size > 1 and not is_sync_step
                context = self.model.no_sync() if use_no_sync else contextlib.nullcontext()

                with context:
                    input_ids = batch["input_ids"].to(self.device)
                    # Randomly replace first class token with null token (num_classes) with probability 0.1
                    batch_size = input_ids.shape[0]
                    # Create random mask for class token dropout
                    dropout_mask = torch.rand(batch_size, device=input_ids.device) < 0.1
                    # Replace first token (class token) with null token where mask is True
                    # # null token is at the end of class tokens
                    input_ids[dropout_mask, 0] = self.num_visual_tokens + self.num_class_tokens
                    labels = input_ids.clone()

                    # print(f"input_ids max: {input_ids.max().item()}, min: {input_ids.min().item()}")

                    outputs = self.model(
                        input_ids=input_ids,
                        labels=labels,
                        return_dict=True,
                    )

                    # Compute loss
                    loss = outputs.loss
                    loss = loss / self.gradient_accumulation_steps
                
                micro_batch_loss = loss.item() * self.gradient_accumulation_steps
                epoch_loss += micro_batch_loss

                # Backward step
                loss.backward()

                if is_sync_step:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm,
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.rank == 0:
                        self.logger.log({"epoch": epoch, "train/loss": micro_batch_loss, "train/grad_norm": grad_norm.item()})

            if self.rank == 0:
                if (epoch == 0) or ((epoch + 1) % 10 == 0) or (epoch == self.epochs - 1):
                    model_save_path = os.path.join(self.ckpt_dir, f'sft_epoch_{epoch}')
                    model_to_save = self.model.module if self.world_size > 1 else self.model
                    save_model(model_to_save, model_save_path)

            eval_metrics = self.eval()
            
            if self.world_size > 1:
                epoch_loss_tensor = torch.tensor(epoch_loss).to(self.device)
                dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.AVG)
                epoch_loss = epoch_loss_tensor.item()

            if self.rank == 0:
                eval_metrics.update({"epoch": epoch, "train/epoch_loss": epoch_loss})
                self.logger.log(eval_metrics)
            
            if self.world_size > 1:
                dist.barrier()

        if self.rank == 0:
            self.logger.finish()

def init_gpt(vocab_size, n_ctx=257, device='cuda:0'):
    from transformers import Qwen3ForCausalLM, AutoConfig
    config = AutoConfig.from_pretrained(
        "Qwen/Qwen3-0.6B-Base",
        vocab_size=vocab_size,
        max_position_embeddings=n_ctx,
        use_cache=False,
    )
    model = Qwen3ForCausalLM(config).to(device)
    model_size = sum(t.numel() for t in model.parameters())
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"GPT size: {model_size/1000**2:.1f}M parameters")
    return model


def save_model(model, model_save_path) -> None:
    os.makedirs(model_save_path, exist_ok=True)
    model.save_pretrained(model_save_path)


def setup():
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()


def main_worker(project_name, epochs, batch_size, gradient_accumulation_steps):
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        # It's important to set the device before initializing the SFTTrainer
        # because SFTTrainer uses the device to move the model to the correct GPU.
        # torchrun will set LOCAL_RANK for each process.
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        setup()
        rank = dist.get_rank()
    else:
        rank = 0
        local_rank = 0

    seed_everything(42)

    sft_trainer = SFTTrainer(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        entity='kevinxli',
        project_name=project_name,
        epochs=epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps
    )
    sft_trainer.train()
    
    if world_size > 1:
        cleanup()


if __name__ == "__main__":
    project_name = 'imagenet_gpt2'
    epochs = 300
    total_batch_size = 256
    gradient_accumulation_steps = 1

    # Rely on environment variables set by the launcher (e.g., torchrun)
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    effective_batch_size = total_batch_size // gradient_accumulation_steps

    if world_size > 1:
        print(f"Using {world_size} GPUs for DDP training.")
        per_gpu_batch_size = effective_batch_size // world_size
    else:
        print("Running on a single GPU.")
        per_gpu_batch_size = effective_batch_size
    
    main_worker(project_name, epochs, per_gpu_batch_size, gradient_accumulation_steps)
