import argparse
import os
import torch
import json
from omegaconf import OmegaConf
import torchvision
from einops import rearrange
from transformers import GPT2LMHeadModel, LogitsProcessor, Qwen3ForCausalLM
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

from flowmo import train_utils


class FilterLogitsProcessor(LogitsProcessor):
    def __init__(self, filter_vocab_size: int):
        self.filter_vocab_size = filter_vocab_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores[:, self.filter_vocab_size:] = -float("Inf")
        return scores


def load_sft_model(ckpt_path, device):
    """Loads a saved SFT model."""
    model = GPT2LMHeadModel.from_pretrained(ckpt_path)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def generate(
    model, 
    class_idxs, 
    num_visual_tokens, 
    num_class_tokens, 
    cfg_scale=1.0, 
    temperature=1.0, 
    top_k=0, 
    top_p=1.0
):
    """Autoregressively generates visual tokens with CFG using model.generate."""
    device = next(model.parameters()).device
    
    cond_prompts = [[num_visual_tokens + c] for c in class_idxs]
    cond_tokens = torch.tensor(cond_prompts, device=device)

    negative_prompt_ids = None
    if cfg_scale > 1.0:
        uncond_prompts = [[num_visual_tokens + num_class_tokens]] * len(class_idxs)
        negative_prompt_ids = torch.tensor(uncond_prompts, device=device)

    logits_processor = [FilterLogitsProcessor(num_visual_tokens)]
    
    gen_kwargs = {
        "max_length": model.config.n_ctx,
        "guidance_scale": cfg_scale,
        "negative_prompt_ids": negative_prompt_ids,
        "logits_processor": logits_processor,
    }

    if temperature > 0:
        gen_kwargs.update({
            "do_sample": True,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        })
    else: # Greedy
        gen_kwargs["do_sample"] = False

    generated_sequences = model.generate(
        cond_tokens,
        **gen_kwargs
    )

    return generated_sequences[:, 1:]


def save_image(tensor_image: torch.Tensor, path: str):
    """Converts a [-1,1] ranged CHW tensor to a PNG file."""
    tensor_image = tensor_image.detach().cpu().clamp(-1, 1)
    tensor_image = (tensor_image + 1) / 2
    pil_image = T.ToPILImage()(tensor_image)
    pil_image.save(path)


def main():
    parser = argparse.ArgumentParser(description="Generate images from a pretrained SFT model.")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to the SFT model checkpoint directory.")
    parser.add_argument("--class-idxs", nargs='+', type=int, required=True, help="List of class indices for conditional generation.")
    parser.add_argument("--output", type=str, default="generated_image.png", help="Filename for the generated image.")
    parser.add_argument("--cfg-scale", type=float, default=1.0, help="Classifier-Free Guidance scale. Use > 1.0 for CFG.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling. Use 0 for greedy decoding.")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k filtering.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p (nucleus) filtering.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # These are based on pretrain.py and need to be consistent
    sub_codebook_size = 512
    num_visual_tokens = sub_codebook_size * 2

    # Recreate class_to_idx to get num_class_tokens
    with open('encoded_tokens_flowmo_lo.json', 'r') as f:
        data = json.load(f)
    items = [{'image_name': k} for k in data.keys()]
    class_data = {}
    for item in items:
        class_id = item['image_name'].split('_')[0]
        if class_id not in class_data:
            class_data[class_id] = []
        class_data[class_id].append(item)
    unique_classes = sorted(class_data.keys())
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    num_class_tokens = len(class_to_idx)
    
    model = load_sft_model(args.ckpt_path, device)
    
    decoder_model_name = "flowmo_lo"
    decoder_ckpth_iteration = 1325000
    config_path = f'results/{decoder_model_name}/config.yaml'
    decoder_config = OmegaConf.load(config_path)
    checkpoint_path = f"results/{decoder_model_name}/checkpoints/{decoder_ckpth_iteration:08d}.pth"
    
    decoder_model = train_utils.build_model(decoder_config)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    decoder_model.load_state_dict(state_dict['model_ema_state_dict'], strict=False)
    decoder_model.eval()
    decoder_model.to(device)

    print("Generating visual tokens...")
    visual_tokens = generate(
        model,
        args.class_idxs,
        num_visual_tokens,
        num_class_tokens,
        cfg_scale=args.cfg_scale,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )

    # print(visual_tokens)
    # print(visual_tokens.shape)
    
    print("Decoding tokens to image...")
    # De-factorize tokens
    de_factorized_tokens = visual_tokens.clone()
    de_factorized_tokens[:, 0::2] -= sub_codebook_size

    code_length = decoder_model.code_length

    # print(de_factorized_tokens)
    # print(de_factorized_tokens.shape)
    
    # Decode token ids to get the quantized vectors (-1 or 1).
    # The shape will be (batch_size, num_tokens, quantizer_dim)
    decoded_code = decoder_model.quantizer.decode(de_factorized_tokens)
    # print(decoded_code)
    # print(decoded_code.shape)
    
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
    reconstructed_images = decoder_model.reconstruct(images=torch.zeros(total_images_in_batch, 3, decoder_config.data.image_size, decoder_config.data.image_size, device=device), code=code)
    
    torchvision.utils.save_image(reconstructed_images, args.output, nrow=len(args.class_idxs), normalize=True, value_range=(-1, 1))
    print(f"Images saved to {args.output}")


if __name__ == "__main__":
    main()
