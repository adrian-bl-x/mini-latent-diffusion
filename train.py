# train.py
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from transformers import CLIPTokenizer, CLIPModel
from diffusers import AutoencoderKL

# Import elements from our models definition script
from models import TrainConfig, Flickr30kParquetDataset, PowerfulUNetDenoiser, DDPMScheduler, get_device


def drop_text_condition(text_tokens: torch.Tensor, drop_prob: float) -> torch.Tensor:
    if drop_prob == 0:
        return text_tokens
    mask = torch.bernoulli(torch.full((text_tokens.shape[0], 1, 1), 1.0 - drop_prob, device=text_tokens.device))
    return text_tokens * mask


def train_ldm():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    cfg = TrainConfig(num_epochs=args.epochs)
    device = get_device()

    # Set PyTorch environment optimizations for modern GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print(f"WSL2 Computing Node Active. Device: {device}")

    # 1. Load Pretrained Components
    print("Loading Frozen SD-VAE...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()
    for p in vae.parameters(): p.requires_grad = False

    print("Loading Frozen CLIP Text Encoder...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = clip_model.text_model.to(device)
    text_encoder.eval()
    for p in text_encoder.parameters(): p.requires_grad = False

    # 2. Setup Data Pipelines
    print("Loading Flickr30 Dataset...")
    ds = Flickr30kParquetDataset("train", cfg.image_size, tokenizer, max_length=cfg.max_length)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True,
                        drop_last=True)

    # 3. Model Engine Init
    denoiser = PowerfulUNetDenoiser(in_channels=cfg.latent_channels, text_dim=cfg.text_dim,
                                    model_channels=cfg.base_ch).to(device)
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=cfg.lr, weight_decay=1e-2)
    scheduler = DDPMScheduler(timesteps=cfg.timesteps, device=device)
    scaler = torch.amp.GradScaler('cuda')

    print(f"\nTraining Isolated U-Net: Size={cfg.image_size}, Channels={cfg.base_ch}, Batch={cfg.batch_size}")

    for epoch in range(cfg.num_epochs):
        denoiser.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{cfg.num_epochs}")
        for images, input_ids, _, _ in pbar:
            images, input_ids = images.to(device, non_blocking=True), input_ids.to(device, non_blocking=True)

            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample() * 0.18215
                text_tokens = text_encoder(input_ids=input_ids).last_hidden_state

            t = scheduler.sample_timesteps(images.shape[0])
            x_t, noise = scheduler.add_noise(latents, t)
            text_tokens = drop_text_condition(text_tokens, cfg.guidance_drop_prob)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                pred_noise = denoiser(x_t, t, text_tokens)
                loss = F.mse_loss(pred_noise, noise)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        if (epoch + 1) % cfg.save_every == 0 or (epoch + 1) == cfg.num_epochs:
            os.makedirs(cfg.out_dir, exist_ok=True)
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': denoiser.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict()
            }
            torch.save(checkpoint, os.path.join(cfg.out_dir, f"ldm_checkpoint_epoch_{epoch + 1}.pt"))
            print(f" [Saved Full Checkpoint at Epoch {epoch + 1}]")

    print("Training finished successfully!")


if __name__ == "__main__":
    train_ldm()
