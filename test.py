import os
import torch
from transformers import CLIPTokenizer, CLIPModel
from diffusers import AutoencoderKL
from torchvision.utils import save_image

from models import TrainConfig, PowerfulUNetDenoiser, DDPMScheduler, get_device


def sample_images(checkpoint_path: str, prompt: str, guidance_scale: float = 7.5, output_name: str = "output.png"):
    device = get_device()
    cfg = TrainConfig()

    print(f"Loading sampling frameworks onto: {device}...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = clip_model.text_model.to(device)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

    # Initialize blank denoiser matching our custom configurations
    denoiser = PowerfulUNetDenoiser(in_channels=cfg.latent_channels, text_dim=cfg.text_dim,
                                    model_channels=cfg.base_ch).to(device)

    print(f"Reloading trained checkpoint weights: {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    denoiser.load_state_dict(checkpoint['model_state_dict'])

    denoiser.eval()
    vae.eval()
    scheduler = DDPMScheduler(timesteps=cfg.timesteps, device=device)

    # Prepare Classifier-Free Guidance inputs (Prompt text vs empty text)
    text_inputs = tokenizer(prompt, padding="max_length", truncation=True, max_length=cfg.max_length,
                            return_tensors="pt")
    uncond_inputs = tokenizer("", padding="max_length", truncation=True, max_length=cfg.max_length, return_tensors="pt")

    with torch.no_grad():
        cond_embeddings = text_encoder(text_inputs.input_ids.to(device)).last_hidden_state
        uncond_embeddings = text_encoder(uncond_inputs.input_ids.to(device)).last_hidden_state

    # 512 image resolution maps cleanly to a 64x64 compressed VAE matrix representation
    latent_size = cfg.image_size // 8
    latents = torch.randn(1, cfg.latent_channels, latent_size, latent_size, device=device)

    print(f"Running reverse DDPM loop ({cfg.timesteps} timesteps)...")
    for t in reversed(range(cfg.timesteps)):
        t_tensor = torch.tensor([t], device=device)

        with torch.no_grad():
            # Get guesses from both paths
            with torch.amp.autocast('cuda'):
                noise_pred_cond = denoiser(latents, t_tensor, cond_embeddings)
                noise_pred_uncond = denoiser(latents, t_tensor, uncond_embeddings)

            # Compute guided direction extrapolation step
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # Scheduler Step to move matrix from step t -> t-1
            alpha = scheduler.alphas[t]
            alpha_cumprod = scheduler.alphas_cumprod[t]
            beta = scheduler.betas[t]

            noise = torch.randn_like(latents) if t > 0 else 0
            latents = (1 / torch.sqrt(alpha)) * (
                        latents - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * noise_pred) + torch.sqrt(beta) * noise

    print("Decoding latent map through VAE...")
    with torch.no_grad():
        latents = latents / 0.18215
        decoded_image = vae.decode(latents).sample
        decoded_image = (decoded_image / 2 + 0.5).clamp(0, 1)
        save_image(decoded_image, output_name)

    print(f"Success! Image saved to disk as: '{output_name}'")


if __name__ == "__main__":
    # Example execution statement: Change epoch extension matching your checkpoint folder values
    sample_images(
        checkpoint_path="checkpoints/ldm_checkpoint_epoch_20.pt",
        prompt="Three people walking through a beautiful meadow towards the ocean.",
        guidance_scale=3,
        output_name="test_output.png"
    )
