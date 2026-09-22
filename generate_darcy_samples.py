"""Generate five reproducible Darcy samples from the pretrained PIDM-ME."""

import hashlib
import time
from pathlib import Path

import torch
import yaml

from src.denoising_utils import DenoisingDiffusion
from src.residuals_darcy import ResidualsDarcy
from src.unet_model import Unet3D


def main():
    torch.set_num_threads(4)

    folder = Path("trained_models/darcy/PIDM-ME")
    checkpoint_path = folder / "model/checkpoint_300000.pt"
    config = yaml.safe_load(
        (folder / "model/model.yaml").read_text()
    )
    if (
        config["gov_eqs"] != "darcy"
        or config["x0_estimation"] != "mean"
        or config["residual_grad_guidance"]
    ):
        raise ValueError("This script expects the unguided Darcy PIDM-ME.")

    model = Unet3D(dim=32, channels=2, sigmoid_last_channel=False)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    checkpoint_hash = hashlib.sha256(
        checkpoint_path.read_bytes()
    ).hexdigest()

    physics = ResidualsDarcy(
        model=model,
        fd_acc=config["fd_acc"],
        pixels_per_dim=64,
        pixels_at_boundary=True,
        reverse_d1=True,
        device="cpu",
    )
    diffusion = DenoisingDiffusion(
        config["diff_steps"], torch.device("cpu"), False
    )

    destination = folder / "samples"
    destination.mkdir(exist_ok=True)

    for seed in range(42, 47):
        torch.manual_seed(seed)
        start = time.perf_counter()

        with torch.no_grad():
            sequences, _ = diffusion.p_sample_loop(
                conditioning_input=None,
                shape=(1, 2, 64, 64),
                save_output=True,
                surpress_noise=True,
                residual_func=physics,
                eval_residuals=False,
                M_correction=0,
                N_correction=0,
            )

        sample = sequences[-1]
        assert torch.isfinite(sample).all()
        elapsed = time.perf_counter() - start

        torch.save(
            {
                "sample": sample,
                "seed": seed,
                "generation_seconds": elapsed,
                "checkpoint_sha256": checkpoint_hash,
                "torch_version": str(torch.__version__),
                "config": config,
                "corrections_applied": 0,
            },
            destination / f"seed_{seed}.pt",
        )
        print(
            f"Seed {seed}: {elapsed:.2f} s, "
            f"K_min={sample[:, 1].min().item():.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
