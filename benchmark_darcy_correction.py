"""Compare Darcy correction methods on one saved diffusion sample."""

import argparse
import csv
import time
from pathlib import Path

import torch

from src.grad_utils import generalized_image_to_b_xy_c
from src.residuals_darcy import ResidualsDarcy


def metrics(physics, fields, reference):
    with torch.no_grad():
        residual = physics.compute_residual_direct(fields)
        n = physics.pixels_per_dim

        pressure = fields[..., 0]
        original_pressure = reference[..., 0]

        # Remove the arbitrary constant pressure offset for this metric.
        pressure = pressure - pressure.mean(dim=1, keepdim=True)
        original_pressure = original_pressure - original_pressure.mean(
            dim=1, keepdim=True
        )

        return {
            "objective": (0.5 * residual.square().mean()).item(),
            "pde_rms": residual[..., 0].square().mean().sqrt().item(),
            # Boundary channels contain zeros away from the boundary.
            "boundary_rms": (
                residual[..., 1:].square().sum()
                / (fields.shape[0] * 4 * n)
            ).sqrt().item(),
            "pressure_change_rms": (
                (pressure - original_pressure).square().mean().sqrt().item()
            ),
            "permeability_unchanged": torch.equal(
                fields[..., 1], reference[..., 1]
            ),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method", choices=["legacy", "backtracking"], required=True
    )
    parser.add_argument("--steps", type=int, default=1)

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "trained_models/darcy/PIDM-ME/samples/seed_42.pt"
        ),
    )

    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    torch.set_num_threads(4)

    saved = torch.load(
        args.input,
        map_location="cpu",
        weights_only=True,
    )

    image = saved["sample"]
    n = image.shape[-1]
    reference = generalized_image_to_b_xy_c(image).detach().clone()

    physics = ResidualsDarcy(
        model=None,
        fd_acc=2,
        pixels_per_dim=n,
        pixels_at_boundary=True,
        reverse_d1=True,
        device="cpu",
        correction_method=args.method,
    )

    fields = reference.clone()
    rows = [{
        "method": args.method,
        "step": 0,
        "correction_seconds": 0.0,
        **metrics(physics, fields, reference),
    }]
    print(rows[0], flush=True)

    total_seconds = 0.0
    for step in range(1, args.steps + 1):
        start = time.perf_counter()
        fields, _ = physics.residual_correction(fields)
        total_seconds += time.perf_counter() - start
        fields = fields.detach()

        assert torch.isfinite(fields).all(), "Non-finite corrected sample"

        row = {
            "method": args.method,
            "step": step,
            "correction_seconds": total_seconds,
            **metrics(physics, fields, reference),
        }
        assert row["permeability_unchanged"]
        rows.append(row)
        print(row, flush=True)


    destination = args.input.with_name(
        f"{args.input.stem}_{args.method}.csv"
    )
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Results saved to {destination}", flush=True)


if __name__ == "__main__":
    main()
