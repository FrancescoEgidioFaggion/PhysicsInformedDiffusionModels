import pytest
import torch

from src.grad_utils import generalized_b_xy_c_to_image
from src.residuals_darcy import ResidualsDarcy


@pytest.fixture
def physics():
    return ResidualsDarcy(
        model=None,
        fd_acc=2,
        pixels_per_dim=8,
        pixels_at_boundary=True,
        reverse_d1=True,
        device="cpu",
    )


@pytest.fixture
def fields():
    generator = torch.Generator().manual_seed(42)
    pressure = 0.01 * torch.randn(2, 64, generator=generator)
    permeability = torch.ones_like(pressure)
    return torch.stack((pressure, permeability), dim=-1)


def evaluate(physics, fields):
    return physics.compute_residual(
        generalized_b_xy_c_to_image(fields),
        pass_through=True,
    )["residual"]


def objective_per_sample(residual):
    return 0.5 * residual.square().flatten(1).mean(dim=1)


def test_decrease_and_invariants(physics, fields):
    original = fields.clone()
    before = objective_per_sample(evaluate(physics, fields))

    # Correction must also work inside a sampling no_grad context.
    with torch.no_grad():
        corrected, residual = physics.residual_correction_backtracking(
            fields
        )

    after = objective_per_sample(residual)
    assert torch.all(after < before)
    assert torch.equal(fields, original)
    assert torch.equal(corrected[..., 1], original[..., 1])
    assert not corrected.requires_grad
    assert not residual.requires_grad
    torch.testing.assert_close(residual, evaluate(physics, corrected))


def test_batch_independence(physics, fields):
    batched, _ = physics.residual_correction_backtracking(fields)

    individual = [
        physics.residual_correction_backtracking(fields[i:i + 1])[0]
        for i in range(len(fields))
    ]

    torch.testing.assert_close(
        batched, torch.cat(individual), rtol=0, atol=0
    )


def test_rejected_step_preserves_sample(physics, fields):
    corrected, residual = physics.residual_correction_backtracking(
        fields,
        initial_step=1e10,
        max_backtracks=0,
    )

    assert torch.equal(corrected, fields)
    torch.testing.assert_close(residual, evaluate(physics, fields))


def test_zero_residual_is_unchanged(physics, fields):
    # Constant pressure with zero source is an exact equilibrium.
    physics.f_s.zero_()
    fields[..., 0] = 0.0

    corrected, residual = physics.residual_correction_backtracking(fields)

    assert torch.equal(corrected, fields)
    assert torch.count_nonzero(residual).item() == 0


@pytest.mark.parametrize(
    "options",
    [
        {"initial_step": 0.0},
        {"initial_step": float("nan")},
        {"contraction": 1.0},
        {"armijo": 0.0},
        {"max_backtracks": -1},
    ],
)
def test_invalid_options(physics, fields, options):
    with pytest.raises(ValueError):
        physics.residual_correction_backtracking(fields, **options)



def test_pressure_gradient_matches_finite_difference():
    # Build the finite-difference kernels in double precision.
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        physics = ResidualsDarcy(
            model=None,
            fd_acc=2,
            pixels_per_dim=8,
            pixels_at_boundary=True,
            reverse_d1=True,
            device="cpu",
        )
    finally:
        torch.set_default_dtype(previous_dtype)

    generator = torch.Generator().manual_seed(123)
    pressure = (
        0.01 * torch.randn(1, 64, generator=generator, dtype=torch.float64)
    ).requires_grad_(True)
    permeability = (
        0.5 + torch.rand(1, 64, generator=generator, dtype=torch.float64)
    )

    def objective(p):
        fields = torch.stack((p, permeability), dim=-1)
        return 0.5 * evaluate(physics, fields).square().mean()

    gradient, = torch.autograd.grad(objective(pressure), pressure)

    direction = torch.randn(
        pressure.shape, generator=generator, dtype=torch.float64
    )
    direction /= direction.norm()

    epsilon = 1e-6
    with torch.no_grad():
        numerical = (
            objective(pressure + epsilon * direction)
            - objective(pressure - epsilon * direction)
        ) / (2 * epsilon)

    autodiff = (gradient * direction).sum()
    torch.testing.assert_close(
        autodiff, numerical, rtol=1e-5, atol=1e-7
    )


def test_accepted_step_satisfies_armijo(physics, fields):
    armijo = 1e-4
    corrected, _ = physics.residual_correction_backtracking(
        fields, armijo=armijo
    )

    for index in range(len(fields)):
        original = fields[index:index + 1]
        pressure = original[..., 0].clone().requires_grad_(True)
        differentiable = torch.stack(
            (pressure, original[..., 1]), dim=-1
        )
        before = objective_per_sample(
            evaluate(physics, differentiable)
        ).sum()
        gradient, = torch.autograd.grad(before, pressure)

        delta = corrected[index:index + 1, ..., 0] - pressure.detach()
        norm_squared = gradient.square().sum()

        # Recover alpha from delta = -alpha * gradient.
        accepted_step = -(delta * gradient).sum() / norm_squared
        assert accepted_step.item() > 0

        torch.testing.assert_close(
            delta,
            -accepted_step * gradient,
            rtol=1e-4,
            atol=1e-7,
        )

        after = objective_per_sample(
            evaluate(physics, corrected[index:index + 1])
        ).sum()

        bound = before.detach() - armijo * accepted_step * norm_squared
        tolerance = 10 * torch.finfo(after.dtype).eps * before.detach().abs()
        assert after <= bound + tolerance


def test_dispatch_uses_backtracking_without_full_jacobian(
    physics, fields, monkeypatch
):
    import src.residuals_darcy as darcy_module

    def forbidden_jacobian(*args, **kwargs):
        raise AssertionError("Full Jacobian must not be constructed.")

    monkeypatch.setattr(darcy_module, "jacfwd", forbidden_jacobian)
    physics.correction_method = "backtracking"

    corrected, residual = physics.residual_correction(fields)
    expected, expected_residual = (
        physics.residual_correction_backtracking(fields)
    )

    torch.testing.assert_close(corrected, expected)
    torch.testing.assert_close(residual, expected_residual)


def test_legacy_is_default(physics):
    assert physics.correction_method == "legacy"


def test_unknown_correction_method_is_rejected():
    with pytest.raises(ValueError, match="correction_method"):
        ResidualsDarcy(
            model=None,
            fd_acc=2,
            pixels_per_dim=8,
            pixels_at_boundary=True,
            reverse_d1=True,
            correction_method="unknown",
        )


@pytest.mark.parametrize(
    "mode, post_steps, sampling_steps",
    [
        ("xt", 1, 0),
        ("xt", 0, 1),
        ("x0", 0, 1),
    ],
)
def test_sampling_integration(fields, mode, post_steps, sampling_steps):
    from src.denoising_utils import DenoisingDiffusion
    from src.grad_utils import generalized_image_to_b_xy_c

    # A deterministic denoiser isolates the correction integration.
    fixed_fields = fields[:1]

    class FixedModel(torch.nn.Module):
        def forward(self, noisy, time):
            image = generalized_b_xy_c_to_image(fixed_fields)
            return image.expand(noisy.shape[0], -1, -1, -1).clone()

    physics = ResidualsDarcy(
        model=FixedModel(),
        fd_acc=2,
        pixels_per_dim=8,
        pixels_at_boundary=True,
        reverse_d1=True,
        device="cpu",
        correction_method="backtracking",
    )
    diffusion = DenoisingDiffusion(5, torch.device("cpu"))

    def sample(m, n):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(123)
            with torch.no_grad():
                return diffusion.p_sample_loop(
                    conditioning_input=None,
                    shape=(1, 2, 8, 8),
                    save_output=True,
                    residual_func=physics,
                    eval_residuals=True,
                    M_correction=m,
                    N_correction=n,
                    correction_mode=mode,
                )

    baseline, _ = sample(0, 0)
    result, auxiliary = sample(post_steps, sampling_steps)

    original = generalized_image_to_b_xy_c(baseline[0][-1])
    actual = generalized_image_to_b_xy_c(result[0][-1])
    expected, _ = physics.residual_correction(original)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        auxiliary["residual"], evaluate(physics, actual)
    )
    assert torch.equal(actual[..., 1], original[..., 1])
