<h1 align="center">Physics-Informed Diffusion Models (ICLR 2025)</h1>
<h4 align="center">
<a href="https://arxiv.org/abs/2403.14404"><img src="https://img.shields.io/badge/arXiv-2403.14404-blue" alt="arXiv"></a>
</h4>
<div align="center">
  <span class="author-block">
    <a>Jan-Hendrik Bastek</a><sup>1</sup>,</span>
  <span class="author-block">
    <a>WaiChing Sun</a><sup>2</sup> and</span>
  <span class="author-block">
    <a>Dennis M. Kochmann</a><sup>1</sup></span>
</div>
<div align="center">
  <span class="author-block"><sup>1</sup>ETH Zurich,</span>
  <span class="author-block"><sup>2</sup>Columbia University</span>
</div>

$~$
<p align="center"><img src="circular_samples.gif" width="550"\></p>

## Introduction & Setup
We introduce a framework to inform diffusion models of constraints generated samples must adhere to during model training, as presented in [Physics-Informed Diffusion Models](https://arxiv.org/abs/2403.14404).
To conduct similar studies as those presented in the preprint, start by cloning this repository via
```
git clone https://github.com/jhbastek/PhysicsInformedDiffusionModels.git
```
We provide three scripts:

`main_toy.py` reproduces the toy study presented in Appendix F.1. It is helpful to understand the implications of the PIDM loss and several variants. Simply change the config file and run the script to reproduce the results or experiment with different parameters.

To reproduce the results for the Darcy flow and topology optimization study, you will first have to download the data and pretrained models from the [ETHZ Research Collection](https://doi.org/10.3929/ethz-b-000674074) and place them (unzipped) as follows:
```
.
├── data
│   ├── darcy
│   │   └── ...
│   └── mechanics
│       └── ...
└── trained_models
    ├── darcy
    │   └── ...
    └── mechanics
        └── ...
```

On macOS, if standard extraction fails, try using `ditto -x -k <source> <target>`.

After this, you can run the following scripts:

`main.py` reproduces the Darcy flow and topology optimization study presented in Section 4. Simply adjust the parameters and governing equations in `model.yaml` and run the script to train the models. Note that the name of the run and logging parameters can be directly adjusted in `main.py`, if necessary.

`sample.py` evaluates trained models. Provide the `directory_path`, `name`, and `load_model_step` of the model to evaluate and run the script. Note that the full evaluation of the in- and out-of-distribution test sets for the topology optimization study may take some time.

## Dependencies

The framework was developed and tested on Python 3.11 using CUDA 12.0.
To run the toy model, the following packages are required:
Package | Version (>=)
:-|:-
`pytorch`                   | `2.0.1`
`tqdm`                      | `4.65.0`
`matplotlib`                | `3.7.2`
`imageio`                   | `2.28.1`
`einops`                    | `0.6.1`
`wandb` (optional)          | `0.15.2`

To run the Darcy flow and topology optimization study, the following additional packages are required:
Package | Version (>=)
:-|:-
`findiff`                   | `0.10.0`
`solidspy`                  | `1.1.0.post1`
`pandas`                    | `2.1.3`
`einops-exts`               | `0.0.4`
`rotary_embedding_torch`    | `0.2.3`
`torchvision`               | `0.15.2`
`opencv`                    | `4.9.0.80`

## Jacobian-free Darcy residual correction

The Darcy example supports an optional physics-correction method based on
reverse-mode differentiation and per-sample Armijo backtracking.

For fixed permeability $K$, the method minimizes

$$
\Phi(p;K)=\frac{1}{2M}\sum_{i=1}^{M}R_i(p,K)^2
$$

where $R_i(p,K)$ denotes one scalar entry of the flattened PDE-and-boundary
residual tensor and $M$ is the total number of such entries.

Only the pressure field is updated; permeability remains unchanged. The line search accepts a step only
when it provides sufficient decrease of the objective.

Unlike the original correction method, this implementation does not construct
the full residual Jacobian. The original `legacy` method remains the default
for backward compatibility.

### Enabling the correction in `sample.py`

`sample.py` loads its configuration from the YAML file stored alongside the
selected checkpoint, for example:

```text
trained_models/darcy/PIDM-ME/model/model.yaml
```

To apply ten backtracking corrections after the reverse diffusion process has
produced the final sample, add or update:

```yaml
correction_method: backtracking
M_correction: 10
N_correction: 0
```

`correction_method` selects either the original `legacy` implementation or the
new `backtracking` implementation. `M_correction` is the number of correction
steps applied to the final generated sample. `N_correction` is the number of
final reverse-diffusion timesteps at which correction is applied; setting it to
zero disables correction inside the reverse process.

The standalone benchmark does not read these three options. Its method and
number of post-sampling corrections are selected through `--method` and
`--steps`.

### Darcy CPU environment

The tested CPU environment can be installed with:

```bash
python -m pip install -r requirements-darcy-cpu.txt
```

The Darcy checkpoint must be downloaded from the model collection linked
above and placed in:

```text
trained_models/darcy/PIDM-ME/model/checkpoint_300000.pt
```

Generate five samples using fixed random seeds (42–46):

```bash
python generate_darcy_samples.py
```

Run the comparison for each sample:

```bash
for seed in 42 43 44 45 46; do
  python benchmark_darcy_correction.py \
    --input trained_models/darcy/PIDM-ME/samples/seed_${seed}.pt \
    --method legacy --steps 10

  python benchmark_darcy_correction.py \
    --input trained_models/darcy/PIDM-ME/samples/seed_${seed}.pt \
    --method backtracking --steps 10
done
```

Run the focused test suite:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python -m pytest -q tests/test_darcy_backtracking.py
```

### Benchmark

The following values are averages over five generated $64 \times 64$ Darcy
samples, using ten post-sampling correction steps on an Intel i7 MacBook Pro
CPU with four threads.

| Method | Correction time [s] | Objective reduction | PDE RMS reduction | Final objective |
|---|---:|---:|---:|---:|
| Legacy | 48.42 | 19.19% | 13.68% | 4.370e-4 |
| Backtracking | 1.80 | 27.54% | 20.50% | 3.944e-4 |

The backtracking method reduced the objective for every tested sample while
keeping permeability unchanged. Boundary RMS changed only marginally on
average and was not monotonic for every sample. Runtime values are
hardware-dependent.

## Citation

If this code is useful for your research, please consider citing
```bibtex
@inproceedings{
bastek2025physicsinformed,
title={Physics-Informed Diffusion Models},
author={Jan-Hendrik Bastek and WaiChing Sun and Dennis Kochmann},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=tpYeermigp}
}
```
