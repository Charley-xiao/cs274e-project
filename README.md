# CS274E Project

## TL;DR

```bash
# 0) env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) train (RF baseline, conditional)
python -m src.train --cfg configs/rf.yaml

# 2) train (RF + divergence regularizer)
python -m src.train --cfg configs/rf_div.yaml

# 3) sample (class-conditional with optional guidance)
python -m src.sample --ckpt runs/rf_cond/last.ckpt --nfe 1 2 4 \
  --classes 0 1 2 3 4 --guidance_scale 1.5

# 4) eval (FID/IS)
python -m src.eval --real data/eurosat_val --fake samples_out/nfe4 --outfile fid.json
```

## Project layout

```
cs274e-project/
├─ README.md
├─ requirements.txt
├─ configs/
│  ├─ rf.yaml          # baseline RF
│  └─ rf_div.yaml      # RF + divergence
├─ src/
│  ├─ data.py          # eurosat_dataloaders(): (train_loader, val_loader) -> (images, labels)
│  ├─ model.py         # create_model(cfg, num_classes): forward(x_t, t, y)
│  ├─ loss.py          # rf_loss(), rf_div_loss()
│  ├─ ode.py           # integrate(v_theta, nfe, solver, y, guidance_scale)
│  ├─ train.py         # training loop (reads YAML config)
│  ├─ sample.py        # save samples for NFE {1,2,4,8}, per-class if provided
│  └─ eval.py          # FID/IS via torch-fidelity
└─ runs/               # auto: checkpoints, sample grids, metrics.json
```

## Dependencies

See `requirements.txt`.

> EuroSAT (RGB) comes from `torchvision.datasets.EuroSAT`. It will **auto-download** to `data/` on first use.

## Configs

### `configs/rf.yaml` (baseline)

```yaml
name: "rf_cond"
seed: 42
data:
  root: "./data"
  image_size: 64
  batch_size: 128
  num_workers: 4
model:
  in_channels: 3
  # add model hyperparams as needed (channels, depth, etc.)
opt:
  lr: 2e-4
  wd: 2e-2
  ema: 0.999
train:
  epochs: 120
  t_sampling: "uniform"   # or "beta_half"
  log_every: 100
  grad_clip: 1.0
sample:
  solver: "heun"          # "euler" | "heun"
  guidance_scale: 0.0     # runtime CFG scale
cond:
  num_classes: 10         # EuroSAT-RGB has 10 classes
  p_uncond: 0.1           # classifier-free dropout prob during training
```

### `configs/rf_div.yaml` (add divergence term)

```yaml
name: "rf_cond_div"
seed: 42
data:
  root: "./data"
  image_size: 64
  batch_size: 128
  num_workers: 4
model:
  in_channels: 3
opt:
  lr: 2e-4
  wd: 2e-2
  ema: 0.999
train:
  epochs: 120
  t_sampling: "uniform"
  log_every: 100
  grad_clip: 1.0
sample:
  solver: "heun"
  guidance_scale: 0.0
cond:
  num_classes: 10
  p_uncond: 0.1
loss:
  lambda_div: 1.0e-3      # sweep: {0, 1e-4, 1e-3, 5e-3}
  hutch_probes: 1         # 1–4; increases cost linearly
```

## Training

### Baseline RF

```bash
python -m src.train --cfg configs/rf.yaml
# Checkpoints and quick sample grids land in runs/rf_cond/
```

### RF-Div

```bash
python -m src.train --cfg configs/rf_div.yaml
# Try: loss.lambda_div=1e-4 (override from CLI)
python -m src.train --cfg configs/rf_div.yaml loss.lambda_div=1e-4
```

### Notes

* **Conditioning:** the model sees class labels `y` (LongTensor). We use **classifier-free dropout**: with prob `p_uncond`, labels become `-1` (unconditional token) during training.
* **Time sampling:** start with `uniform`; later try `"beta_half"` (U-shaped).
* **EMA:** used for sampling and logging.

## Sampling

**Unconditional**

```bash
python -m src.sample --ckpt runs/rf_cond/last.ckpt --nfe 1 2 4 8
```

**Per-class (e.g., classes 0–4) with guidance**

```bash
python -m src.sample --ckpt runs/rf_cond/last.ckpt --nfe 1 2 4 \
  --classes 0 1 2 3 4 --guidance_scale 1.5
```

Outputs:

* `samples_out/grid_nfe{K}.png`: grids
* `samples_out/nfe{K}/00000.png`: individual images

> **Guidance:** we use classifier-free guidance on the **velocity field**:
> $v_{\text{guided}} = (1+s),v(x,t,y) - s,v(x,t,\text{uncond})$.


## Evaluation (FID / IS)

Generate a folder of fakes (e.g., `samples_out/nfe4/`), and point to a folder of **real** validation images (we export val images during data prep or rely on torchvision caching).

```bash
python -m src.eval --real data/eurosat_val --fake samples_out/nfe4 --outfile fid.json
```

Keep the **same image size** and normalization (([-1,1]) → ([0,1]) before saving). Use ~10k images for stable FID.


## Repro & logging

* **Seeds:** set in config (`seed`).
* **Runs:** every experiment lives under `runs/{name}/` with `last.ckpt`, `metrics.json`, and sample grids.
* **Report quick facts:** #params, training time/epoch, NFEs vs FID/IS.


## Implementation notes

* **Model API:** `create_model(cfg_model, num_classes)` builds a net that accepts `(x_t, t, y)`; treat `y=-1` as unconditional (learned null embedding).
* **Data API:** `eurosat_dataloaders(root, image_size, batch_size, num_workers)` returns `(train_loader, val_loader)` yielding `(images, labels)`. Images normalized to ([-1,1]).
* **ODE:** fixed-step Euler/Heun from (t=0\to1). We report NFEs as \# of steps.
