# Electrode-Deployment-Aware Sparse-to-Dense Prior Learning for ERT

The four reconstruction methods share the U-Net++ backbone in [`UNetPP.py`](UNetPP.py). Each network has one output channel. Brackets indicate the input channel order.

| Method | Stage 1 input → dense prior | Reconstruction input → resistivity | Training entry points |
| --- | --- | --- | --- |
| SCI-Baseline | — | `WA_sparse → rho` | `train_baseline_wa_sparse.py` |
| DPO-Recon | `WA_sparse → pred_WA_dense` | `pred_WA_dense → rho` | `train_stage1_fangan1.py`, `train_stage2_fangan2.py` |
| MDP-Recon | Shares Stage 1 with DPO-Recon | `[pred_WA_dense, WA_sparse] → rho` | `train_stage1_fangan1.py`, `train_stage2_fangan1.py` |
| Proposed | `[WA_sparse, rhoa_sparse] → pred_WA_dense` | `[pred_WA_dense, WA_sparse] → rho` | `train_stage1_fangan5.py`, `train_stage2_fangan5.py` |

## Setup and weights

Use Python 3.9 or later. Run commands from the repository root.

```sh
python -m pip install -r requirements.txt
gh release download v1.0.0 --repo Lyp-alvin/sparse-to-dense-ert --dir weights
```

The six files are also available from [Release v1.0.0](https://github.com/Lyp-alvin/sparse-to-dense-ert/releases/tag/v1.0.0). Each `best_*.pth` contains the `model_state_dict` from the corresponding best validation checkpoint. The released Stage 1 weights can generate priors for Stage 2 training. Resuming a training run requires its full checkpoint from `runs/`.

## Example samples

`data_chaofen/` contains the four validation cases **1981, 1977, 1893, and 1886** shown in the manuscript. The original MAT arrays use a 256 × 1024 grid in the log₁₀ resistivity domain. `WA` denotes the conventional Wenner-alpha inversion image. `rhoa` denotes the apparent-resistivity pseudosection. The mask arrays mark valid pseudosection entries.

| Directory | Filename pattern | MAT keys |
| --- | --- | --- |
| `val_model` | `rho_<id>.mat` | `rho` |
| `val_wa_sparse` | `wainv_<id>.mat` | `WA_sparse` |
| `val_wa_dense` | `wainv_<id>.mat` | `WA_dense` |
| `val_rhoa_sparse` | `rhoa_2d_<id>.mat` | `rhoa_sparse`, `mask_sparse` |
| `val_rhoa_dense` | `rhoa_2d_<id>.mat` | `rhoa_dense`, `mask_dense` |

Run all four methods on the bundled cases with the following Python code. It uses the reference inference batch size of four and enables cuDNN TF32 on CUDA. Each output tensor has shape `(4, 1, 256, 1024)` in the order given by `case_ids`.

```python
from pathlib import Path

import torch

from Dataset import Stage1WARhoaToWADataset
from UNetPP import UNetPP
from pipeline_utils import load_checkpoint

case_ids = (1981, 1977, 1893, 1886)
dataset = Stage1WARhoaToWADataset("data_chaofen", "val")
samples = [dataset[dataset.ids.index(case_id)] for case_id in case_ids]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.allow_tf32 = True
wa = torch.stack([sample["WA_sparse"] for sample in samples]).to(device)
rhoa = torch.stack([sample["rhoa_sparse"] for sample in samples]).to(device)


@torch.inference_mode()
def predict(checkpoint_name, inputs):
    model = UNetPP(in_channels=inputs.shape[1], out_channels=1).to(device).eval()
    load_checkpoint(Path("weights") / checkpoint_name, model, map_location=device)
    return model(inputs)


shared_prior = predict("best_stage1_fangan1.pth", wa)
proposed_prior = predict("best_stage1_fangan5.pth", torch.cat([wa, rhoa], dim=1))
predictions = {
    "SCI-Baseline": predict("best_baseline_wa_sparse.pth", wa),
    "DPO-Recon": predict("best_stage2_fangan2.pth", shared_prior),
    "MDP-Recon": predict("best_stage2_fangan1.pth", torch.cat([shared_prior, wa], dim=1)),
    "Proposed": predict("best_stage2_fangan5.pth", torch.cat([proposed_prior, wa], dim=1)),
}
```

## Training

Training requires paired `train_*` and `val_*` directories in the same layout as the table. The bundled cases are for inference. Replace `PATH_TO_FULL_DATA` with the full dataset directory. The Stage 2 scripts use the best Stage 1 checkpoints from `runs/` and generate the matching train and validation priors before training.

```sh
python train_baseline_wa_sparse.py --data-root PATH_TO_FULL_DATA
python train_stage1_fangan1.py --data-root PATH_TO_FULL_DATA
python train_stage2_fangan2.py --data-root PATH_TO_FULL_DATA
python train_stage2_fangan1.py --data-root PATH_TO_FULL_DATA
python train_stage1_fangan5.py --data-root PATH_TO_FULL_DATA
python train_stage2_fangan5.py --data-root PATH_TO_FULL_DATA
```

To train a reconstruction model with the released Stage 1 weights, pass `--stage1-checkpoint weights/best_stage1_fangan1.pth` for DPO-Recon or MDP-Recon, or `--stage1-checkpoint weights/best_stage1_fangan5.pth` for the proposed method. Use a fresh `--work-dir` when switching Stage 1 weights or datasets to generate a new prediction cache.
