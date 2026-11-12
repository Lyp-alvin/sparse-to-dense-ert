from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


def _load_mat_array(path: Path, key: str, dtype: np.dtype = np.float32) -> np.ndarray:
    data = sio.loadmat(path)
    if key not in data:
        raise KeyError(f"Missing key '{key}' in {path}")
    return np.asarray(data[key], dtype=dtype)


def _to_tensor_1chw(array_2d: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array_2d.astype(np.float32)).unsqueeze(0)


def _extract_ids(folder: Path, pattern: str) -> set[int]:
    regex = re.compile(pattern)
    ids: set[int] = set()
    for path in folder.glob("*.mat"):
        match = regex.match(path.name)
        if match:
            ids.add(int(match.group(1)))
    return ids


def _is_readable_mat(path: Path, key: str) -> bool:
    try:
        data = sio.loadmat(path)
    except Exception:
        return False
    return key in data


def _filter_valid_stage2_ids(
    ids: list[int],
    root_dir: Path,
    split: str,
    pred_dir: Path,
    pred_pattern: str,
    pred_key: str,
) -> list[int]:
    valid_ids: list[int] = []
    skipped: list[int] = []
    for sample_id in ids:
        rho_path = root_dir / f"{split}_model" / f"rho_{sample_id}.mat"
        wa_sparse_path = root_dir / f"{split}_wa_sparse" / f"wainv_{sample_id}.mat"
        pred_path = pred_dir / pred_pattern.format(sample_id=sample_id)
        if (
            _is_readable_mat(rho_path, "rho")
            and _is_readable_mat(wa_sparse_path, "WA_sparse")
            and _is_readable_mat(pred_path, pred_key)
        ):
            valid_ids.append(sample_id)
        else:
            skipped.append(sample_id)
    if skipped:
        preview = ", ".join(str(x) for x in skipped[:10])
        more = "" if len(skipped) <= 10 else " ..."
        warnings.warn(
            f"Skipped {len(skipped)} unreadable Stage-2 samples for split='{split}': {preview}{more}",
            RuntimeWarning,
            stacklevel=2,
        )
    return valid_ids


def discover_split_ids(root_dir: str | Path, split: str) -> list[int]:
    root = Path(root_dir)
    groups = {
        "model": _extract_ids(root / f"{split}_model", r"rho_(\d+)\.mat$"),
        "rhoa_dense": _extract_ids(root / f"{split}_rhoa_dense", r"rhoa_2d_(\d+)\.mat$"),
        "rhoa_sparse": _extract_ids(root / f"{split}_rhoa_sparse", r"rhoa_2d_(\d+)\.mat$"),
        "wa_dense": _extract_ids(root / f"{split}_wa_dense", r"wainv_(\d+)\.mat$"),
        "wa_sparse": _extract_ids(root / f"{split}_wa_sparse", r"wainv_(\d+)\.mat$"),
    }
    common_ids = set.intersection(*groups.values())
    return sorted(common_ids)


class Stage1WADataset(Dataset):
    def __init__(self, root_dir: str | Path, split: str) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.ids = discover_split_ids(self.root_dir, split)
        if not self.ids:
            raise ValueError(f"No paired Stage-1 WA samples found for split='{split}' in {self.root_dir}")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        sample_id = self.ids[index]
        wa_sparse = _load_mat_array(
            self.root_dir / f"{self.split}_wa_sparse" / f"wainv_{sample_id}.mat",
            "WA_sparse",
        )
        wa_dense = _load_mat_array(
            self.root_dir / f"{self.split}_wa_dense" / f"wainv_{sample_id}.mat",
            "WA_dense",
        )
        return {
            "sample_id": sample_id,
            "WA_sparse": _to_tensor_1chw(wa_sparse),
            "WA_dense": _to_tensor_1chw(wa_dense),
        }


class Stage1WARhoaToWADataset(Dataset):
    def __init__(self, root_dir: str | Path, split: str) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.ids = discover_split_ids(self.root_dir, split)
        if not self.ids:
            raise ValueError(f"No paired Stage-1 WA+rhoa samples found for split='{split}' in {self.root_dir}")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        sample_id = self.ids[index]
        wa_sparse = _load_mat_array(
            self.root_dir / f"{self.split}_wa_sparse" / f"wainv_{sample_id}.mat",
            "WA_sparse",
        )
        rhoa_sparse = _load_mat_array(
            self.root_dir / f"{self.split}_rhoa_sparse" / f"rhoa_2d_{sample_id}.mat",
            "rhoa_sparse",
        )
        wa_dense = _load_mat_array(
            self.root_dir / f"{self.split}_wa_dense" / f"wainv_{sample_id}.mat",
            "WA_dense",
        )
        return {
            "sample_id": sample_id,
            "WA_sparse": _to_tensor_1chw(wa_sparse),
            "rhoa_sparse": _to_tensor_1chw(rhoa_sparse),
            "WA_dense": _to_tensor_1chw(wa_dense),
        }


class Stage2Fangan1Dataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        split: str,
        stage1_pred_dir: str | Path,
        pred_prefix: str = "pred_stage1_fangan1",
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.stage1_pred_dir = Path(stage1_pred_dir)
        self.pred_prefix = pred_prefix
        base_ids = set(discover_split_ids(self.root_dir, split))
        pred_ids = _extract_ids(self.stage1_pred_dir, rf"{re.escape(self.pred_prefix)}_(\d+)\.mat$")
        self.ids = _filter_valid_stage2_ids(
            sorted(base_ids.intersection(pred_ids)),
            self.root_dir,
            split,
            self.stage1_pred_dir,
            f"{self.pred_prefix}" + "_{sample_id}.mat",
            "pred_WA_dense",
        )
        if not self.ids:
            raise ValueError(
                f"No Stage-2 Fangan1 samples found for split='{split}'. "
                f"Checked dataset={self.root_dir} and pred_dir={self.stage1_pred_dir}"
            )

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        sample_id = self.ids[index]
        wa_sparse = _load_mat_array(
            self.root_dir / f"{self.split}_wa_sparse" / f"wainv_{sample_id}.mat",
            "WA_sparse",
        )
        rho = _load_mat_array(
            self.root_dir / f"{self.split}_model" / f"rho_{sample_id}.mat",
            "rho",
        )
        pred_file = self.stage1_pred_dir / f"{self.pred_prefix}_{sample_id}.mat"
        pred_wa_dense = _load_mat_array(pred_file, "pred_WA_dense")
        return {
            "sample_id": sample_id,
            "WA_sparse": _to_tensor_1chw(wa_sparse),
            "pred_WA_dense": _to_tensor_1chw(pred_wa_dense),
            "rho": _to_tensor_1chw(rho),
        }


class Stage2Fangan2Dataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        split: str,
        stage1_pred_dir: str | Path,
        pred_prefix: str = "pred_stage1_fangan2",
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.stage1_pred_dir = Path(stage1_pred_dir)
        self.pred_prefix = pred_prefix
        base_ids = set(discover_split_ids(self.root_dir, split))
        pred_ids = _extract_ids(self.stage1_pred_dir, rf"{re.escape(self.pred_prefix)}_(\d+)\.mat$")
        self.ids = _filter_valid_stage2_ids(
            sorted(base_ids.intersection(pred_ids)),
            self.root_dir,
            split,
            self.stage1_pred_dir,
            f"{self.pred_prefix}" + "_{sample_id}.mat",
            "pred_WA_dense",
        )
        if not self.ids:
            raise ValueError(
                f"No Stage-2 Fangan2 samples found for split='{split}'. "
                f"Checked dataset={self.root_dir} and pred_dir={self.stage1_pred_dir}"
            )

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        sample_id = self.ids[index]
        rho = _load_mat_array(
            self.root_dir / f"{self.split}_model" / f"rho_{sample_id}.mat",
            "rho",
        )
        pred_file = self.stage1_pred_dir / f"{self.pred_prefix}_{sample_id}.mat"
        pred_wa_dense = _load_mat_array(pred_file, "pred_WA_dense")
        return {
            "sample_id": sample_id,
            "pred_WA_dense": _to_tensor_1chw(pred_wa_dense),
            "rho": _to_tensor_1chw(rho),
        }


class DirectWASparseToRhoDataset(Dataset):
    def __init__(self, root_dir: str | Path, split: str) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.ids = discover_split_ids(self.root_dir, split)
        if not self.ids:
            raise ValueError(f"No direct WA-sparse samples found for split='{split}' in {self.root_dir}")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        sample_id = self.ids[index]
        wa_sparse = _load_mat_array(
            self.root_dir / f"{self.split}_wa_sparse" / f"wainv_{sample_id}.mat",
            "WA_sparse",
        )
        rho = _load_mat_array(
            self.root_dir / f"{self.split}_model" / f"rho_{sample_id}.mat",
            "rho",
        )
        return {
            "sample_id": sample_id,
            "WA_sparse": _to_tensor_1chw(wa_sparse),
            "rho": _to_tensor_1chw(rho),
        }
