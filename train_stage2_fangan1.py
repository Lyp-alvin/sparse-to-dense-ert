from __future__ import annotations

from train_stage2_wa_common import build_parser, run_training


def main() -> None:
    parser = build_parser(
        "Train Stage 2 UNet++ for fangan1 ({WA_dense_pred, WA_sparse} -> rho).",
        "runs/stage2_fangan1",
        "runs/stage1_fangan1",
    )
    args = parser.parse_args()
    run_training(args, suffix="fangan1", mode="fangan1")


if __name__ == "__main__":
    main()
