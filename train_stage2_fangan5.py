from __future__ import annotations

from train_stage2_wa_common import build_parser, run_training


def main() -> None:
    parser = build_parser(
        "Train Stage 2 UNet++ for fangan5 ([pred_WA_dense, WA_sparse] -> rho).",
        "runs/stage2_fangan5",
        "runs/stage1_fangan5",
    )
    parser.set_defaults(stage1_checkpoint_name="best_stage1_fangan5.pth")
    args = parser.parse_args()
    run_training(args, suffix="fangan5", mode="fangan5")


if __name__ == "__main__":
    main()
