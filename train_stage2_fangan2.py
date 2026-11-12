from __future__ import annotations

from train_stage2_wa_common import build_parser, run_training


def main() -> None:
    parser = build_parser(
        "Train Stage 2 UNet++ for fangan2 (WA_dense_pred -> rho).",
        "runs/stage2_fangan2",
        "runs/stage1_fangan1",
    )
    parser.set_defaults(stage1_checkpoint_name="best_stage1_fangan1.pth")
    args = parser.parse_args()
    run_training(args, suffix="fangan2", mode="fangan2")


if __name__ == "__main__":
    main()
