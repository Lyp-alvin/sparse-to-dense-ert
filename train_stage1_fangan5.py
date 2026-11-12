from __future__ import annotations

from train_stage1_wa_common import build_parser, run_training


def main() -> None:
    parser = build_parser(
        "Train Stage 1 UNet++ for fangan5 ([WA_sparse, rhoa_sparse] -> WA_dense).",
        "runs/stage1_fangan5",
    )
    args = parser.parse_args()
    run_training(args, suffix="fangan5", mode="fangan5")


if __name__ == "__main__":
    main()
