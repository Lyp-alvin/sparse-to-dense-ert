from __future__ import annotations

from train_stage1_wa_common import build_parser, run_training


def main() -> None:
    parser = build_parser(
        "Train Stage 1 UNet++ for fangan1 (WA_sparse -> WA_dense).",
        "runs/stage1_fangan1",
    )
    args = parser.parse_args()
    run_training(args, suffix="fangan1")


if __name__ == "__main__":
    main()
