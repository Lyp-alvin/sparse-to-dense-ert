from __future__ import annotations

from train_direct_rho_common import build_parser, run_training


def main() -> None:
    parser = build_parser(
        "Train direct baseline WA_sparse -> rho.",
        "runs/baseline_wa_sparse",
    )
    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
