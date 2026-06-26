import os
import sys

sys.path.append(".")
from src.utility.parser_util import parse_args
from src.utility.run_training import run_two_stage_training


def main() -> None:
    args = parse_args()
    os.makedirs("src/logs", exist_ok=True)
    log_path = f"src/logs/{args.run_name}.log"
    log_file = open(log_path, "w")  # noqa: SIM115
    sys.stdout = log_file
    try:
        run_two_stage_training(args)
    finally:
        log_file.close()
        sys.stdout = sys.__stdout__
        print(f"Training complete. Log saved to {log_path}")


if __name__ == "__main__":
    main()
