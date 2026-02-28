# apps/fashion-mnist/src/main.py
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="HANSAL ML Pipeline")
    parser.add_argument("stage", choices=["etl", "train"])
    args, remaining = parser.parse_known_args()
    sys.argv = [args.stage] + remaining

    if args.stage == "etl":
        from etl import main as run
    elif args.stage == "train":
        from train import main as run
    run()

if __name__ == "__main__":
    main()
