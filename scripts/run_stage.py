from __future__ import annotations
import argparse
from training.train_stage import train_stage
if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--stage",type=int,required=True); args=p.parse_args(); train_stage(args.stage)
