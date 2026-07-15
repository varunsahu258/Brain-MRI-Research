from __future__ import annotations
import argparse
from pathlib import Path
from evaluation.ablation_report import build_ablation_table
if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--predictions", required=True); p.add_argument("--output", default="outputs/results/ablation_table.csv"); args=p.parse_args()
    table=build_ablation_table(args.predictions); Path(args.output).parent.mkdir(parents=True,exist_ok=True); table.to_csv(args.output,index=False); table.to_markdown(Path(args.output).with_suffix(".md"),index=False)
