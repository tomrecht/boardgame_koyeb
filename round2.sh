#!/bin/bash
# Round 2 (2026-10-06 early morning): more relatives in the soup, each vs the
# NEW champion (the 4-way blend, now model.onnx), 300 pairs on the same seeds.
cd "$(dirname "$0")"
for b in b6way b5way b4champ; do
  MODEL_A=blends/$b.onnx MODEL_B=model.onnx TAG=r2_$b SEED_BASE=7600000 N_WORKERS=4 python3 match_models.py 300 > blends/r2_$b.log 2>&1
done
for b in b6way b5way b4champ; do TAG=r2_$b MODEL_A=blends/$b.onnx python3 match_models.py analyze; done > blends/round2_summary.txt 2>&1
