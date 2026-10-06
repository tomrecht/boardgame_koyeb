#!/bin/bash
# Blend sweep (2026-10-05): each candidate vs the deployed champion, 300 colour-
# swapped pairs on the SAME seeds (comparable across candidates), app config.
# The winner then gets a confirmation match on FRESH seeds (winner's curse).
cd "$(dirname "$0")"
for b in b75_25 b50_50 b25_75 b3way b4way; do
  MODEL_A=blends/$b.onnx MODEL_B=model.onnx TAG=sw_$b N_WORKERS=4 python3 match_models.py 300 > blends/sw_$b.log 2>&1
done
for b in b75_25 b50_50 b25_75 b3way b4way; do TAG=sw_$b MODEL_A=blends/$b.onnx python3 match_models.py analyze; done > blends/sweep_summary.txt 2>&1
