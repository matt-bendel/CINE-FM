#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="/storage/CINE_data" # TODO!!
PCTL=99.5
SHARD=64
COMP="lzf"

# TODO!!
# ---- EDIT THESE ARRAYS ----
TRAIN_PKLS=(
  "/storage/training_max1_ocmr232425_CS.pkl"
)

VAL_PKLS=(
  "/storage/testing_max1_ocmr232425_CS.pkl"
)

TEST_PKLS=(
  "/storage/testing_station6_max1_ocmr232425_CS.pkl"
  "/storage/testing_station7_max1_ocmr232425_CS.pkl"
)

echo "=== TRAIN ==="
for PKL in "${TRAIN_PKLS[@]}"; do
  echo "[train] $PKL"
  python utils/preprocess_data_to_h5.py \
    --pkl "$PKL" \
    --out_root "$OUT_ROOT" \
    --split train \
    --shard_size $SHARD \
    --compression $COMP \
    --pctl $PCTL
done

echo "=== VAL ==="
for PKL in "${VAL_PKLS[@]}"; do
  echo "[val] $PKL"
  python utils/preprocess_data_to_h5.py \
    --pkl "$PKL" \
    --out_root "$OUT_ROOT" \
    --split val \
    --shard_size $SHARD \
    --compression $COMP \
    --pctl $PCTL
done

echo "=== TEST ==="
for PKL in "${TEST_PKLS[@]}"; do
  echo "[test] $PKL"
  python utils/preprocess_data_to_h5.py \
    --pkl "$PKL" \
    --out_root "$OUT_ROOT" \
    --split test \
    --shard_size $SHARD \
    --compression $COMP \
    --pctl $PCTL
done

echo "All splits updated under $OUT_ROOT/{train,val,test}"

# TODO!!
echo "=== TRAIN LATENTS ==="
python utils/precompute_latents.py \
  --config configs/vae.yaml \
  --in_root "$OUT_ROOT" \
  --out_root "$OUT_ROOT/latents" \
  --split train \
  --vae_import CardiacVAE.model.vae \
  --vae_class CardiacVAE \
  --vae_ckpt /storage/matt_models/cardiac_vae/videos/step_0195000/state.pt \
  --strict_load \
