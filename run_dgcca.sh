#!/bin/bash

DATA_DIR="./data"
RESULTS_DIR="./results"

HIDDEN="32 64"
LATENT=8  
TOP_K=8   
EPS=0.00000001
EPOCHS=150
LR=0.001
B1=0.9
B2=0.999
WD=0.0001
SEED=42

python3 src/main.py \
    --input_dir "$DATA_DIR" \
    --output_dir "$RESULTS_DIR" \
    --hidden_layers $HIDDEN \
    --latent_dims $LATENT \
    --top_k $TOP_K \
    --eps $EPS \
    --epochs $EPOCHS \
    --lr $LR \
    --beta1 $B1 \
    --beta2 $B2 \
    --weight_decay $WD \
    --seed $SEED