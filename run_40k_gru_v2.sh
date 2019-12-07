#!/usr/bin/env bash
FOLDER=40k
NAME=gru_v2
mkdir results_gru_v2_${FOLDER}
mkdir models_gru_v2_${FOLDER}

for SEED in 12093 12345 45678
do
python3 -u -m equation_verification.nn_tree_experiment_new \
    --seed $SEED \
    --train-path data/40k_train.json \
    --validation-path data/40k_val_shallow.json \
    --test-path data/40k_test.json \
    --model-class GRUTreesV2 \
    --checkpoint-every-n-epochs 1 \
    --result-path results_gru_v2_${FOLDER}/exp_${NAME}_${SEED}.json \
    --model-prefix models_gru_v2_${FOLDER}/exp_${NAME}_${SEED} > results_gru_v2_${FOLDER}/exp_${NAME}_${SEED}.log &
done
