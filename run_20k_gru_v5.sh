#!/usr/bin/env bash
FOLDER=20k
NAME=gru_v5
mkdir results_gru_v5_${FOLDER}
mkdir models_gru_v5_${FOLDER}

for SEED in 12093 12345 45678
do
python3 -u -m equation_verification.nn_tree_experiment_new \
    --seed $SEED \
    --train-path data/20k_train.json \
    --validation-path data/20k_valid.json \
    --test-path data/20k_test.json \
    --model-class GRUTreesV5 \
    --checkpoint-every-n-epochs 1 \
    --result-path results_gru_v5_${FOLDER}/exp_${NAME}_${SEED}.json \
    --model-prefix models_gru_v5_${FOLDER}/exp_${NAME}_${SEED} > results_gru_v5_${FOLDER}/exp_${NAME}_${SEED}.log &
done
