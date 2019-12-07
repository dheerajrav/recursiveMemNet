#!/usr/bin/env bash
FOLDER=20k
NAME=gru_v6
mkdir results_gru_v6_${FOLDER}
mkdir models_gru_v6_${FOLDER}

for SEED in 12093 12345 45678
do
python3 -u -m equation_verification.nn_tree_experiment_new \
    --seed $SEED \
    --train-path data/20k_train.json \
    --validation-path data/20k_valid.json \
    --test-path data/20k_test.json \
    --model-class GRUTreesV6 \
    --checkpoint-every-n-epochs 1 \
    --result-path results_gru_v6_${FOLDER}/exp_${NAME}_${SEED}.json \
    --model-prefix models_gru_v6_${FOLDER}/exp_${NAME}_${SEED} > results_gru_v6_${FOLDER}/exp_${NAME}_${SEED}.log &
done
