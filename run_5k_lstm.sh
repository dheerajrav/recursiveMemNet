#!/usr/bin/env bash
FOLDER=ODE
NAME=LSTM_ODE
mkdir results_${FOLDER}
mkdir models_${FOLDER}

for SEED in 12093
do
python3 -u -m equation_verification.nn_tree_experiment_arth \
    --seed $SEED \
    --train-path data/5k_train.json \
    --validation-path data/5k_train.json \
    --test-path data/5k_train.json \
    --model-class LSTMTrees \
    --verbose \
    --checkpoint-every-n-epochs 1 \
    --result-path results_${FOLDER}/exp_${NAME}_${SEED}.json \
    --model-prefix models_${FOLDER}/exp_${NAME}_${SEED} > results_${FOLDER}/exp_${NAME}_${SEED}.log &
done
