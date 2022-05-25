# modify model_type, norm (l_2, l_inf) and seq (True or False) to chose which model to evaluate. 
# modify weights and solve_for_weights (True or False) for different voting strategy
# count needs changing according to the number of models, for non_seq_trained case, count=3
python examples/voting.py \
    --model_type mnist_small_exact_0_1 \
    --norm l_inf \
    --count 3 \
    --weights None \
    --solve_for_weights 
    --seq