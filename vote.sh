# modify model_type, norm (l_2, l_inf) and seq (True or False) to chose which model to evaluate. 
# modify weights and solve_for_weights (True or False) for different voting strategy
python examples/voting.py \
    --model_type mnist_small_exact_0_1 \
    --norm l_2 \
    --count 6 \
    --weights None \
    --solve_for_weights \
    --seq