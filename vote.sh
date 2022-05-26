# Modify --model_type (mnist_small_exact_0_1, mnist_small_0_1, mnist_small_0_3, mnist_large_0_1, mnist_large_0_3, mnist_small_exact, mnist_small, mnist_large, cifar_small_2px, cifar_small_8px, cifar_large_2px, cifar_large_8px, cifar_small_36px,cifar_large_36px), --norm (l_2, l_inf) and --seq (include or exclude flag) to chose which model to evaluate. 
# Modify --solve_for_weights (include or exclude flag) to toggle between weighted voting and uniform voting strategy. The cascading strategy is always evaluated.
# Modify --count according to the number of constituent models. These numbers are available in Tables 5,6,7, and 8 in the appendix.
python examples/voting.py \
    --model_type mnist_small_exact_0_1 \
    --norm l_inf \
    --count 3 \
    --weights None \
    --solve_for_weights 
    --seq
