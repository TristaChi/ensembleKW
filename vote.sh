
python examples/voting.py \
    --model_type mnist_small_exact_0_1 \
    --root evalData/l_inf/ \
    --count 6 \
    --weights None \
#     --solve_for_weights True
# python examples/voting.py \
#     --model_type mnist_small_0_1 \
#     --root evalData/l_inf/ \
#     --count 7 \
#     --weights None \
#     # --solve_for_weights True
# python examples/voting.py \
#     --model_type mnist_small_0_3 \
#     --root evalData/l_inf/ \
#     --count 3 \
#     --weights None \
# #     --solve_for_weights True
# # python examples/voting.py \
# #     --model_type mnist_large_0_1 \
# #     --root evalData/l_inf/ \
# #     --count 5 \
# #     --weights None \
# #     --solve_for_weights $solve
# # python examples/voting.py \
# #     --model_type mnist_large_0_3 \
# #     --root evalData/l_inf/ \
# #     --count 3 \
# #     --weights None \
# #     --solve_for_weights $solve
# python examples/voting.py \
#     --model_type cifar_small_2px \
#     --root evalData/l_inf/ \
#     --count 5 \
#     --weights None \
# #     --solve_for_weights True
# python examples/voting.py \
#     --model_type cifar_small_8px \
#     --root evalData/l_inf/ \
#     --count 3 \
#     --weights None \
# #     --solve_for_weights True
# # python examples/voting.py \
# #     --model_type cifar_large_2px \
# #     --root evalData/l_inf/ \
# #     --count 4 \
# #     --weights None \
# #     --solve_for_weights True
# python examples/voting.py \
#     --model_type cifar_large_8px \
#     --root evalData/l_inf/ \
#     --count 2 \
#     --weights None \
#     --solve_for_weights True
python examples/voting.py \
    --model_type cifar_small_36px \
    --root evalData/l_2/ \
    --count 2 \
    --weights None
python examples/voting.py \
    --model_type mnist_large \
    --root evalData/l_2/ \
    --count 6 \
    --weights None
python examples/voting.py \
    --model_type mnist_small \
    --root evalData/l_2/ \
    --count 6 \
    --weights None
python examples/voting.py \
    --model_type mnist_small_exact \
    --root evalData/l_2/ \
    --count 6 \
    --weights None