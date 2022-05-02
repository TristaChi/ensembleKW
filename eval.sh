# dataset = mnist
# file = /longterm/chi/KwModels/MNIST/small1
# model = ${file}/2_batch_size_50_epochs_60_epsilon_0.1_lr_0.001_norm_test_l1_norm_train_l1_median_opt_adam_proj_50_schedule_length_20_seed_0_starting_epsilon_0.01_best.pth
# output = ${file}/eval_2.output

# nohup \
# python examples/${dataset}_evaluate.py \
#     --epsilon 0.1\
#     --proj 50 \
#     --norm l1 \
#     --dataset mnist \
#     --load ${model} \
#     --output ${output} \
#     --verbose 100 \
#     --cuda_ids 2 \
# > ${file}/eval_2.log 2>&1 &

