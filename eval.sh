##################### mnist l1 #########################

# mnist small1, small3
# file=/longterm/chi/KwModels/MNIST/small3
# model="${file}/cas_batch_size_50_cascade_6_epochs_60_epsilon_0.3_lr_0.001_norm_test_l1_norm_train_l1_median_opt_adam_print_log_True_proj_50_schedule_length_20_seed_0_starting_epsilon_0.01_best.pth"
# output="${file}/best"

# python examples/evaluate.py \
#     --epsilon 0.1\
#     --proj 50 \
#     --norm l1 \
#     --dataset mnist \
#     --load ${model} \
#     --output ${output} \
#     --verbose 100 \
#     --cuda_ids 2 \
# > ${output}.log 

##################### mnist l1 #########################

# mnist small158
file=/longterm/chi/KwModels/MNIST/small158
model="${file}/1_batch_size_50_cascade_6_epochs_60_epsilon_1.58_lr_0.001_norm_test_l2_norm_train_l2_normal_opt_adam_proj_50_schedule_length_20_seed_0_starting_epsilon_0.01_best.pth"
output="${file}/best"

python examples/evaluate.py \
    --epsilon 1.58\
    --proj 50 \
    --norm l2 \
    --dataset mnist \
    --load ${model} \
    --output ${output} \
    --verbose 100 \
    --cuda_ids 1 \
> ${output}.log 