##################### mnist l1 #########################

# mnist small1
file=/longterm/chi/KwModels/MNIST/small1
model="${file}/4_batch_size_50_epochs_60_epsilon_0.1_lr_0.001_norm_test_l1_norm_train_l1_median_opt_adam_proj_50_schedule_length_20_seed_0_starting_epsilon_0.01_best.pth"
output="${file}/eval_4"

nohup \
python examples/mnist_evaluate.py \
    --epsilon 0.1\
    --proj 50 \
    --norm l1 \
    --dataset mnist \
    --load ${model} \
    --output ${output}.output \
    --verbose 100 \
    --cuda_ids 2 \
> ${output}.log 2>&1 &

