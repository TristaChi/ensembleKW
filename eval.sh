############################################## mnist l1 ##################################################

# mnist small1, small3
# file=/longterm/chi/KwModels/MNIST/small3
# model="${file}/cas_batch_size_50_cascade_6_epochs_60_epsilon_0.3_lr_0.001_norm_test_l1_norm_train_l1_median_opt_adam_print_log_True_proj_50_schedule_length_20_seed_0_starting_epsilon_0.01_best.pth"

# eps=3
# model="/home/chi/NNRobustness/ensembleKW/models/models_scaled/mnist_small_0_${eps}.pth"
# output="/home/chi/NNRobustness/ensembleKW/evalData/l_inf/mnist_small_0_${eps}/mnist_small_0_${eps}_"


# python examples/evaluate.py \
#     --epsilon 0.${eps}\
#     --proj 50 \
#     --norm l1 \
#     --dataset mnist \
#     --load ${model} \
#     --output ${output} \
#     --verbose 100 \
#     --cuda_ids 2 \
# > ${output}.log 


# mnist large1, large3
# eps=3
# model="/home/chi/NNRobustness/ensembleKW/models/models_scaled/mnist_large_0_${eps}.pth"
# output="/home/chi/NNRobustness/ensembleKW/evalData/l_inf/mnist_large_0_${eps}/mnist_large_0_${eps}_"

# python examples/evaluate.py \
#     --model large \
#     --epsilon 0.${eps} \
#     --proj 50 \
#     --norm l1 \
#     --dataset mnist \
#     --load ${model} \
#     --output ${output} \
#     --verbose 100 \
#     --cuda_ids 2 \
# > ${output}.log 

############################################## CIFAR l1.l2 ##################################################
# cifar small2, small8, large, resnet
# epspx=36
# model_type=large
# norm=l2
# # model="/home/chi/NNRobustness/ensembleKW/models/models_scaled/cifar_${model_type}_${epspx}px.pth"
# model="/home/chi/NNRobustness/ensembleKW/models/models_scaled_l2/cifar_${model_type}_${epspx}px.pth"
# # output="/home/chi/NNRobustness/ensembleKW/evalData/l_inf/cifar_${model_type}_${epspx}px/cifar_${model_type}_${epspx}px_"
# output="/home/chi/NNRobustness/ensembleKW/evalData/l_2/cifar_${model_type}/cifar_${model_type}_${epspx}px_"

# # eps=0.0348
# # eps=0.139
# eps=0.157 # l2

# python examples/evaluate.py \
#     --model ${model_type} \
#     --epsilon  ${eps} \
#     --proj 50 \
#     --norm ${norm} \
#     --dataset cifar \
#     --load ${model} \
#     --output ${output} \
#     --verbose 100 \
#     --cuda_ids 1 \
# > ${output}.log 

############################################## mnist l2 ##################################################

# mnist small exact, small, large
model_type=small_exact
norm=l2
model="/home/chi/NNRobustness/ensembleKW/models/models_scaled_l2/mnist_${model_type}.pth"
output="/home/chi/NNRobustness/ensembleKW/evalData/l_2/mnist_${model_type}/cifar_${model_type}_"

eps=1.58

python examples/evaluate.py \
    --model ${model_type} \
    --epsilon  ${eps} \
    --proj 50 \
    --norm ${norm} \
    --dataset mnist \
    --load ${model} \
    --output ${output} \
    --verbose 100 \
    --cuda_ids 0 \
> ${output}.log 