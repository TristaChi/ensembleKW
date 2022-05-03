######################### MNIST l1 #########################
# file=/longterm/chi/KwModels/MNIST

####### mnist small1 #######
# count=1
# nohup \
# python examples/mnist.py \
#     --epochs 60 \
#     --epsilon 0.1 \
#     --starting_epsilon 0.01 \
#     --schedule_length 20 \
#     --proj 50 \
#     --prefix ${file}/small1/${count} \
#     --verbose 100 \
#     --norm_train l1_median \
#     --norm_test l1 \
#     --cuda_ids 2 \
#     --print_log False \
# > ${file}/small1/${count}.log 2>&1 &


####### mnist large1 #######
# count=1
# nohup \
# python examples/mnist.py \
# 	--model large \
#     --epochs 60 \
#     --epsilon 0.1 \
#     --starting_epsilon 0.01 \
#     --schedule_length 20 \
#     --proj 50 \
#     --prefix ${file}/large1/${count} \
#     --verbose 200 \
#     --norm_train l1_median \
#     --norm_test l1 \
#     --test_batch_size 8 \
#     --cuda_ids 1 \
#     --print_log False \
# > ${file}/large1/${count}.log 2>&1 &


######################### CIFAR l1 #########################
file=/longterm/chi/KwModels/CIFAR

####### cifar small2 #######
count=1
nohup \
python examples/cifar.py \
    --epochs 60 \
    --epsilon 0.139 \
    --starting_epsilon 0.001 \
    --schedule_length 20 \
    --proj 50 \
    --prefix ${file}/small2/${count} \
    --verbose 200 \
    --norm_train l1_median \
    --norm_test l1 \
    --test_batch_size 25 \
    --cuda_ids 2 \
    --print_log False \
> ${file}/small2/${count}.log 2>&1 &


####### cifar large2 #######
count=1
nohup \
python examples/cifar.py \
    --epochs 60 \
    --epsilon 0.139 \
    --test_batch_size 8 \
    --model large \
    --starting_epsilon 0.001 \
    --schedule_length 20 \
    --proj 50 \
    --prefix ${file}/large2/${count} \
    --verbose 200 \
    --norm_train l1_median \
    --norm_test l1 \
    --cuda_ids 2 \
    --print_log False \
> ${file}/large2/${count}.log 2>&1 &
