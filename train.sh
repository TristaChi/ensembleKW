################################################## MNIST l1 ##################################################
file=/longterm/chi/KwModels/MNIST
# file=./models/MNIST

####### mnist small exact 1 #######
# ??? proj defult value
# name=cas
# python examples/mnist.py \
#     --epochs 60 \
#     --epsilon 0.1 \
#     --starting_epsilon 0.01 \
#     --schedule_length 20 \
#     --prefix ${file}/smallExact1/${name} \
#     --verbose 200 \
#     --norm_train l1_median \
#     --norm_test l1 \
#     --cascade 6 \
#     --cuda_ids 1 \
# > ${file}/smallExact1/${name}.out

####### mnist small1, small3 #######
name=cas
eps=3
python examples/mnist.py \
    --epochs 60 \
    --epsilon 0.${eps} \
    --starting_epsilon 0.01 \
    --schedule_length 20 \
    --proj 50 \
    --prefix ${file}/small${eps}/${name} \
    --verbose 100 \
    --norm_train l1_median \
    --norm_test l1 \
    --cascade 6 \
    --cuda_ids 2 \
    --print_log False \
> ${file}/small${eps}/${name}.out


####### mnist large1 #######
# name=cas
# nohup \
# python examples/mnist.py \
# 	--model large \
#     --epochs 60 \
#     --epsilon 0.1 \
#     --starting_epsilon 0.01 \
#     --schedule_length 20 \
#     --proj 50 \
#     --prefix ${file}/large1/${name} \
#     --verbose 200 \
#     --norm_train l1_median \
#     --norm_test l1 \
#     --cascade 6 \
#     --test_batch_size 4 \
#     --cuda_ids 1 \
#     --print_log False \
# > ${file}/large1/${name}.log 2>&1 &


################################################## MNIST l2 ##################################################
# file=/longterm/chi/KwModels/MNIST

####### mnist small158 #######
# name=cas
# nohup \
# python examples/mnist.py \
#     --epochs 60 \
#     --epsilon 1.58 \
#     --starting_epsilon 0.01 \
#     --schedule_length 20 \
#     --proj 50 \
#     --prefix ${file}/small158/${name} \
#     --verbose 100 \
#     --norm_train l2_normal \
#     --norm_test l2 \
#     --cascade 6 \
#     --cuda_ids 2 \
#     --print_log False \
# > ${file}/small158/${name}.out 2>&1 &


####### mnist large1 #######
# name=cas
# nohup \
# python examples/mnist.py \
# 	--model large \
#     --epochs 60 \
#     --epsilon 0.1 \
#     --starting_epsilon 0.01 \
#     --schedule_length 20 \
#     --proj 50 \
#     --prefix ${file}/large1/${name} \
#     --verbose 200 \
#     --norm_train l1_median \
#     --norm_test l1 \
#     --cascade 6 \
#     --test_batch_size 4 \
#     --cuda_ids 1 \
#     --print_log False \
# > ${file}/large1/${name}.log 2>&1 &



################################################## CIFAR l1 ##################################################
file=/longterm/chi/KwModels/CIFAR
# file=./models/CIFAR

####### cifar small2 #######
# name=cas
# python examples/cifar.py \
#     --epochs 60 \
#     --epsilon 0.0348 \
#     --starting_epsilon 0.001 \
#     --schedule_length 20 \
#     --proj 50 \
#     --prefix ${file}/small2/${name} \
#     --verbose 200 \
#     --norm_train l1_median \
#     --norm_test l1 \
#     --cascade 6 \
#     --test_batch_size 25 \
#     --cuda_ids 2 \
#     --print_log False \
# > ${file}/small2/${name}.out 


####### cifar large2 #######
# name=cas
# nohup \
# python examples/cifar.py \
#     --epochs 60 \
#     --epsilon 0.139 \
#     --test_batch_size 4 \
#     --model large \
#     --starting_epsilon 0.001 \
#     --schedule_length 20 \
#     --proj 50 \
#     --prefix ${file}/large2/${name} \
#     --verbose 200 \
#     --norm_train l1_median \
#     --norm_test l1 \
#     --cascade 6 \
#     --cuda_ids 0 \
#     --print_log False \
# > ${file}/large2/${name}.out 2>&1 &
