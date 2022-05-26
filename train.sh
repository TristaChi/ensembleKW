file=<path_to_models>
################################################## MNIST l_inf ##################################################

####### mnist small exact 1 #######
name=smallExact_1
python examples/mnist.py \
    --epochs 60 \
    --epsilon 0.1 \
    --starting_epsilon 0.01 \
    --schedule_length 20 \
    --prefix ${file}/${name} \
    --verbose 200 \
    --norm_train l1 \
    --norm_test l1 \
> ${file}/${name}.out

####### mnist small, large, eps=1,3 #######
# model_type=small or model_type=large
# eps=1 or eps=3
model_type=small
eps=3
python examples/mnist.py \
    --epochs 60 \
    --model ${model_type} \
    --epsilon 0.${eps} \
    --starting_epsilon 0.01 \
    --schedule_length 20 \
    --proj 50 \
    --prefix ${file}/${model_type}_${eps} \
    --verbose 100 \
    --norm_train l1_median \
    --norm_test l1 \
    --print_log False \
> ${file}/${model_type}_${eps}.out

################################################## MNIST l_2 ##################################################
####### mnist small exact 158 #######
name=smallExact_158
python examples/mnist.py \
    --epochs 60 \
    --epsilon 1.58 \
    --starting_epsilon 0.01 \
    --schedule_length 20 \
    --prefix ${file}/${name} \
    --verbose 200 \
    --norm_train l2 \
    --norm_test l2 \
> ${file}/${name}.out

####### mnist small, large #######
model_type=small
# model_type=small or model_type=large
python examples/mnist.py \
    --epochs 60 \
    --model ${model_type} \
    --epsilon 1.58 \
    --starting_epsilon 0.01 \
    --schedule_length 20 \
    --proj 50 \
    --prefix ${file}/${model_type}_158 \
    --verbose 100 \
    --norm_train l2_normal \
    --norm_test l2 \
    --print_log False \
> ${file}/${model_type}_158.out

################################################## CIFAR l_inf ##################################################
####### cifar small, large, epspx=2,8 #######
# model_type=small or model_type=large
# epspx=2 or epspx=8
model_type=small
epspx=2
eps=0.0348 # epspx=2
# eps=0.139 # epspx=8
python examples/cifar.py \
    --epochs 60 \
    --model ${model_type} \
    --epsilon ${eps} \
    --starting_epsilon 0.001 \
    --schedule_length 20 \
    --proj 50 \
    --prefix ${file}/${model_type}_${eps} \
    --verbose 100 \
    --norm_train l1_median \
    --norm_test l1 \
    --print_log False \
> ${file}/${model_type}_${eps}.out


################################################## CIFAR l_2 ##################################################
####### cifar small, large, epspx=36 #######
# model_type=small or model_type=large
model_type=small
epspx=36
eps=0.157
python examples/cifar.py \
    --epochs 60 \
    --model ${model_type} \
    --epsilon ${eps} \
    --starting_epsilon 0.001 \
    --schedule_length 20 \
    --proj 50 \
    --prefix ${file}/${model_type}_${eps} \
    --verbose 100 \
    --norm_train l2_normal \
    --norm_test l2 \
    --print_log False \
> ${file}/${model_type}_${eps}.out
