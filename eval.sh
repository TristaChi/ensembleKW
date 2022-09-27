# ############################################## Mnist linf ##################################################
# # small exact, sequentially trained
# eps=1
# model_type=small_exact
# norm=l1
# model="models/seq_trained/l_inf/mnist_${model_type}_0_${eps}.pth"
# output="evalData/seq_trained/l_inf/mnist_${model_type}_0_${eps}/mnist_${model_type}_0_${eps}_"
# python examples/evaluate.py \
#     --model ${model_type} \
#     --epsilon  0.${eps} \
#     --norm ${norm} \
#     --dataset mnist \
#     --load ${model} \
#     --output ${output} \
#     --verbose 100 \
#     --cuda_ids 1 \
# > ${output}.log 

# # mnist small exact, non-sequentially trained
# # run with id=1 and id=2 to evaluate two constituent models
# eps=1
# model_type=small_exact
# norm=l1
# id=1
# model="models/non_seq_trained/l_inf/more_mnist_${model_type}_0_${eps}_${id}.pth"
# output="evalData/non_seq_trained/l_inf/mnist_${model_type}_0_${eps}/more_mnist_${model_type}_0_${eps}_${id}"
# python examples/evaluate.py \
#     --model ${model_type} \
#     --epsilon  0.${eps} \
#     --norm ${norm} \
#     --dataset mnist \
#     --load ${model} \
#     --output ${output} \
#     --verbose 100 \
#     --cuda_ids 1 \
# > ${output}.log 

# cp evalData/seq_trained/l_inf/mnist_${model_type}_0_${eps}/mnist_${model_type}_0_${eps}_0_train evalData/non_seq_trained/l_inf/mnist_${model_type}_0_${eps}/more_mnist_${model_type}_0_${eps}_0_train
# cp evalData/seq_trained/l_inf/mnist_${model_type}_0_${eps}/mnist_${model_type}_0_${eps}_0_test evalData/non_seq_trained/l_inf/mnist_${model_type}_0_${eps}/more_mnist_${model_type}_0_${eps}_0_test

# # mnist small, large, eps = 1,3, sequentially trained
# # model_type=small or model_type=large
# # eps=1 or eps=3
# eps=1
# model_type=large
# norm=l1
# model="models/seq_trained/l_inf/mnist_${model_type}_0_${eps}.pth"
# output="evalData/seq_trained/l_inf/mnist_${model_type}_0_${eps}/mnist_${model_type}_0_${eps}_"
# python examples/evaluate.py \
#     --model ${model_type} \
#     --epsilon  0.${eps} \
#     --proj 50 \
#     --norm ${norm} \
#     --dataset mnist \
#     --load ${model} \
#     --output ${output} \
#     --verbose 100 \
#     --cuda_ids 1 \
# > ${output}.log 

# # mnist small, eps = 1,3, non-sequentially trained
# # run with id=1 and id=2 to evaluate two constituent models
# # eps=1 or eps=3
# eps=1
# model_type=small
# norm=l1
# id=1
# model="models/non_seq_trained/l_inf/more_mnist_${model_type}_0_${eps}_${id}.pth"
# output="evalData/non_seq_trained/l_inf/mnist_${model_type}_0_${eps}/more_mnist_${model_type}_0_${eps}_${id}"
# python examples/evaluate.py \
#     --model ${model_type} \
#     --epsilon  0.${eps} \
#     --proj 50 \
#     --norm ${norm} \
#     --dataset mnist \
#     --load ${model} \
#     --output ${output} \
#     --verbose 100 \
#     --cuda_ids 1 \
# > ${output}.log 

# cp evalData/seq_trained/l_inf/mnist_${model_type}_0_${eps}/mnist_${model_type}_0_${eps}_0_train evalData/non_seq_trained/l_inf/mnist_${model_type}_0_${eps}/more_mnist_${model_type}_0_${eps}_0_train
# cp evalData/seq_trained/l_inf/mnist_${model_type}_0_${eps}/mnist_${model_type}_0_${eps}_0_test evalData/non_seq_trained/l_inf/mnist_${model_type}_0_${eps}/more_mnist_${model_type}_0_${eps}_0_test

# ############################################## mnist l2 ##################################################

# # small exact, sequentially trained
# eps=158
# model_type=small_exact
# norm=l2
# model="models/seq_trained/l_2/mnist_${model_type}.pth"
# output="evalData/seq_trained/l_2/mnist_${model_type}/mnist_${model_type}_"
# python examples/evaluate.py \
#     --model ${model_type} \
#     --epsilon  1.58 \
#     --norm ${norm} \
#     --dataset mnist \
#     --load ${model} \
#     --output ${output} \
#     --verbose 100 \
#     --cuda_ids 1 \
# > ${output}.log 

# # mnist small exact, non-sequentially trained
# # run with id=1 and id=2 to evaluate two constituent models
# eps=158
# model_type=small_exact
# norm=l2
# id=1
# model="models/non_seq_trained/l_2/more_mnist_${model_type}_${id}.pth"
# output="evalData/non_seq_trained/l_2/mnist_${model_type}/more_mnist_${model_type}_${id}"
# python examples/evaluate.py \
#     --model ${model_type} \
#     --epsilon  1.58 \
#     --norm ${norm} \
#     --dataset mnist \
#     --load ${model} \
#     --output ${output} \
#     --verbose 100 \
#     --cuda_ids 1 \
# > ${output}.log 

# cp evalData/seq_trained/l_2/mnist_${model_type}/mnist_${model_type}_0_train evalData/non_seq_trained/l_2/mnist_${model_type}/more_mnist_${model_type}_0_train
# cp evalData/seq_trained/l_2/mnist_${model_type}/mnist_${model_type}_0_test evalData/non_seq_trained/l_2/mnist_${model_type}/more_mnist_${model_type}_0_test

# # mnist small, large, sequentially trained
# # model_type=small or model_type=large
# eps=158
# model_type=large
# norm=l2
# model="models/seq_trained/l_2/mnist_${model_type}.pth"
# output="evalData/seq_trained/l_2/mnist_${model_type}/mnist_${model_type}_"
# python examples/evaluate.py \
#     --model ${model_type} \
#     --epsilon  1.58 \
#     --proj 50 \
#     --norm ${norm} \
#     --dataset mnist \
#     --load ${model} \
#     --output ${output} \
#     --verbose 100 \
#     --cuda_ids 1 \
# > ${output}.log 

# # mnist small, non-sequentially trained
# # run with id=1 and id=2 to evaluate two constituent models
# eps=158
# model_type=small
# norm=l2
# id=1
# model="models/non_seq_trained/l_2/more_mnist_${model_type}_${id}.pth"
# output="evalData/non_seq_trained/l_2/mnist_${model_type}/more_mnist_${model_type}_${id}"
# python examples/evaluate.py \
#     --model ${model_type} \
#     --epsilon  1.58 \
#     --proj 50 \
#     --norm ${norm} \
#     --dataset mnist \
#     --load ${model} \
#     --output ${output} \
#     --verbose 100 \
#     --cuda_ids 1 \
# > ${output}.log 

# cp evalData/seq_trained/l_2/mnist_${model_type}/mnist_${model_type}_0_train evalData/non_seq_trained/l_2/mnist_${model_type}/more_mnist_${model_type}_0_train
# cp evalData/seq_trained/l_2/mnist_${model_type}/mnist_${model_type}_0_test evalData/non_seq_trained/l_2/mnist_${model_type}/more_mnist_${model_type}_0_test

# ############################################## CIFAR linf ##################################################
# # cifar small, large, epspx=2,8, sequentially trained
# # model_type=small or model_type=large
# # epspx=2 or epspx=8
# epspx=2
# model_type=large
# norm=l1
# model="models/seq_trained/l_inf/cifar_${model_type}_${epspx}px.pth"
# output="evalData/seq_trained/l_inf/cifar_${model_type}_${epspx}px/cifar_${model_type}_${epspx}px_"
# eps=0.0348 # epspx=2
# # eps=0.139 # epspx=8
# python examples/evaluate.py \
#     --model ${model_type} \
#     --epsilon  ${eps} \
#     --proj 50 \
#     --norm ${norm} \
#     --dataset cifar \
#     --load ${model} \
#     --output ${output} \
#     --verbose 100 \
#     --cuda_ids 2 \
# > ${output}.log 

# # cifar small epspx=2,8, non-sequentially trained
# # epspx=2 or epspx=8
# # run with id=1 and id=2 to evaluate two constituent models
# epspx=2
# model_type=small
# norm=l1
# model="models/non_seq_trained/l_inf/more_cifar_${model_type}_${epspx}px_${id}.pth"
# output="evalData/non_seq_trained/l_inf/cifar_${model_type}_${epspx}px/more_cifar_${model_type}_${epspx}px_${id}"
# eps=0.0348 # epspx=2
# # eps=0.139 # epspx=8
# python examples/evaluate.py \
#     --model ${model_type} \
#     --epsilon  ${eps} \
#     --proj 50 \
#     --norm ${norm} \
#     --dataset cifar \
#     --load ${model} \
#     --output ${output} \
#     --verbose 100 \
#     --cuda_ids 2 \
# > ${output}.log 

# cp evalData/seq_trained/l_inf/cifar_${model_type}_${epspx}px/cifar_${model_type}_${epspx}px_0_train evalData/non_seq_trained/l_inf/cifar_${model_type}_${epspx}px/more_cifar_${model_type}_${epspx}px_0_train
# cp evalData/seq_trained/l_inf/cifar_${model_type}_${epspx}px/cifar_${model_type}_${epspx}px_0_test evalData/non_seq_trained/l_inf/cifar_${model_type}_${epspx}px/more_cifar_${model_type}_${epspx}px_0_test

# # cifar ResNet epspx=2,8, sequentially trained
# # epspx=2 or epspx=8
# epspx=2
# model_type=resnet
# norm=l1
# model="models/seq_trained/l_inf/cifar_${model_type}_${epspx}px.pth"
# output="evalData/seq_trained/l_inf/cifar_${model_type}_${epspx}px/cifar_${model_type}_${epspx}px_"
# eps=0.0348 # epspx=2
# # eps=0.139 # epspx=8
# python examples/evaluate.py \
#     --model ${model_type} \
#     --epsilon  ${eps} \
#     --proj 50 \
#     --norm ${norm} \
#     --dataset cifar \
#     --load ${model} \
#     --output ${output} \
#     --verbose 100 \
#     --cuda_ids 0,1 \
# > ${output}.log 
############################################## CIFAR l2 ##################################################
# cifar small, large, sequentially trained
# model_type=small or model_type=large
epspx=36
model_type=large
norm=l2
model="models/seq_trained/l_2/cifar_${model_type}_${epspx}px.pth"
output="evalData/seq_trained/l_2/cifar_${model_type}_${epspx}px/cifar_${model_type}_${epspx}px_"
eps=0.627 # l2
python examples/evaluate.py \
    --model ${model_type} \
    --epsilon  ${eps} \
    --proj 50 \
    --norm ${norm} \
    --dataset cifar \
    --load ${model} \
    --output ${output} \
    --verbose 100 \
    --cuda_ids 0,1 \
> ${output}.log 

# # cifar small, non-sequentially trained
# # run with id=1 and id=2 to evaluate two constituent models
# epspx=36
# model_type=small
# norm=l2
# model="models/non_seq_trained/l2/more_cifar_${model_type}_${epspx}px_${id}.pth"
# output="evalData/non_seq_trained/l_2/cifar_${model_type}_${epspx}px/more_cifar_${model_type}_${epspx}px_${id}"
# eps=0.627 # l2
# python examples/evaluate.py \
#     --model ${model_type} \
#     --epsilon  ${eps} \
#     --proj 50 \
#     --norm ${norm} \
#     --dataset cifar \
#     --load ${model} \
#     --output ${output} \
#     --verbose 100 \
#     --cuda_ids 2 \
# > ${output}.log 

# cp evalData/seq_trained/l_2/cifar_${model_type}_${epspx}px/cifar_${model_type}_${epspx}px_0_train evalData/non_seq_trained/l_2/cifar_${model_type}_${epspx}px/more_cifar_${model_type}_${epspx}px_0_train
# cp evalData/seq_trained/l_2/cifar_${model_type}_${epspx}px/cifar_${model_type}_${epspx}px_0_test evalData/non_seq_trained/l_2/cifar_${model_type}_${epspx}px/more_cifar_${model_type}_${epspx}px_0_test

# # cifar ResNet epspx=36, sequentially trained
# epspx=36
# model_type=resnet
# norm=l2
# model="models/seq_trained/l_2/cifar_${model_type}_${epspx}px.pth"
# output="evalData/seq_trained/l_2/cifar_${model_type}_${epspx}px/cifar_${model_type}_${epspx}px_"
# eps=0.627 # l2
# python examples/evaluate.py \
#     --model ${model_type} \
#     --epsilon  ${eps} \
#     --proj 50 \
#     --norm ${norm} \
#     --dataset cifar \
#     --load ${model} \
#     --output ${output} \
#     --verbose 100 \
#     --cuda_ids 0,1 \
# > ${output}.log 