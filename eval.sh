############################################## Mnist l1 ##################################################
# small exact, sequencial trained
eps=1
model_type=small_exact
norm=l1
model="models/seq_trained/l_inf/mnist_${model_type}_0_${eps}.pth"
output="evalData/seq_trained/l_inf/mnist_${model_type}_0_${eps}/mnist_${model_type}_0_${eps}_"
python examples/evaluate.py \
    --model ${model_type} \
    --epsilon  ${eps} \
    --norm ${norm} \
    --dataset mnist \
    --load ${model} \
    --output ${output} \
    --verbose 100 \
    --cuda_ids 1 \
> ${output}.log 

# mnist small exact, non sequencial trained, count=1,2
eps=1
model_type=small_exact
norm=l1
count=1
model="models/non_seq_trained/l_inf/more_mnist${model_type}${eps}_${count}.pth"
output="evalData/non_seq_trained/l_inf/mnist_${model_type}_0_${eps}/more_mnist_${model_type}_0_${eps}_${count}"
python examples/evaluate.py \
    --model ${model_type} \
    --epsilon  ${eps} \
    --norm ${norm} \
    --dataset mnist \
    --load ${model} \
    --output ${output} \
    --verbose 100 \
    --cuda_ids 1 \
> ${output}.log 

# mnist small, large, eps = 1,3, sequencial trained
eps=1
model_type=large
norm=l1
model="models/seq_trained/l_inf/mnist_${model_type}_0_${eps}.pth"
output="evalData/seq_trained/l_inf/mnist_${model_type}_0_${eps}/mnist_${model_type}_0_${eps}_"
python examples/evaluate.py \
    --model ${model_type} \
    --epsilon  ${eps} \
    --proj 50 \
    --norm ${norm} \
    --dataset mnist \
    --load ${model} \
    --output ${output} \
    --verbose 100 \
    --cuda_ids 1 \
> ${output}.log 

# mnist small, eps = 1,3, non sequencial trained, count = 1,2
eps=1
model_type=small
norm=l1
count=1
model="models/non_seq_trained/l_inf/more_mnist${model_type}${eps}_${count}.pth"
output="evalData/non_seq_trained/l_inf/mnist_${model_type}_0_${eps}/more_mnist_${model_type}_0_${eps}_${count}"
python examples/evaluate.py \
    --model ${model_type} \
    --epsilon  ${eps} \
    --proj 50 \
    --norm ${norm} \
    --dataset mnist \
    --load ${model} \
    --output ${output} \
    --verbose 100 \
    --cuda_ids 1 \
> ${output}.log 

############################################## mnist l2 ##################################################

# small exact, sequencial trained
eps=158
model_type=small_exact
norm=l2
model="models/seq_trained/l_2/mnist_${model_type}.pth"
output="evalData/seq_trained/l_2/mnist_${model_type}/mnist_${model_type}_"
python examples/evaluate.py \
    --model ${model_type} \
    --epsilon  1.58 \
    --norm ${norm} \
    --dataset mnist \
    --load ${model} \
    --output ${output} \
    --verbose 100 \
    --cuda_ids 1 \
> ${output}.log 

# mnist small exact, non sequencial trained, count=1,2
eps=1
model_type=small_exact
norm=l2
count=1
model="models/seq_trained/l_2/more_mnist_${model_type}_${count}.pth"
output="evalData/seq_trained/l_2/mnist_${model_type}/more_mnist_${model_type}_${count}"
python examples/evaluate.py \
    --model ${model_type} \
    --epsilon  ${eps} \
    --norm ${norm} \
    --dataset mnist \
    --load ${model} \
    --output ${output} \
    --verbose 100 \
    --cuda_ids 1 \
> ${output}.log 

# mnist small, large, eps = 1,3, sequencial trained
eps=1
model_type=large
norm=l1
model="models/seq_trained/l_2/mnist_${model_type}.pth"
output="evalData/seq_trained/l_2/mnist_${model_type}/mnist_${model_type}_"
python examples/evaluate.py \
    --model ${model_type} \
    --epsilon  ${eps} \
    --proj 50 \
    --norm ${norm} \
    --dataset mnist \
    --load ${model} \
    --output ${output} \
    --verbose 100 \
    --cuda_ids 1 \
> ${output}.log 

# mnist small, eps = 1,3, non sequencial trained, count = 1,2
eps=1
model_type=small
norm=l1
count=1
model="models/seq_trained/l_2/more_mnist_${model_type}_${count}.pth"
output="evalData/seq_trained/l_2/mnist_${model_type}/more_mnist_${model_type}_${count}"
python examples/evaluate.py \
    --model ${model_type} \
    --epsilon  ${eps} \
    --proj 50 \
    --norm ${norm} \
    --dataset mnist \
    --load ${model} \
    --output ${output} \
    --verbose 100 \
    --cuda_ids 1 \
> ${output}.log 

############################################## CIFAR l1 ##################################################
# cifar small, large, epspx=2,8
epspx=36
model_type=large
norm=l1
model="models/models_scaled/cifar_${model_type}_${epspx}px.pth"
output="evalData/l_inf/cifar_${model_type}_${epspx}px/cifar_${model_type}_${epspx}px_"
eps=0.0348 # epspx=2
# eps=0.139 # epspx=8
python examples/evaluate.py \
    --model ${model_type} \
    --epsilon  ${eps} \
    --proj 50 \
    --norm ${norm} \
    --dataset cifar \
    --load ${model} \
    --output ${output} \
    --verbose 100 \
    --cuda_ids 2 \
> ${output}.log 

############################################## CIFAR l2 ##################################################

# cifar small, large
epspx=36
model_type=large
norm=l2
model="models/models_scaled_l2/cifar_${model_type}_${epspx}px.pth"
output="evalData/l_2/cifar_${model_type}_${epspx}px/cifar_${model_type}_${epspx}px_"
eps=0.157 # l2
python examples/evaluate.py \
    --model ${model_type} \
    --epsilon  ${eps} \
    --proj 50 \
    --norm ${norm} \
    --dataset cifar \
    --load ${model} \
    --output ${output} \
    --verbose 100 \
    --cuda_ids 2 \
> ${output}.log 



