############################################## Mnist l1 ##################################################
# mnist small exact, small, large
# for eps in 1 3
# do
#     # eps=1

#     model_type=large
#     norm=l1
#     model="/home/chi/NNRobustness/ensembleKW/models/models_scaled/mnist_${model_type}_0_${eps}.pth"
#     output="/home/chi/NNRobustness/ensembleKW/evalData/l_inf/mnist_${model_type}_0_${eps}/mnist_${model_type}_0_${eps}_"


#     python examples/evaluate.py \
#         --model ${model_type} \
#         --epsilon  ${eps} \
#         --proj 50 \
#         --norm ${norm} \
#         --dataset mnist \
#         --load ${model} \
#         --output ${output} \
#         --verbose 100 \
#         --cuda_ids 1 \
#     > ${output}.log 

# done


############################################## CIFAR l1.l2 ##################################################
# cifar small2, small8, large, resnet
epspx=36
model_type=large
norm=l2
# model="/home/chi/NNRobustness/ensembleKW/models/models_scaled/cifar_${model_type}_${epspx}px.pth"
model="/home/chi/NNRobustness/ensembleKW/models/models_scaled_l2/cifar_${model_type}_${epspx}px.pth"
# output="/home/chi/NNRobustness/ensembleKW/evalData/l_inf/cifar_${model_type}_${epspx}px/cifar_${model_type}_${epspx}px_"
output="/home/chi/NNRobustness/ensembleKW/evalData/l_2/cifar_${model_type}_${epspx}px/cifar_${model_type}_${epspx}px_"

# eps=0.0348
# eps=0.139
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

############################################## mnist l2 ##################################################

# mnist small exact, small, large
# model_type=large
# norm=l2
# model="/home/chi/NNRobustness/ensembleKW/models/models_scaled_l2/mnist_${model_type}.pth"
# output="/home/chi/NNRobustness/ensembleKW/evalData/l_2/mnist_${model_type}/mnist_${model_type}_"

# eps=1.58

# python examples/evaluate.py \
#     --model ${model_type} \
#     --epsilon  ${eps} \
#     --proj 50 \
#     --norm ${norm} \
#     --dataset mnist \
#     --load ${model} \
#     --output ${output} \
#     --verbose 100 \
#     --cuda_ids 1 \
# > ${output}.log 


############################################## delete ##################################################

# epspx=8
# model_type=large
# norm=l1
# model="/home/chi/NNRobustness/ensembleKW/models/models_scaled/cifar_${model_type}_${epspx}px.pth"
# output="/home/chi/NNRobustness/ensembleKW/evalData/l_inf/cifar_${model_type}_${epspx}px/cifar_${model_type}_${epspx}px_"
# eps=0.139
# python examples/evaluate.py \
#     --model ${model_type} \
#     --epsilon  ${eps} \
#     --proj 50 \
#     --norm ${norm} \
#     --dataset cifar \
#     --load ${model} \
#     --output ${output} \
#     --verbose 100 \
#     --cuda_ids 0 \
# > ${output}.log 

# epspx=8
# model_type=small
# norm=l1
# model="/home/chi/NNRobustness/ensembleKW/models/models_scaled/cifar_${model_type}_${epspx}px.pth"
# output="/home/chi/NNRobustness/ensembleKW/evalData/l_inf/cifar_${model_type}_${epspx}px/cifar_${model_type}_${epspx}px_"
# eps=0.139
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

# epspx=2
# model_type=small
# norm=l1
# model="/home/chi/NNRobustness/ensembleKW/models/models_scaled/cifar_${model_type}_${epspx}px.pth"
# output="/home/chi/NNRobustness/ensembleKW/evalData/l_inf/cifar_${model_type}_${epspx}px/cifar_${model_type}_${epspx}px_"
# eps=0.0348
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