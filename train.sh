# shopt -s expand_aliases
# alias docker_python='docker run --rm --gpus all -it --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --network="host" -v /home/$USER:/home/$USER -v /longterm/$USER:/longterm/$USER -v /home/$USER/.keras/datasets:/home/$USER/.keras -v /longterm/chi/tensorflow_dataset:/longterm/chi/tensorflow_dataset -w /home/$USER/RandomSmooting/convex_adversarial/examples/ nvcr.io/nvidia/pytorch:chi python'

# mnist large1
# nohup \
# python examples/mnist.py \
# 	--model large \
#     --epochs 60 \
#     --epsilon 0.1 \
#     --starting_epsilon 0.01 \
#     --schedule_length 20 \
#     --proj 50 \
#     --prefix /longterm/chi/KwModels/MNIST/large1/1 \
#     --verbose 200 \
#     --norm_train l1_median \
#     --norm_test l1 \
#     --test_batch_size 8 \
#     --cuda_ids 1 \
#     > /longterm/chi/KwModels/MNIST/large1/1.log 2>&1 &

# mnist small1
nohup \
python examples/mnist.py \
    --epochs 60 \
    --epsilon 0.1 \
    --starting_epsilon 0.01 \
    --schedule_length 20 \
    --proj 50 \
    --prefix /longterm/chi/KwModels/MNIST/small1/4 \
    --verbose 100 \
    --norm_train l1_median \
    --norm_test l1 \
    --cuda_ids 2 \
    > /longterm/chi/KwModels/MNIST/small1/4.log 2>&1 &

