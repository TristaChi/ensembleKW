python examples/attack_ensemble.py \
    --config=examples/configs/attack.py \
    --config.model.architecture="exact" \
    --config.model.directory="/home/zifanw/ensembleKW/models/seq_trained/l_2/mnist_small_exact.pth" \
    --config.attack.norm="l2" \
    --config.attack.eps=1.58 \
    --config.attack.step_size=0.03 

python examples/attack_ensemble.py \
    --config=examples/configs/attack.py \
    --config.model.architecture="small" \
    --config.model.directory="/home/zifanw/ensembleKW/models/seq_trained/l_2/mnist_small.pth" \
    --config.attack.norm="l2" \
    --config.attack.eps=1.58 \
    --config.attack.step_size=0.03 

python examples/attack_ensemble.py \
    --config=examples/configs/attack.py \
    --config.model.architecture="large" \
    --config.model.directory="/home/zifanw/ensembleKW/models/seq_trained/l_2/mnist_large.pth" \
    --config.data.batch_size=8 \
    --config.attack.norm="l2" \
    --config.attack.eps=1.58 \
    --config.attack.step_size=0.03 


python examples/attack_ensemble.py \
    --config=examples/configs/attack.py \
    --config.model.architecture="small" \
    --config.model.directory="/home/zifanw/ensembleKW/models/seq_trained/l_2/cifar_small_36px.pth" \
    --config.data.dataset='cifar' \
    --config.attack.eps=0.141 \
    --config.attack.norm="l2" \
    --config.attack.step_size=0.0003 \
    --config.data.batch_size=128

python examples/attack_ensemble.py \
    --config=examples/configs/attack.py \
    --config.model.architecture="large" \
    --config.model.directory="/home/zifanw/ensembleKW/models/seq_trained/l_2/cifar_large_36px.pth" \
    --config.data.dataset='cifar' \
    --config.attack.eps=0.141 \
    --config.attack.norm="l2" \
    --config.attack.step_size=0.0003 \
    --config.data.batch_size=8


python examples/attack_ensemble.py \
    --config=examples/configs/attack.py \
    --config.model.architecture="resnet" \
    --config.model.directory="/home/zifanw/ensembleKW/models/seq_trained/l_2/cifar_resnet_36px.pth" \
    --config.data.dataset='cifar' \
    --config.attack.eps=0.141 \
    --config.attack.norm="l2" \
    --config.attack.step_size=0.0003 \
    --config.data.batch_size=4

python examples/attack_ensemble.py \
    --config=examples/configs/attack.py \
    --config.model.architecture="resnet" \
    --config.model.directory="/home/zifanw/ensembleKW/models/seq_trained/l_2/cifar_resnet_144px.pth" \
    --config.data.dataset='cifar' \
    --config.attack.eps=0.565 \
    --config.attack.norm="l2" \
    --config.attack.step_size=0.01 \
    --config.data.batch_size=4

