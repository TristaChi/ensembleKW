python examples/attack_ensemble.py \
    --config=examples/configs/attack.py \
    --config.model.architecture="exact" \
    --config.model.directory="/home/zifanw/ensembleKW/models/seq_trained/l_inf/mnist_small_exact_0_1.pth"

python examples/attack_ensemble.py \
    --config=examples/configs/attack.py \
    --config.model.architecture="small" \
    --config.model.directory="/home/zifanw/ensembleKW/models/seq_trained/l_inf/mnist_small_0_1.pth"

python examples/attack_ensemble.py \
    --config=examples/configs/attack.py \
    --config.model.architecture="small" \
    --config.model.directory="/home/zifanw/ensembleKW/models/seq_trained/l_inf/mnist_small_0_3.pth" \
    --config.attack.eps=0.3 \
    --config.attack.step_size=0.012

python examples/attack_ensemble.py \
    --config=examples/configs/attack.py \
    --config.model.architecture="large" \
    --config.model.directory="/home/zifanw/ensembleKW/models/seq_trained/l_inf/mnist_large_0_1.pth" \
    --config.data.batch_size=8

python examples/attack_ensemble.py \
    --config=examples/configs/attack.py \
    --config.model.architecture="large" \
    --config.model.directory="/home/zifanw/ensembleKW/models/seq_trained/l_inf/mnist_large_0_3.pth" \
    --config.attack.eps=0.3 \
    --config.attack.step_size=0.012 \
    --config.data.batch_size=8

python examples/attack_ensemble.py \
    --config=examples/configs/attack.py \
    --config.model.architecture="small" \
    --config.model.directory="/home/zifanw/ensembleKW/models/seq_trained/l_inf/cifar_small_2px.pth" \
    --config.data.dataset='cifar' \
    --config.attack.eps=0.0078 \
    --config.attack.step_size=0.0003 \
    --config.data.batch_size=128

python examples/attack_ensemble.py \
    --config=examples/configs/attack.py \
    --config.model.architecture="large" \
    --config.model.directory="/home/zifanw/ensembleKW/models/seq_trained/l_inf/cifar_large_2px.pth" \
    --config.data.dataset='cifar' \
    --config.attack.eps=0.0078 \
    --config.attack.step_size=0.0003 \
    --config.data.batch_size=8

python examples/attack_ensemble.py \
    --config=examples/configs/attack.py \
    --config.model.architecture="small" \
    --config.model.directory="/home/zifanw/ensembleKW/models/seq_trained/l_inf/cifar_small_8px.pth" \
    --config.data.dataset='cifar' \
    --config.attack.eps=0.031 \
    --config.attack.step_size=0.00124 \
    --config.data.batch_size=128

python examples/attack_ensemble.py \
    --config=examples/configs/attack.py \
    --config.model.architecture="large" \
    --config.model.directory="/home/zifanw/ensembleKW/models/seq_trained/l_inf/cifar_large_8px.pth" \
    --config.data.dataset='cifar' \
    --config.attack.eps=0.031 \
    --config.attack.step_size=0.00124 \
    --config.data.batch_size=8

python examples/attack_ensemble.py \
    --config=examples/configs/attack.py \
    --config.model.architecture="resnet" \
    --config.model.directory="/home/zifanw/ensembleKW/models/seq_trained/l_inf/cifar_resnet_2px.pth" \
    --config.data.dataset='cifar' \
    --config.attack.eps=0.0078 \
    --config.attack.step_size=0.0003 \
    --config.data.batch_size=4

python examples/attack_ensemble.py \
    --config=examples/configs/attack.py \
    --config.model.architecture="resnet" \
    --config.model.directory="/home/zifanw/ensembleKW/models/seq_trained/l_inf/cifar_resnet_8px.pth" \
    --config.data.dataset='cifar' \
    --config.attack.eps=0.031 \
    --config.attack.step_size=0.00124 \
    --config.data.batch_size=4