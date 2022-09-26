# python examples/attack_ensemble.py \
#     --config=examples/configs/attack.py \
#     --config.model.architecture="small" \
#     --config.model.directory="/home/zifanw/ensembleKW/models/seq_trained/l_inf/cifar_small_2px.pth" \
#     --config.data.dataset='cifar' \
#     --config.attack.eps=0.0078 \
#     --config.attack.step_size=0.0003 \
#     --config.data.batch_size=128 \
#     --config.data.normalization="meanstd" \
#     --config.attack.do_surrogate=true

python examples/attack_ensemble.py \
    --config=examples/configs/attack.py \
    --config.model.architecture="small" \
    --config.model.directory="/home/zifanw/ensembleKW/models/seq_trained/l_inf/cifar_small_8px.pth" \
    --config.data.dataset='cifar' \
    --config.attack.eps=0.031 \
    --config.attack.step_size=0.00124 \
    --config.data.batch_size=128 \
    --config.data.normalization="meanstd" \
    --config.attack.do_surrogate=true

python examples/attack_ensemble.py \
    --config=examples/configs/attack.py \
    --config.model.architecture="small" \
    --config.model.directory="/home/zifanw/ensembleKW/models/seq_trained/l_2/cifar_small_36px.pth" \
    --config.data.dataset='cifar' \
    --config.attack.eps=0.141 \
    --config.attack.norm="l2" \
    --config.attack.step_size=0.0003 \
    --config.data.batch_size=128 \
    --config.data.normalization="meanstd" \
    --config.attack.do_surrogate=true

python examples/attack_ensemble.py \
    --config=examples/configs/attack.py \
    --config.model.architecture="exact" \
    --config.model.directory="/home/zifanw/ensembleKW/models/seq_trained/l_2/mnist_small_exact.pth" \
    --config.attack.norm="l2" \
    --config.attack.eps=1.58 \
    --config.attack.step_size=0.03 \
    --config.attack.do_surrogate=true

python examples/attack_ensemble.py \
    --config=examples/configs/attack.py \
    --config.model.architecture="small" \
    --config.model.directory="/home/zifanw/ensembleKW/models/seq_trained/l_2/mnist_small.pth" \
    --config.attack.norm="l2" \
    --config.attack.eps=1.58 \
    --config.attack.step_size=0.03 \
    --config.attack.do_surrogate=true

python examples/attack_ensemble.py \
    --config=examples/configs/attack.py \
    --config.model.architecture="exact" \
    --config.model.directory="/home/zifanw/ensembleKW/models/seq_trained/l_inf/mnist_small_exact_0_1.pth" \
    --config.attack.do_surrogate=true

python examples/attack_ensemble.py \
    --config=examples/configs/attack.py \
    --config.model.architecture="small" \
    --config.model.directory="/home/zifanw/ensembleKW/models/seq_trained/l_inf/mnist_small_0_1.pth" \
    --config.attack.do_surrogate=true

python examples/attack_ensemble.py \
    --config=examples/configs/attack.py \
    --config.model.architecture="small" \
    --config.model.directory="/home/zifanw/ensembleKW/models/seq_trained/l_inf/mnist_small_0_3.pth" \
    --config.attack.eps=0.3 \
    --config.attack.step_size=0.012 \
    --config.attack.do_surrogate=true
