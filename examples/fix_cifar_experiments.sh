
python examples/attack_ensemble.py \
    --config=examples/configs/attack.py \
    --config.model.architecture="small" \
    --config.model.directory="/home/zifanw/ensembleKW/models/seq_trained/l_inf/cifar_small_2px.pth" \
    --config.data.dataset='cifar' \
    --config.attack.eps=0.0078 \
    --config.attack.step_size=0.0003 \
    --config.data.batch_size=128 \
    --config.data.normalization="meanstd"

python examples/attack_ensemble.py \
    --config=examples/configs/attack.py \
    --config.model.architecture="large" \
    --config.model.directory="/home/zifanw/ensembleKW/models/seq_trained/l_inf/cifar_large_2px.pth" \
    --config.data.dataset='cifar' \
    --config.attack.eps=0.0078 \
    --config.attack.step_size=0.0003 \
    --config.data.batch_size=8 \
    --config.data.normalization="meanstd"

python examples/attack_ensemble.py \
    --config=examples/configs/attack.py \
    --config.model.architecture="small" \
    --config.model.directory="/home/zifanw/ensembleKW/models/seq_trained/l_inf/cifar_small_8px.pth" \
    --config.data.dataset='cifar' \
    --config.attack.eps=0.031 \
    --config.attack.step_size=0.00124 \
    --config.data.batch_size=128 \
    --config.data.normalization="meanstd"

python examples/attack_ensemble.py \
    --config=examples/configs/attack.py \
    --config.model.architecture="large" \
    --config.model.directory="/home/zifanw/ensembleKW/models/seq_trained/l_inf/cifar_large_8px.pth" \
    --config.data.dataset='cifar' \
    --config.attack.eps=0.031 \
    --config.attack.step_size=0.00124 \
    --config.data.batch_size=8 \
    --config.data.normalization="meanstd"

python examples/attack_ensemble.py \
    --config=examples/configs/attack.py \
    --config.model.architecture="small" \
    --config.model.directory="/home/zifanw/ensembleKW/models/seq_trained/l_2/cifar_small_36px.pth" \
    --config.data.dataset='cifar' \
    --config.attack.eps=0.141 \
    --config.attack.norm="l2" \
    --config.attack.step_size=0.0003 \
    --config.data.batch_size=128 \
    --config.data.normalization="meanstd"

python examples/attack_ensemble.py \
    --config=examples/configs/attack.py \
    --config.model.architecture="large" \
    --config.model.directory="/home/zifanw/ensembleKW/models/seq_trained/l_2/cifar_large_36px.pth" \
    --config.data.dataset='cifar' \
    --config.attack.eps=0.141 \
    --config.attack.norm="l2" \
    --config.attack.step_size=0.0003 \
    --config.data.batch_size=8 \
    --config.data.normalization="meanstd"