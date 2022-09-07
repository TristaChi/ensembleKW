import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.verbose = 1
    config.seed = 0

    config.data = ml_collections.ConfigDict()
    config.data.dataset = 'mnist'
    config.data.n_examples = 10000
    config.data.normalization = '01'
    config.data.batch_size = 8

    config.io = ml_collections.ConfigDict()
    config.io.output_file = 'attack_op'

    config.model = ml_collections.ConfigDict()
    config.model.directory = 'models/seq_trained/l_inf/mnist_small_0_1.pth'
    config.model.architecture = 'small'

    config.attack = ml_collections.ConfigDict()
    config.attack.norm = 'linf'
    config.attack.opt = 'sgd'
    config.attack.eps = 0.1
    config.attack.steps = 100
    config.attack.step_size = 0.01
    config.attack.do_surrogate = False
    config.attack.bounded_input = False

    return config