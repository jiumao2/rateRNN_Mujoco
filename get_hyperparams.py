def get_hyperparams():
    config = dict()

    # network
    config['tau'] = 0.01  # in seconds
    config['n_neuron'] = 10
    config['dt'] = 0.001  # in seconds
    config['dt_env'] = 0.01  # in seconds
    config['baseline_input'] = 0.2  # in seconds
    config['initiation_length'] = 100000
    config['weights_clip'] = 5
    config['warmup_steps'] = 100

    # train
    config['num_workers'] = 16
    config['noise_standard_deviation'] = 1
    config['l1'] = 0.1

    # classical evolutionary algorithm
    config['num_parents'] = 16
    config['num_children'] = 256


    return config
