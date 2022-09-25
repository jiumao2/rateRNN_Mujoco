def get_hyperparams():
    config = dict()

    # network

    config['tau'] = 0.01  # in seconds
    config['n_neuron'] = 15
    config['dt'] = 0.001  # in seconds
    config['dt_env'] = 0.01  # in seconds
    config['baseline_input'] = 0.2  # in seconds
    config['initiation_length'] = 10000
    config['min_reward_init'] = 3000
    config['weights_clip'] = 5
    config['warmup_steps'] = 100

    # train
    config['episode_length'] = 1000  # timesteps
    config['num_workers'] = 16
    config['num_episodes_per_epoch'] = 1000
    config['noise_standard_deviation'] = 1
    config['learning_rate'] = 0.001
    config['l2'] = 0

    # classical evolutionary algorithm
    config['num_parents'] = 16
    config['num_children'] = 256

    # mlp policy
    config['num_hidden_units'] = 64
    config['population_size'] = 1250
    config['mutation_power'] = 0.01
    config['truncation_size'] = 64

    return config
