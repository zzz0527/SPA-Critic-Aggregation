from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.value_lr = 3e-4
    config.T = 5
    config.N = 64
    config.M = 0
    config.actor_dropout_rate = 0.1
    config.actor_num_blocks = 3
    config.decay_steps = int(1e6)
    config.actor_layer_norm = True
    config.value_layer_norm = False
    config.actor_tau = 0.001
    config.beta_schedule = "vp"

    return config
