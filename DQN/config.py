import types

trainer_dict_1 = {
        'seed': 42,
        'epochs': 50000,
        'minimal_replay_size': 5000, # 最小训练样本数量
        'start_e': 1,
        'end_e': 0.05,
        'exploration_fraction':  0.5, # the fraction of exploration rate it takes from start-e to go end-e
        'exploration_epoch': 50000/2,  # 用于探索率下降的epoch
        'update_iter': 10,  # 更新频率
        'batch_size': 64,
        'eval_freq': 100,  # 评估频率
        'replay_buffer_config':
            {
                'buffer_size': 10000,
            }
    }

env_config_1 = {
    'env_name': 'CartPole-v1',
    'max_episode_steps': 1000,

}

agent_config_1 = {
    'name': 'DQN',
    'hidden_dim': 128,
    'gamma': 0.99,
    'epsilon': 0.1,
    'target_update': 100,
    'lr': 1e-4,
    'batch_size': 64,
    'update_count': 100,
}

replay_buffer_config_1 = {

}



# trainer_config_dict = {
#         'seed': 42,
#         'epochs': 1000,
#         'start_e': 1,
#         'end_e': 0.05,
#         'exploration_fraction':  0.5, # the fraction of exploration rate it takes from start-e to go end-e
#
#         'update_iter': 100,
#         'batch_size': 64,
#         'eval_freq': 100,
# }
# trainer_config = types.SimpleNamespace(**trainer_config_dict)

evaluator_config_dict =  {
    'global': {
        'seed': 42,
        'log_dir': 'results'
    },
    'environments': [
        {'name': 'CartPole-v1', 'max_episode_steps': 500},
        # {'name': 'LunarLander-v2', 'max_steps': 1000}
    ],
    'agents': [
        {
            'name': 'DQN',
            'params': [
                {
                    'hidden_dim': 128,
                    'gamma': 0.99,
                    'epsilon': 0.1,
                    'target_update': 100,
                    'lr': 1e-4,
                    'batch_size': 64,
                    'update_count': 100,
                },
                {'hidden_dim': 128, 'lr': 1e-4}
            ]
        },
        # {
        #     'name': 'PPO',
        #     'params': [
        #         {'learning_rate': 3e-4},
        #         {'learning_rate': 1e-4}
        #     ]
        # }
    ]
}
evaluator_config = types.SimpleNamespace(**evaluator_config_dict)


default_DQN_config_dict = {
        'hidden_dim': 128,
        'gamma': 0.99,
        'epsilon': 0.1,
        'target_update': 100,
        'lr': 1e-4,
        'batch_size': 64,
        'update_count': 100,
        'metrics': [],
    }

default_DQN_config = types.SimpleNamespace(**default_DQN_config_dict)