from pathlib import Path

config = dict()

config['data_path'] = Path(r"C:\Users\Yarden\Computer Science\Masters\2\A\Recomendation System\Final Project\data")
config['raw_data_path'] = config['data_path'] / 'raw_data'
config['models_path'] = Path(r"C:\Users\Yarden\Computer Science\Masters\2\A\Recomendation System\Final Project\models")

config['train_file_name'] = 'recSys15TrainOnly.txt'
config['test_file_name'] = 'recSys15Valid.txt'


config['shared_arguments'] = {
    'batch_size': 50,
    'cuda': True
}

config['GRU4REC_arguments'] = {
    'hidden_size': 100,
    'num_layers': 3,
    'dropout_input': 0.,
    'dropout_hidden': 0.5,
    'embedding_dim': -1,
    'final_act': 'tanh',
    'batch_size': config['shared_arguments']['batch_size'],
    'use_cuda': config['shared_arguments']['cuda'],
    'sigma': None
}

config['LossFunction_arguments'] = {
    'use_cuda': config['shared_arguments']['cuda'],
    'loss_type': 'TOP1'
}

config['Optimizer_arguments'] = {
    'optimizer_type': 'Adagrad',
    'lr': 0.01,
    'weight_decay': 0,
    'momentum': 0,
    'eps': 1e-6
}

config['evaluation'] = {
    'load_model': "model_00000.pt",
    'k_eval': 20
}

config['arguments'] = {
    'n_epochs': 5,
    'time_sort': False
}

config['training'] = {
    'checkpoint_dir': config['models_path']
}
