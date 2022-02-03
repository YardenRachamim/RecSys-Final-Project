from data.datasets import get_dataset
from data.dataloaders import GRU4Rec15Loader
from configuration import config
from torch_modules.models import GRU4REC
from torch_modules.loss_functions import LossFunction
from torch_training.optim import Optimizer
from torch_training.train import Trainer

train_path = config['data_path'] / config['train_file_name']
test_path = config['data_path'] / config['test_file_name']

train_dataset = get_dataset(train_path, 'Rec15')
test_dataset = get_dataset(test_path, 'Rec15')

input_size = train_dataset.n_items
config['GRU4REC_arguments']['input_size'] = input_size
config['GRU4REC_arguments']['output_size'] = input_size
model = GRU4REC(**config['GRU4REC_arguments'])

loss_function = LossFunction(**config['LossFunction_arguments'])  # cuda is used with cross entropy only
optimizer = Optimizer(model.parameters(), **config['Optimizer_arguments'])
trainer = Trainer(model,
                  train_dataset=train_dataset,
                  optim=optimizer, use_cuda=config['shared_arguments']['cuda'],
                  loss_func=loss_function,
                  batch_size=config['shared_arguments']['batch_size'])

n_epochs = config['arguments']['n_epochs']

trainer.train(0, n_epochs - 1)