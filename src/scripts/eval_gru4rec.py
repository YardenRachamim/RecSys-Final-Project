import torch
from data.datasets import get_dataset
from data.dataloaders import GRU4Rec15Loader
from configuration import config
from torch_modules.models import GRU4REC
from torch_modules.loss_functions import LossFunction
from torch_training.optim import Optimizer
from torch_training.eval import Evaluation
from torch_utils.model_utils import load_trained_model

test_path = config['data_path'] / config['test_file_name']
test_dataset = get_dataset(test_path, 'Rec15')

model = load_trained_model(config['evaluation']['load_model'])
model.gru.flatten_parameters()

loss_function = LossFunction(**config['LossFunction_arguments'])  # cuda is used with cross entropy only
k = config['evaluation']['k_eval']
use_cuda = False
batch_size = config['shared_arguments']['batch_size']

evaluator = Evaluation(model, loss_function, use_cuda, k=k)
loss, recall, mrr = evaluator.eval(test_dataset, batch_size)