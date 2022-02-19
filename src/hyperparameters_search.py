import argparse
import pickle
import torch
import lib
from lib.utils import read_data, read_paramters_for_eval
import numpy as np
import os
import datetime
from pathlib import Path
import pandas as pd
from IPython.display import display 

parser = argparse.ArgumentParser()
# paths 
parser.add_argument('--checkpoint_dir', type=str, default='/content/drive/MyDrive/RecSys/FinalProject/models/checkpoint_digenetica/one direction')
parser.add_argument('--results_csv_path', type=str, default='/content/drive/MyDrive/RecSys/FinalProject/documents/hyper-parameters search results/results_2')
parser.add_argument('--data_folder', default='/content/drive/MyDrive/RecSys/FinalProject/data/digenetica', type=str)
parser.add_argument('--train_data', default='train.txt', type=str)
parser.add_argument('--valid_data', default='valid.txt', type=str)

#default hyper params
parser.add_argument('--dropout_input', default=0, type=float)
parser.add_argument('--hidden_size', default=100, type=int) #Literature uses 100 / 1000 --> better is 100
parser.add_argument('--k_eval', default=20, type=int) #value of K durig Recall and MRR Evaluation
parser.add_argument("--is_bidirectional", action='store_true') # should be in use for training 
parser.add_argument('--final_act', default='tanh', type=str) #Final Activation Function
parser.add_argument('--weight_decay', default=0, type=float) #no weight decay
parser.add_argument('--eps', default=1e-6, type=float) #not used
parser.add_argument("-seed", type=int, default=22, help="Seed for random initialization") #Random seed setting
parser.add_argument("-sigma", type=float, default=None, help="init weight -1: range [-sigma, sigma], -2: range [0, sigma]") # weight initialization [-sigma sigma] in literature
parser.add_argument("--embedding_dim", type=int, default=-1, help="using embedding") 
parser.add_argument('--time_sort', default=False, type=bool) #In case items are not sorted by time stamp
parser.add_argument('--model_name', default='GRU4REC-CrossEntropy', type=str)
parser.add_argument('--save_dir', default='models', type=str)


# Get the arguments
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
#use random seed defined
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Write Checkpoints with arguments used in a text file for reproducibility
def make_checkpoint_dir():
    print("PARAMETER" + "-"*10)
    now = datetime.datetime.now()
    S = '{:02d}{:02d}{:02d}{:02d}'.format(now.month, now.day, now.hour, now.minute)
    save_dir = os.path.join(args.checkpoint_dir, S)
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    args.checkpoint_dir = save_dir
    with open(os.path.join(args.checkpoint_dir, 'parameter.txt'), 'w') as f:
        for attr, value in sorted(args.__dict__.items()):
            print("{}={}".format(attr.upper(), value))
            f.write("{}={}\n".format(attr.upper(), value))
    print("---------" + "-"*10)

# weight initialization if it was defined
def init_model(model):
    if args.sigma is not None:
        for p in model.parameters():
            if args.sigma != -1 and args.sigma != -2:
                sigma = args.sigma
                p.data.uniform_(-sigma, sigma)
            elif len(list(p.size())) > 1:
                sigma = np.sqrt(6.0 / (p.size(0) + p.size(1)))
                if args.sigma == -1:
                    p.data.uniform_(-sigma, sigma)
                else:
                    p.data.uniform_(0, sigma)


def main():
    print("Loading train data from {}".format(os.path.join(args.data_folder, args.train_data)))
    print("Loading valid data from {}".format(os.path.join(args.data_folder, args.valid_data)))
    data_path = Path(args.data_folder)
    train_file_name = args.train_data
    val_file_name = args.valid_data
    train_data, valid_data = read_data(data_path, train_file_name, val_file_name)
    make_checkpoint_dir()      
    #set all the parameters according to the defined arguments
    input_size = len(train_data.items)
    hidden_size = args.hidden_size
    output_size = input_size
    bidirectional = args.is_bidirectional
    dropout_input = args.dropout_input
    embedding_dim = args.embedding_dim
    final_act = args.final_act
    weight_decay = args.weight_decay
    eps = args.eps
    time_sort = args.time_sort

    loss_functions_ = ["TOP1"] 
    batch_sizes_ = [100, 512]
    num_layers_ = [1, 3] 
    lr_ = [0.01, 0.03] 
    hidden_size_ = [1000, 100] 
    n_epochs_ = [10, 20, 5]
    optimizer_ = ["Adagrad"]
    momentum_ = [0]
    dropout_ = [0.5]
    results_csv_path = f"{args.results_csv_path}_{loss_functions_[0]}"
    results = pd.DataFrame(columns=['loss_fun', 'num_layers', 'hidden_size', 'batch_size','lr','n_epochs', 'optimizer', 'momentum','dropout','Recall@20','MRR@20','Time'])
   
    i = 0
    for loss_fun in loss_functions_:
      for num_layers in num_layers_:
        for hidden_size in hidden_size_:
          for batch_size in batch_sizes_:
            for lr in lr_:
              for n_epochs in n_epochs_:
                for optimizer_type in optimizer_:
                  for momentum in momentum_:
                    for dropout in dropout_:
                        print(f"current parameters: loss_fun: {loss_fun}, num_layers: {num_layers}, hidden_size: {hidden_size}, batch_size: {batch_size}, lr: {lr}, optimizer: {optimizer_type}, momentum: {momentum}, dropout: {dropout}")
                        loss_function = lib.LossFunction(loss_type=loss_fun, use_cuda=args.cuda) #cuda is used with cross entropy only
                        results_dict = {"loss_fun":loss_fun, "num_layers":num_layers,"hidden_size":hidden_size,
                        "batch_size":batch_size, "lr":lr, "n_epochs":n_epochs, "optimizer":optimizer_type,
                        "momentum":momentum, "dropout":dropout}
                        #Initialize the model
                        print(f"bidirectional is {bidirectional}")
                        model = lib.GRU4REC(input_size, hidden_size, output_size, final_act=final_act,
                                          num_layers=num_layers, use_cuda=args.cuda, batch_size=batch_size,
                                          dropout_input=dropout_input, dropout_hidden=dropout, embedding_dim=embedding_dim,
                                          bidirectional=bidirectional)
                        #weights initialization
                        init_model(model)
                        #optimizer
                        optimizer = lib.Optimizer(model.parameters(), optimizer_type=optimizer_type, lr=lr,
                                                weight_decay=weight_decay, momentum=momentum, eps=eps)
                        #trainer class
                        trainer = lib.Trainer(model, train_data=train_data, eval_data=valid_data, optim=optimizer,
                                            use_cuda=args.cuda, loss_func=loss_function, batch_size=batch_size, args=args)
                        print('#### START TRAINING....')
                        recall, mrr, time = trainer.train(0, n_epochs - 1)
                        results_dict["Recall@20"] = recall
                        results_dict["MRR@20"] = mrr
                        results_dict["Time"] = time
                        results = results.append(results_dict,ignore_index=True)
                        results.to_csv(f"{results_csv_path}_{i}.csv", sep=',', index=False)
                        i += 1


    display(results)
    results.to_csv(f"{results_csv_path}_ALL.csv", sep=',', index=False)

if __name__ == '__main__':
    main()
