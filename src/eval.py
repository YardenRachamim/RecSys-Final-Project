import argparse
import pickle
import torch
import lib
from lib.utils import read_data, read_paramters_for_eval
import numpy as np
import os
import datetime
from pathlib import Path


parser = argparse.ArgumentParser()
#paths
parser.add_argument('--data_folder', default='/content/drive/MyDrive/RecSys/FinalProject/data/youchoose', type=str)
parser.add_argument('--train_data', default='train.txt', type=str)
parser.add_argument('--valid_data', default='valid.txt', type=str)

parser.add_argument('--k_eval', default=20, type=int) #value of K durig Recall and MRR Evaluation
parser.add_argument("-seed", type=int, default=22, help="Seed for random initialization") #Random seed setting
parser.add_argument('--load_model', default=None,  type=str)

# Get the arguments
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
#use random seed defined
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

def main():
    print(f"evaluation")
    # Read parameters file
    if args.load_model is not None:
        batch_size, loss_type = read_paramters_for_eval(args.load_model)
        
        print("Loading train data from {}".format(os.path.join(args.data_folder, args.train_data)))
        print("Loading valid data from {}".format(os.path.join(args.data_folder, args.valid_data)))
        data_path = Path(args.data_folder)
        train_file_name = args.train_data
        val_file_name = args.valid_data
        train_data, valid_data = read_data(data_path, train_file_name, val_file_name)
        
        #loss function
        loss_function = lib.LossFunction(loss_type=loss_type, use_cuda=args.cuda) #cuda is used with cross entropy only
        print("Loading pre-trained model from {}".format(args.load_model))
        checkpoint = torch.load(args.load_model)
        model = checkpoint["model"]
        
        model.gru.flatten_parameters()
        evaluation = lib.Evaluation(model, loss_function, use_cuda=args.cuda, k=args.k_eval)
        loss, recall, mrr = evaluation.eval(valid_data, batch_size)
        print("Final result: recall = {:.2f}, mrr = {:.2f}".format(recall, mrr))
    else:
        print("No Pretrained Model was found!")


if __name__ == '__main__':
    main()
