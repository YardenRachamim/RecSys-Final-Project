import argparse
import pickle
import lib
from lib.utils import read_data
from pathlib import Path
import os 
from tqdm import tqdm
from collections import Counter
import numpy as np
import torch
import pandas as pd 

from IPython.display import display


parser = argparse.ArgumentParser()

parser.add_argument('--data_folder', default='/content/drive/MyDrive/RecSys/FinalProject/data/youchoose', type=str)
parser.add_argument('--train_data', default='train.txt', type=str)
parser.add_argument('--valid_data', default='valid.txt', type=str)
parser.add_argument('--k_eval', default=20, type=int) #value of K durig Recall and MRR Evaluation

# Get the arguments
args = parser.parse_args()


def main():
    print("Loading train data from {}".format(os.path.join(args.data_folder, args.train_data)))
    print("Loading valid data from {}".format(os.path.join(args.data_folder, args.valid_data)))
    data_path = Path(args.data_folder)
    train_file_name = args.train_data
    val_file_name = args.valid_data
    train_data, valid_data = read_data(data_path, train_file_name, val_file_name)
    
    data  = train_data.df.set_index('item_idx').append(valid_data.df.set_index('item_idx'))
    data['item_count'] = data.groupby('ItemID').transform('count').iloc[:, 0]
    item2count = data['item_count'].sort_values(ascending=False).reset_index().drop_duplicates().set_index('item_idx')
    data = valid_data.df.set_index('item_idx')
    data['item_count'] = data.groupby('ItemID').transform('count').iloc[:, 0]
    n_sessions = data['SessionID'].nunique()
    
    K = args.k_eval
    mrrs = []
    recalls = []
    
    # For each session
    for n, g in tqdm(data.groupby('SessionID'), total=n_sessions):
      session_pop = Counter()
      session = []
    
      for i, target in enumerate(g.index):
        recommendations = []
        session_pop = Counter(session)
        session.append(target)
        
        if i == 0:
          continue
    
        occour = session_pop.most_common()
        items = list(map(lambda t: t[0], occour))
        local_counts = list(map(lambda t: t[1], occour))
        global_counts = list(map(lambda it: item2count.loc[it].iloc[0], items))
    
        df = pd.DataFrame([items, local_counts, global_counts], index=['item', 'local', 'global']).T
        
        df.sort_values(by=['local', 'global'], inplace=True, ascending=False)
        recommendations = list(df['item'])
        n_rec = len(recommendations)
    
        if n_rec < K:
          recommendations += list(item2count.iloc[: K-n_rec].index)
        elif n_rec >= K:
          recommendations = recommendations[:K]
    
        target = torch.tensor(target)
        recommendations = torch.tensor(recommendations).view(1, K)
        recall = lib.metric.get_recall(recommendations, target)
        mrr = lib.metric.get_mrr(recommendations, target)
    
        recalls.append(recall)
        mrrs.append(mrr)
    
    mean_recall = np.mean(recalls)
    mrrs = [t.item() for t in mrrs]
    mean_mrr = np.mean(mrrs)
    
    print(f"recall: {mean_recall}, mrr: {mean_mrr}")
    

if __name__ == '__main__':
    main()