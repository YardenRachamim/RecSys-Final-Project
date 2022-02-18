import os
from pathlib import Path
import pickle
import lib

def read_data(data_path: Path, train_file_name: str, val_file_name: str):
    train_cache_file_name = train_file_name.split('.')[0] + ".pkl"
    val_cache_file_name = val_file_name.split('.')[0] + ".pkl" 
    train_cache_path = data_path / train_cache_file_name
    is_train_cache_exists = train_cache_path.is_file()
    val_cache_path = data_path / val_cache_file_name
    is_val_cache_exists = val_cache_path.is_file() 

    if is_train_cache_exists:
        with open(train_cache_path, 'rb') as fis:
            train_data = pickle.load(fis)
    else:
        train_data = lib.Dataset(data_path / train_file_name)
        with open(train_cache_path, 'wb') as fos:
            pickle.dump(train_data, fos)
    
    if is_val_cache_exists:
        with open(val_cache_path, 'rb') as fis:
            valid_data = pickle.load(fis)
    else:
        valid_data = lib.Dataset(data_path / val_file_name, itemmap=train_data.itemmap)
        with open(val_cache_path, 'wb') as fos:
            pickle.dump(valid_data, fos)
            
    return train_data, valid_data
    
def read_paramters_for_eval(load_model: str):
    ckeckpoint_dir = Path(load_model).parent
    
    with open(ckeckpoint_dir / 'parameter.txt', 'r') as fis:
      for line in fis:
        if line.startswith("BATCH_SIZE"):
            batch_size = int(line.split('=')[1].strip())
            print(line)
        elif line.startswith("LOSS_TYPE"):
            loss_type = line.split('=')[1].strip()
            print(line)
    
    return batch_size, loss_type
            