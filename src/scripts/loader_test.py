from deprecated.datasets import get_dataset
from deprecated.dataloaders import GRU4Rec15Loader
from configuration import config
from tqdm import tqdm


train_path = config['data_path'] / config['train_file_name']
dataset = get_dataset(train_path, 'Rec15', use_cache=False)
loader = GRU4Rec15Loader(dataset)

print(len(loader))

i = 0
for x, y, is_new_session in tqdm(loader):
    i += 1