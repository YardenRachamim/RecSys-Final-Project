import pandas as pd
import numpy as np
import pickle
from typing import Union, Dict
from pathlib import Path


class Rec15DataSet(object):
    def __init__(self, data_path: Union[Path, str],
                 sep: str = ',',
                 session_key: str = 'SessionID',
                 item_key: str = 'ItemID',
                 time_key: str = 'Time',
                 n_sample: int = -1,
                 itemmap: pd.DataFrame = None,
                 time_sort=False):
        # Read csv
        self.df = pd.read_csv(data_path, sep=sep, dtype={session_key: int, item_key: int, time_key: float})
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.time_sort = time_sort
        if n_sample > 0:
            self.df = self.df[:n_sample]

        self.n_events = self.df.shape[0]
        self.n_items = self.df[self.item_key].nunique()
        self.n_sessions = self.df[self.session_key].nunique()

        # Add colummn item index to data
        self.itemmap = None
        self.add_item_indices(itemmap=itemmap)

        """
        Sort the df by time, and then by session ID. That is, df is sorted by session ID and
        clicks within a session are next to each other, where the clicks within a session are time-ordered.
        """
        self.df.sort_values([session_key, time_key], inplace=True)
        self.click_offsets = self.get_click_offset()
        self.session_idx_arr = self.order_session_idx()

    def add_item_indices(self, itemmap=None):
        """
        Add item index column named "item_idx" to the df
        Args:
            itemmap (pd.DataFrame): mapping between the item Ids and indices
        """
        if itemmap is None:
            item_ids = self.df[self.item_key].unique()  # type is numpy.ndarray
            item2idx = pd.Series(data=np.arange(len(item_ids)),
                                 index=item_ids)
            # Build itemmap is a DataFrame that have 2 columns (self.item_key, 'item_idx)
            itemmap = pd.DataFrame({self.item_key: item_ids,
                                    'item_idx': item2idx[item_ids].values})
        self.itemmap = itemmap
        self.df = pd.merge(self.df, self.itemmap, on=self.item_key, how='inner')

    def get_click_offset(self):
        """
        self.df[self.session_key] return a set of session_key
        self.df[self.session_key].nunique() return the size of session_key set (int)
        self.df.groupby(self.session_key).size() return the size of each session_id
        self.df.groupby(self.session_key).size().cumsum() retunn cumulative sum
        """
        offsets = np.zeros(self.df[self.session_key].nunique() + 1, dtype=np.int32)
        offsets[1:] = self.df.groupby(self.session_key).size().cumsum()
        return offsets

    def order_session_idx(self):
        if self.time_sort:
            sessions_start_time = self.df.groupby(self.session_key)[self.time_key].min().values
            session_idx_arr = np.argsort(sessions_start_time)
        else:
            session_idx_arr = np.arange(self.df[self.session_key].nunique())
        return session_idx_arr

    @property
    def items(self):
        return self.itemmap[self.item_key].unique()


def get_dataset(data_path: Union[str, Path],
                dataset_name: str,
                use_cache: bool = True,
                **kwargs):
    # In order to pass to the creation object the full signature
    kwargs['data_path'] = data_path.resolve()

    # Make sure we have a Path obj
    if isinstance(data_path, str):
        data_path = Path(data_path)
    # Make sure we have a file
    if not data_path.is_file():
        raise Exception(f"{data_path.resolve()} is not a valid file")

    # Get data directory to check if cache exists,if not create instance and use cache
    data_dir = data_path.parent.resolve()

    file_name = data_path.name.split('.')[0] + ".obj"
    cache_dir = data_dir / 'cache'
    cache_path = cache_dir / file_name
    is_cache_exists = cache_path.is_file()
    cache_dir.mkdir(parents=True, exist_ok=True)

    if is_cache_exists and use_cache:
        # Read from cache
        print(f"reading from {cache_path}")
        with open(cache_path, 'rb') as fis:
            dataset = pickle.load(fis)
    else:
        # Create instance and save as pickle
        dataset = get_dataset_obj(dataset_name, **kwargs)
        print(f"saving {file_name} at {cache_dir}")
        with open(cache_path, 'wb') as fos:
            pickle.dump(dataset, fos)

    return dataset


def get_dataset_obj(dataset_name: str, **kwargs):
    if dataset_name == 'Rec15':
        dataset = Rec15DataSet(**kwargs)
    else:
        raise Exception(f"{dataset_name} is not a valid dataset name")

    return dataset
