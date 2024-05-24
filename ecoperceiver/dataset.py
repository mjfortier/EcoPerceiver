import os
import torch
import pandas as pd
import numpy as np
import json
import pickle as pkl
from torch.utils.data import Dataset

class FluxDataset(Dataset):
    def __init__(
            self, data_dir, sites, context_length=48,
            target='NEE_VUT_REF'
            ):
        self.data_dir = data_dir
        self.sites = sites
        self.data = []
        self.context_length = context_length

        self.target = target
        
        for root, _, files in os.walk(self.data_dir):
            in_sites = False
            for site in sites:
                if site in root:
                    in_sites = True
            if not in_sites:
                continue

            if 'data.csv' in files:
                df = pd.read_csv(os.path.join(root, 'data.csv'))

                float_cols = [c for c in df.columns if c != 'timestamp']
                df[float_cols] = df[float_cols].astype(np.float32)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                with open(os.path.join(root, 'modis.pkl'), 'rb') as f:
                    modis_data = pkl.load(f)
                with open(os.path.join(root, 'meta.json'), 'r') as f:
                    meta = json.load(f)

                self.data.append((meta, df, modis_data))

        self.create_lookup_table()
        self.lookup_table = []
        for i, d in enumerate(self.data):
            _, df, _ = d
            for r in range(self.context_length, len(df)+1):
                if 
                self.lookup_table.append((i,r))
    
    def create_lookup_table(self):
        self.lookup_table = []
        for i, data_tuple in enumerate(self.data):
            _, df, _ = data_tuple
            for r in range(self.context_length, len(df)+1):
                self.lookup_table.append((i,r))


    def num_channels(self):
        # returns number of frequency bands in the imagery
        _, _, modis = self.data[0]
        return modis[list(modis.keys())[0]].shape[0]
    
    def columns(self):
        _, labels, _, _, _ = self.__getitem__(0)
        return labels
    
    def mask_targets(self, prev_targets):
        # Add targets to predictors, but only a random number of them to simulate cold starts
        prev_mask = torch.zeros(prev_targets.shape).to(torch.bool)
        n = np.random.randint(0, len(prev_targets))
        prev_mask[-1:] = True
        prev_mask[:n] = True
        return prev_mask | prev_targets.isnan()

    def __len__(self):
        return len(self.lookup_table)

    def __getitem__(self, idx):
        site_num, row_max = self.lookup_table[idx]
        row_min = row_max - (self.context_length)

        _, df, modis = self.data[site_num]
        rows = df.iloc[row_min:row_max]

        rows = rows.reset_index(drop=True)
        modis_data = []
        timestamps = list(rows['timestamp'])
        for i, ts in enumerate(timestamps):
            pixels = modis.get(ts, None)
            if pixels is not None:
                modis_data.append((i, torch.tensor(pixels[:,1:9,1:9], dtype=torch.float32)))
        
        predictor_df = rows.drop(columns=self.remove_columns)
        labels = list(predictor_df.columns)
        target_df = rows[self.target_columns]
        
        predictors = torch.tensor(predictor_df.values)
        mask = predictors.isnan()
        predictors = predictors.nan_to_num(-1.0) # just needs a numeric value, doesn't matter what

        targets = torch.tensor(target_df.values[-1:])
        return predictors, labels, mask, modis_data, targets


def custom_collate_fn(batch):
    predictors, labels, mask, modis_data, targets = zip(*batch)
    # Normal attributes
    predictors = torch.stack(predictors, dim=0)
    mask = torch.stack(mask, dim=0)
    targets = torch.stack(targets, dim=0)

    for l in labels[1:]:
        np.testing.assert_array_equal(labels[0], l, f'Difference found in input arrays {labels[0]} and {l}')
    labels = labels[0]

    # List of modis data. Tuples of (batch, timestep, data)
    modis_list = []
    for b, batch in enumerate(modis_data):
        for t, data in batch:
            modis_list.append((b, t, data))
    modis_data = modis_list

    return predictors, labels, mask, modis_data, targets
