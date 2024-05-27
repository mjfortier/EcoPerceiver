import os
import torch
import pandas as pd
import numpy as np
import json
import pickle as pkl
from torch.utils.data import Dataset

class EcoPerceiverDataset(Dataset):
    '''
    Note:
      If you specify multiple targets, the dataloader will only provide rows
      where all targets are non-null.
    '''
    def __init__(
            self, data_dir, sites, context_length=48, targets=['NEE_VUT_REF']
            ):
        self.data_dir = data_dir
        self.sites = sites
        self.data = []
        self.context_length = context_length
        self.targets = targets
        
        for root, _, files in os.walk(self.data_dir):
            in_sites = False
            for site in sites:
                if site in root:
                    in_sites = True
            if not in_sites:
                continue

            if 'meta.json' in files:
                with open(os.path.join(root, 'meta.json'), 'r') as f:
                    meta = json.load(f)
                
                target_df = pd.read_csv(os.path.join(root, 'targets.csv'), usecols=['timestamp'] + self.targets)
                pred_df = pd.read_csv(os.path.join(root, 'predictors.csv'))
                for df in [target_df, pred_df]:
                    float_cols = [c for c in df.columns if c != 'timestamp']
                    df[float_cols] = df[float_cols].astype(np.float32)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                with open(os.path.join(root, 'modis.pkl'), 'rb') as f:
                    modis_data = pkl.load(f)

                self.data.append((meta, pred_df, modis_data, target_df))

        # Create lookup table
        self.lookup_table = []
        for i, d in enumerate(self.data):
            _, _, _, target_df = d
            condition = (target_df.index > self.context_length - 1)
            for t in self.targets:
                condition &= (target_df[t].notnull())
            self.lookup_table.extend([(i, r) for r in target_df[condition].index.tolist()])


    def num_channels(self):
        # returns number of frequency bands in the imagery
        _, _, modis, _ = self.data[0]
        return modis[list(modis.keys())[0]].shape[0]
    
    def columns(self):
        _, labels, _, _, _, _, _, _ = self.__getitem__(0)
        return labels

    def __len__(self):
        return len(self.lookup_table)
    
    def get_target_dataframe(self):
        target_dfs = []
        for meta, _, _, target_df in self.data:
            target_copy = target_df.copy()
            target_copy['SITE_ID'] = meta['SITE_ID']
            target_copy['Inferred'] = np.nan
            target_dfs.append(target_copy[['SITE_ID', 'timestamp'] + self.targets + ['Inferred']])
        return pd.concat(target_dfs, axis=0)

    def __getitem__(self, idx):
        site_num, row_max = self.lookup_table[idx]
        row_min = row_max - (self.context_length)

        meta, pred_df, modis_data, target_df = self.data[site_num]
        pred_rows = pred_df.iloc[row_min+1:row_max+1].reset_index(drop=True)
        target_rows = target_df.iloc[row_min+1:row_max+1].reset_index(drop=True)

        modis_imgs = []
        timestamps = list(pred_rows['timestamp'])

        for i, ts in enumerate(timestamps):
            pixels = modis_data.get(ts, None)
            if pixels is not None:
                modis_imgs.append((i, torch.tensor(pixels[:,1:9,1:9], dtype=torch.float32)))
        
        pred_rows = pred_rows.drop(columns=['timestamp'])
        target_rows = target_rows.drop(columns=['timestamp'])
        predictor_labels = list(pred_rows.columns)
        target_labels = list(target_rows.columns)
        
        predictors = torch.tensor(pred_rows.values)
        predictor_mask = predictors.isnan()
        predictors = predictors.nan_to_num(-1.0) # just needs a numeric value, doesn't matter what

        targets = torch.tensor(target_rows.values[-1:])
        return (
            predictors,
            predictor_labels,
            predictor_mask,
            modis_imgs,
            targets,
            target_labels,
            meta['SITE_ID'],
            timestamps[-1]
        )


def ep_collate(batch):
    predictors, predictor_labels, predictor_mask, modis_imgs, targets, target_labels, site_ids, timestamps = zip(*batch)
    # Normal attributes
    predictors = torch.stack(predictors, dim=0)
    predictor_mask = torch.stack(predictor_mask, dim=0)
    targets = torch.stack(targets, dim=0)

    # Make sure all labels match up
    for l in predictor_labels[1:]:
        np.testing.assert_array_equal(predictor_labels[0], l, f'Difference found in input arrays {predictor_labels[0]} and {l}')
    predictor_labels = predictor_labels[0]

    for l in target_labels[1:]:
        np.testing.assert_array_equal(target_labels[0], l, f'Difference found in input arrays {target_labels[0]} and {l}')
    target_labels = target_labels[0]

    # List of modis data. Tuples of (batch, timestep, data)
    modis_list = []
    for b, batch in enumerate(modis_imgs):
        for t, data in batch:
            modis_list.append((b, t, data))
    modis_imgs = modis_list

    return {
        'predictors': predictors,
        'predictor_labels': predictor_labels,
        'predictor_mask': predictor_mask,
        'modis_imgs': modis_imgs,
        'targets': targets,
        'target_labels': target_labels,
        'site_ids': site_ids,
        'timestamps': timestamps,
    }
