import os
import sqlite3
import torch as tr
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
from typing import Union
from tqdm import tqdm
from ecoperceiver.dataset import EcoPerceiverLoaderConfig, EcoPerceiverBatch


class ERA5Dataset(Dataset):
    def __init__(self, data_dir: Union[str, os.PathLike], config: EcoPerceiverLoaderConfig):
        self.data_path = Path(data_dir)
        self.config = config
        self.columns = ('id', 'coord_id', 'timestamp') + tuple(self.config.predictors)
        self.window_len = self.config.context_length
        self.sql_file = self.data_path / 'era5.db'

        print('Indexing coordinates...')
        with sqlite3.connect(self.sql_file) as conn:
            result = conn.execute("""
                SELECT id, coord_id 
                FROM ec_data 
                ORDER BY coord_id, id;
            """).fetchall()
            
            df = pd.DataFrame(result, columns=['id', 'coord_id'])
            
            indexes = []
            for _, group in df.groupby('coord_id'):
                ids = group['id'].values[self.config.context_length-1:]
                indexes.extend(ids)
            
            self.data = np.array(indexes, dtype=np.int32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        top_index = self.data[idx]
        bottom_index = top_index - self.config.context_length + 1

        with sqlite3.connect(self.sql_file) as conn:
            ec_data = conn.execute(f"""
                SELECT {",".join(self.columns)} 
                FROM ec_data 
                WHERE id >= {bottom_index} AND id <= {top_index} 
                ORDER BY id;
            """).fetchall()
        
            df = pd.DataFrame(data=ec_data, columns=self.columns)

            coord_id = df['coord_id'].unique()[0]
            aux_result = conn.execute(f'SELECT {",".join(self.config.aux_data)} FROM coord_data WHERE coord_id == "{coord_id}";').fetchall()
            aux_data = {self.config.aux_data[i]: aux_result[0][i] for i in range(len(self.config.aux_data))}
            igbp = aux_data['igbp']

        ec_timestamps = df['timestamp'].tolist()
        ec_data = tr.tensor(df[list(self.config.predictors)].fillna(value=np.nan).astype(np.float32).values)

        aux_data = tr.tensor([
            aux_data['lat'] / 180.0 if aux_data['lat'] is not None else np.nan,
            aux_data['lon'] / 180.0 if aux_data['lon'] is not None else np.nan,
            aux_data['elev'] / 8000.0 if aux_data['elev'] is not None else np.nan]
        ).to(tr.float32) 

        return igbp, ec_timestamps, \
               self.config.predictors, ec_data, \
               ('lat', 'lon', 'elev'), aux_data, \
               self.config.targets

    def collate_fn(self, batch):
        igbp, ts, preds, pred_data, aux, aux_data, targs = zip(*batch)

        preds, targs, aux = preds[0], targs[0], aux[0]
        predictor_values = tr.stack(pred_data, dim=0)
        aux_values = tr.stack(aux_data, dim=0)
        batch_size = predictor_values.shape[0]
        empty_modalities = tuple(() for _ in range(batch_size))

        return EcoPerceiverBatch(
            sites=tuple('' for _ in range(batch_size)),            
            igbp=igbp,
            timestamps=ts,
            predictor_columns=preds,
            predictor_values=predictor_values,
            aux_columns=aux,
            aux_values=aux_values,
            target_columns=targs, 
            target_values=None,
            modis=empty_modalities,
            phenocam_ir=empty_modalities,
            phenocam_rgb=empty_modalities
        )