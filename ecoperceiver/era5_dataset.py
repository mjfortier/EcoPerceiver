import os
import sqlite3
import torch as tr
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from tqdm import tqdm
from ecoperceiver.dataset import EcoPerceiverLoaderConfig


@dataclass
class ERA5Batch:
    sites: Tuple[str]
    igbp: Tuple[str]
    timestamps: Tuple
    predictor_columns: Tuple[str]
    predictor_values: tr.Tensor
    aux_columns: Tuple[str]
    aux_values: tr.Tensor
    modis_values: Optional[tr.Tensor]
    modis_present: Optional[tr.Tensor]
    phenocam_ir: Tuple
    phenocam_rgb: Tuple
    target_columns: Optional[Tuple[str]] = None
    target_values: Optional[tr.Tensor] = None

    def pin_memory(self):
        self.predictor_values = self.predictor_values.pin_memory()
        self.aux_values = self.aux_values.pin_memory()
        if self.modis_values is not None:
            self.modis_values = self.modis_values.pin_memory()
        if self.modis_present is not None:
            self.modis_present = self.modis_present.pin_memory()
        if self.target_values is not None:
            self.target_values = self.target_values.pin_memory()
        return self


class ERA5Dataset(Dataset):
    def __init__(
        self,
        data_dir: Union[str, os.PathLike],
        config: EcoPerceiverLoaderConfig,
        sql_file: Union[str, os.PathLike, None] = None,
    ):
        self.data_path = Path(data_dir)
        self.config = config
        self.columns = ('id', 'coord_id', 'timestamp') + tuple(self.config.predictors)
        self.window_len = self.config.context_length
        self.sql_file = (
            Path(sql_file).expanduser().resolve()
            if sql_file is not None
            else (self.data_path / 'era5.db').resolve()
        )
        if not self.sql_file.exists():
            raise FileNotFoundError(f'ERA5 sqlite file not found: {self.sql_file}')

        print('Indexing coordinates...')
        with sqlite3.connect(self.sql_file) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table';"
                ).fetchall()
            }
            self.has_modis_table = 'modis_data' in tables
            self.has_phenocam_table = 'phenocam_data' in tables

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

    @staticmethod
    def _timestamp_to_modis_date(timestamp) -> int:
        return int(str(timestamp)[:8])

    @staticmethod
    def _modis_from_bytes(bytestring):
        return tr.tensor(np.frombuffer(bytestring, dtype=np.float32).reshape(9, 8, 8))

    def _load_modis_sample(self, conn: sqlite3.Connection, coord_id: int, timestamps) -> tuple[tr.Tensor, ...]:
        if not self.config.use_modis or not self.has_modis_table or len(timestamps) == 0:
            return tuple()

        if self.config.single_image:
            target_date = self._timestamp_to_modis_date(timestamps[-1])
            result = conn.execute(
                """
                SELECT data
                FROM modis_data
                WHERE coord_id = ? AND modis_date <= ?
                ORDER BY modis_date DESC
                LIMIT 1;
                """,
                (int(coord_id), target_date),
            ).fetchone()
            if result is None:
                return tuple()
            return (self._modis_from_bytes(result[0]),)

        modis_tensors = []
        last_loaded_date = None
        unique_dates = sorted({self._timestamp_to_modis_date(ts) for ts in timestamps})
        for target_date in unique_dates:
            result = conn.execute(
                """
                SELECT modis_date, data
                FROM modis_data
                WHERE coord_id = ? AND modis_date <= ?
                ORDER BY modis_date DESC
                LIMIT 1;
                """,
                (int(coord_id), target_date),
            ).fetchone()
            if result is None:
                continue

            loaded_date, bytestring = result
            if loaded_date == last_loaded_date:
                continue
            modis_tensors.append(self._modis_from_bytes(bytestring))
            last_loaded_date = loaded_date

        return tuple(modis_tensors)

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
            modis_data = self._load_modis_sample(conn, coord_id, df['timestamp'].tolist())

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
               self.config.targets, modis_data

    @staticmethod
    def _pack_modis_batch(modis_batch):
        template = next((sample[0] for sample in modis_batch if len(sample) > 0), None)
        if template is None:
            return None, None

        modis_values = template.new_zeros((len(modis_batch), *template.shape))
        modis_present = tr.zeros(len(modis_batch), dtype=tr.bool)
        for i, sample in enumerate(modis_batch):
            if len(sample) == 0:
                continue
            modis_values[i].copy_(sample[0])
            modis_present[i] = True

        return modis_values, modis_present

    def collate_fn(self, batch):
        igbp, ts, preds, pred_data, aux, aux_data, targs, modis = zip(*batch)

        preds, targs, aux = preds[0], targs[0], aux[0]
        predictor_values = tr.stack(pred_data, dim=0)
        aux_values = tr.stack(aux_data, dim=0)
        batch_size = predictor_values.shape[0]
        empty_modalities = tuple(() for _ in range(batch_size))
        modis_values, modis_present = self._pack_modis_batch(modis)

        return ERA5Batch(
            sites=tuple('' for _ in range(batch_size)),            
            igbp=igbp,
            timestamps=ts,
            predictor_columns=preds,
            predictor_values=predictor_values,
            aux_columns=aux,
            aux_values=aux_values,
            modis_values=modis_values,
            modis_present=modis_present,
            phenocam_ir=empty_modalities,
            phenocam_rgb=empty_modalities,
            target_columns=targs, 
            target_values=None,
        )
