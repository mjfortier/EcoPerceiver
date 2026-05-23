import os
import sqlite3
import torch as tr
import numpy as np
from bisect import bisect_right
from collections import OrderedDict
from torch.utils.data import Dataset
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Union
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
    _INDEX_FETCH_SIZE = 100_000

    def __init__(
        self,
        data_dir: Union[str, os.PathLike],
        config: EcoPerceiverLoaderConfig,
        sql_file: Union[str, os.PathLike, None] = None,
        start_timestamp: Optional[int] = None,
        end_timestamp: Optional[int] = None,
    ):
        self.data_path = Path(data_dir)
        self.config = config
        self.columns = ('id', 'coord_id', 'timestamp') + tuple(self.config.predictors)
        self.window_len = self.config.context_length
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.sql_file = (
            Path(sql_file).expanduser().resolve()
            if sql_file is not None
            else (self.data_path / 'era5.db').resolve()
        )
        self._conn: Optional[sqlite3.Connection] = None
        self._conn_pid: Optional[int] = None
        self._worker_pid: Optional[int] = None
        self._coord_cache = OrderedDict()
        self._modis_index_cache = OrderedDict()
        self._coord_cache_size = 4096
        self._modis_cache_size = 64
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

            self.data = self._build_sample_index(conn)

    def _build_sample_index(self, conn: sqlite3.Connection) -> np.ndarray:
        filters = []
        params = []
        if self.end_timestamp is not None:
            filters.append("timestamp <= ?")
            params.append(int(self.end_timestamp))
        where_sql = f"WHERE {' AND '.join(filters)}" if filters else ""

        cursor = conn.execute(
            f"""
            SELECT id, coord_id, timestamp
            FROM ec_data
            {where_sql}
            ORDER BY coord_id, id;
            """,
            params,
        )
        indexes = []
        current_coord_id = None
        coord_count = 0

        while True:
            rows = cursor.fetchmany(self._INDEX_FETCH_SIZE)
            if not rows:
                break
            for row_id, coord_id, timestamp in rows:
                if coord_id != current_coord_id:
                    current_coord_id = coord_id
                    coord_count = 1
                else:
                    coord_count += 1
                if self.start_timestamp is not None and timestamp < self.start_timestamp:
                    continue
                if coord_count >= self.config.context_length:
                    indexes.append(row_id)

        return np.asarray(indexes, dtype=np.int32)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_conn'] = None
        state['_conn_pid'] = None
        state['_worker_pid'] = None
        state['_coord_cache'] = OrderedDict()
        state['_modis_index_cache'] = OrderedDict()
        return state

    def __del__(self):
        self._close_connection()

    def _close_connection(self):
        if self._conn is None:
            return
        try:
            self._conn.close()
        except sqlite3.Error:
            pass
        finally:
            self._conn = None
            self._conn_pid = None

    @staticmethod
    def _remember_cached_item(cache: OrderedDict, key, value, max_size: int):
        cache[key] = value
        cache.move_to_end(key)
        if len(cache) > max_size:
            cache.popitem(last=False)

    def _ensure_worker_state(self):
        current_pid = os.getpid()
        if self._worker_pid == current_pid:
            return
        self._close_connection()
        self._coord_cache = OrderedDict()
        self._modis_index_cache = OrderedDict()
        self._worker_pid = current_pid

    def _get_connection(self) -> sqlite3.Connection:
        self._ensure_worker_state()
        if self._conn is None or self._conn_pid != self._worker_pid:
            self._close_connection()
            self._conn = sqlite3.connect(self.sql_file)
            self._conn_pid = self._worker_pid
        return self._conn

    def _get_coord_data(self, conn: sqlite3.Connection, coord_id: int):
        cached = self._coord_cache.get(coord_id)
        if cached is not None:
            self._coord_cache.move_to_end(coord_id)
            return cached

        aux_row = conn.execute(
            f"""
            SELECT {",".join(self.config.aux_data)}
            FROM coord_data
            WHERE coord_id = ?
            LIMIT 1;
            """,
            (int(coord_id),),
        ).fetchone()
        if aux_row is None:
            raise KeyError(f'No coord_data row found for coord_id {coord_id}')

        aux_data = {
            self.config.aux_data[i]: aux_row[i]
            for i in range(len(self.config.aux_data))
        }
        self._remember_cached_item(self._coord_cache, coord_id, aux_data, self._coord_cache_size)
        return aux_data

    def _get_modis_index(self, conn: sqlite3.Connection, coord_id: int):
        if not self.config.use_modis or not self.has_modis_table:
            return None

        cached = self._modis_index_cache.get(coord_id)
        if cached is not None:
            self._modis_index_cache.move_to_end(coord_id)
            return cached

        rows = conn.execute(
            """
            SELECT modis_date, data
            FROM modis_data
            WHERE coord_id = ?
            ORDER BY modis_date;
            """,
            (int(coord_id),),
        ).fetchall()
        modis_index = (
            tuple(int(row[0]) for row in rows),
            tuple(row[1] for row in rows),
            {},
        )
        self._remember_cached_item(
            self._modis_index_cache,
            coord_id,
            modis_index,
            self._modis_cache_size,
        )
        return modis_index

    def _get_modis_tensor(self, modis_index, target_date: int):
        modis_dates, modis_blobs, tensor_cache = modis_index
        position = bisect_right(modis_dates, target_date) - 1
        if position < 0:
            return None, None

        tensor = tensor_cache.get(position)
        if tensor is None:
            tensor = self._modis_from_bytes(modis_blobs[position])
            tensor_cache[position] = tensor
        return modis_dates[position], tensor

    @staticmethod
    def _timestamp_to_modis_date(timestamp) -> int:
        return int(str(timestamp)[:8])

    @staticmethod
    def _modis_from_bytes(bytestring):
        return tr.tensor(np.frombuffer(bytestring, dtype=np.float32).reshape(9, 8, 8))

    def _load_modis_sample(self, conn: sqlite3.Connection, coord_id: int, timestamps) -> tuple[tr.Tensor, ...]:
        if not self.config.use_modis or not self.has_modis_table or len(timestamps) == 0:
            return tuple()

        modis_index = self._get_modis_index(conn, coord_id)
        if modis_index is None or len(modis_index[0]) == 0:
            return tuple()

        if self.config.single_image:
            target_date = self._timestamp_to_modis_date(timestamps[-1])
            _, tensor = self._get_modis_tensor(modis_index, target_date)
            if tensor is None:
                return tuple()
            return (tensor,)

        modis_tensors = []
        last_loaded_date = None
        unique_dates = sorted({self._timestamp_to_modis_date(ts) for ts in timestamps})
        for target_date in unique_dates:
            loaded_date, tensor = self._get_modis_tensor(modis_index, target_date)
            if tensor is None:
                continue
            if loaded_date == last_loaded_date:
                continue
            modis_tensors.append(tensor)
            last_loaded_date = loaded_date

        return tuple(modis_tensors)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        top_index = self.data[idx]
        bottom_index = top_index - self.config.context_length + 1
        conn = self._get_connection()
        ec_rows = conn.execute(
            f"""
            SELECT {",".join(self.columns)}
            FROM ec_data
            WHERE id >= ? AND id <= ?
            ORDER BY id;
            """,
            (int(bottom_index), int(top_index)),
        ).fetchall()
        if len(ec_rows) == 0:
            raise IndexError(f'No ERA5 rows found for index {idx} (id range {bottom_index}-{top_index})')

        coord_id = ec_rows[0][1]
        ec_timestamps = [row[2] for row in ec_rows]
        predictor_array = np.asarray(
            [
                [np.nan if value is None else value for value in row[3:]]
                for row in ec_rows
            ],
            dtype=np.float32,
        )
        ec_data = tr.from_numpy(predictor_array)

        aux_data = self._get_coord_data(conn, coord_id)
        igbp = aux_data['igbp']
        modis_data = self._load_modis_sample(conn, coord_id, ec_timestamps)

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
