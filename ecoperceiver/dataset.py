import os
import numpy as np
import pandas as pd
import torch
import sqlite3
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Union, Any
from dataclasses import dataclass
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

EC_PREDICTORS = ('DOY', 'TOD', 'TA', 'P', 'RH', 'VPD', 'PA', 'CO2', 'SW_IN', 'SW_IN_POT', 'SW_OUT', 'LW_IN', 'LW_OUT',
                 'NETRAD', 'PPFD_IN', 'PPFD_OUT', 'WS', 'WD', 'USTAR', 'SWC_1', 'SWC_2', 'SWC_3', 'SWC_4', 'SWC_5',
                 'TS_1', 'TS_2', 'TS_3', 'TS_4', 'TS_5', 'WTD', 'G', 'H', 'LE',)

EC_TARGETS = ('NEE', 'GPP_DT', 'GPP_NT', 'RECO_DT', 'RECO_NT', 'FCH4')

AUX_DATA = ('lat', 'lon', 'elev', 'igbp')


@dataclass
class EcoPerceiverLoaderConfig:
    '''Configuration for EcoPerceiverV2 dataloader and preprocessor

    targets - variable selection for targets. Must be a subset of EC_TARGETS
    targets_max_qc - maximum QC flag (inclusive) to allow for target values. A lower value will result
                     in fewer usable samples, but they will be of higher quality
    predictors - variable selection for predictors. Must be a subset of EC_PREDICTORS
    predictors_max_qc - similar to targets_max_qc, but applied to predictor variables
    normalization_config - dictionary object used for normalizing variables. Custom dictionaries can
                           be supplied, but should be based on the DEFAULT_NORM template
    '''
    targets: Tuple[str] = EC_TARGETS
    predictors: Tuple[str] = EC_PREDICTORS
    use_modis: bool = True
    use_phenocam: bool = True
    single_image: bool = True
    context_window_length: int = 32
    aux_data: Tuple[str] = AUX_DATA


@dataclass
class EcoPerceiverBatch:
    sites: Tuple[str] # one value for each sample
    igbp: Tuple[str] # igbp classification
    timestamps: Tuple
    predictor_columns: Tuple[str] # common mapping for all samples in the batch
    predictor_values: torch.Tensor # all eddy covariance data: (batch, context_window, values)
    target_columns: Tuple[str]
    target_values: torch.Tensor
    aux_columns: Tuple[str]
    aux_values: torch.Tensor
    modis: Tuple # all modis data: (batch, ndarray)
    phenocam_ir: Tuple # all phenocam infrared data: (batch, ndarray)
    phenocam_rgb: Tuple # all phenocam rgb data: (batch, ndarray)

    def to(self, device: Any):
        '''
        .to(device) is provided with this dataclass as a shortcut to individually moving
        every piece of data in the class.
        '''
        self.ec_values.to(device)
        raise NotImplementedError()


class EcoPerceiverDataset(Dataset):
    def __init__(self,
                 data_dir: Union[str, os.PathLike],
                 config: EcoPerceiverLoaderConfig,
                 sites=None):
        self.data_path = Path(data_dir)
        self.config = config
        self.columns = ('id', 'site', 'timestamp') + tuple(self.config.predictors) + tuple(self.config.targets)

        self.window_len = self.config.context_window_length
        self.sql_file = self.data_path / 'carbonsense_v2.sql'
        self.sites = sites
        if sites == None:
            with sqlite3.connect(self.sql_file) as conn:
                result = conn.execute("SELECT DISTINCT site FROM ec_data;").fetchall()
                self.sites = [s[0] for s in result]
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) # TODO: add normalize (get mean and std first)
        
        indexes = []
        print('Indexing sites...')
        target_boolean = ' OR '.join([f'{t} IS NOT NULL' for t in self.config.targets])
        with sqlite3.connect(self.sql_file) as conn:
            for site in tqdm(self.sites):
                ids = conn.execute(f'SELECT id FROM ec_data WHERE site == "{site}" AND ({target_boolean}) ORDER BY id;').fetchall()
                ids = [i[0] for i in ids]
                ids = ids[self.config.context_window_length-1:]
                indexes.extend(ids)
        self.data = np.array(indexes, dtype=np.int32)
    
    def __len__(self):
        return len(self.data)

    def _load_image(self, filename):
        with Image.open(self.data_path / 'phenocam' / filename) as img_r:
            img = (np.array(img_r.convert('RGB')) / 255).astype(np.float32)
            return self.transform(img)
            #img = img_r.convert('L' if '_IR_' in filename else 'RGB')   # save in case this doesn't work
    
    def __getitem__(self, idx):
        top_index = self.data[idx]
        bottom_index = top_index - self.config.context_window_length + 1
        with sqlite3.connect(self.sql_file) as conn:
            ec_data = conn.execute(f'SELECT {",".join(self.columns)} FROM ec_data WHERE id >= {bottom_index} AND id <= {top_index} ORDER BY id;').fetchall()
            modis_result = conn.execute(f'SELECT row_id, data FROM modis_data WHERE row_id >= {bottom_index} AND row_id <= {top_index};').fetchall()
            phenocam_result = conn.execute(f'SELECT row_id, files FROM phenocam_data WHERE row_id >= {bottom_index} AND row_id <= {top_index};').fetchall()

            df = pd.DataFrame(data=ec_data, columns=self.columns)
            assert len(df['site'].unique()) == 1, f'Pulled rows from multiple sites\nTop index: {top_index}, Bottom index: {bottom_index}'
            site = df['site'].unique()[0]
            aux_result = conn.execute(f'SELECT {",".join(self.config.aux_data)} FROM site_data WHERE site == "{site}";').fetchall()
            aux_data = {self.config.aux_data[i]: aux_result[0][i] for i in range(len(self.config.aux_data))}
            igbp = aux_data['igbp']
        
        ec_timestamps = df['timestamp'].tolist()
        ec_data = torch.tensor(df[list(self.config.predictors)].fillna(value=np.nan).astype(np.float32).values)
        target_fluxes = torch.tensor(df[list(self.config.targets)].fillna(value=np.nan).astype(np.float32).values)

        modis_data = []
        modis_from_bytes = lambda x: torch.tensor(np.frombuffer(x, dtype=np.float32).reshape(9,8,8))
        if self.config.use_modis:
            for _, bytestring in modis_result:
                modis_data.append(modis_from_bytes(bytestring))

        phenocam_ir = []
        phenocam_rgb = []
        if self.config.use_phenocam:
            for _, filetext in phenocam_result:
                files = filetext.split(',')
                phenocam_ir.extend([self._load_image(f) for f in files if '_IR_' in f])
                phenocam_rgb.extend([self._load_image(f) for f in files if '_IR_' not in f])
        
        if self.config.single_image:
            modis_data = modis_data[:1]
            phenocam_ir = phenocam_ir[:1]
            phenocam_rgb = phenocam_rgb[:1]
        
        aux_data = torch.tensor([
            aux_data['lat'] / 180.0 if aux_data['lat'] is not None else np.nan,
            aux_data['lon'] / 180.0 if aux_data['lon'] is not None else np.nan,
            aux_data['elev'] / 8000.0 if aux_data['elev'] is not None else np.nan]
        ).to(torch.float32) # normalize aux data here

        return site, igbp, ec_timestamps, \
               self.config.predictors, ec_data, \
               self.config.targets, target_fluxes, \
               ('lat', 'lon', 'elev'), aux_data, \
               tuple(modis_data), tuple(phenocam_ir), tuple(phenocam_rgb)
    
    def collate_fn(self, batch):
        sites, igbp, ts, preds, pred_data, targs, targ_data, aux, aux_data, modis, phenocam_ir, phenocam_rgb = zip(*batch)
        preds = preds[0] # only need to keep 1 copy of the columns
        targs = targs[0]
        aux = aux[0]
        pred_data = torch.stack(pred_data, dim=0)
        targ_data = torch.stack(targ_data, dim=0)
        aux_data = torch.stack(aux_data, dim=0)
        return EcoPerceiverBatch(sites, igbp, ts, preds, pred_data, targs, targ_data, aux, aux_data, modis, phenocam_ir, phenocam_rgb)
