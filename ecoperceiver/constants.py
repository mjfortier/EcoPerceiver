EC_PREDICTORS = ('DOY', 'TOD', 'TA', 'P', 'RH', 'VPD', 'PA', 'CO2', 'SW_IN', 'SW_IN_POT', 'SW_OUT', 'LW_IN', 'LW_OUT',
                 'NETRAD', 'PPFD_IN', 'PPFD_OUT', 'WS', 'WD', 'USTAR', 'SWC_1', 'SWC_2', 'SWC_3', 'SWC_4', 'SWC_5',
                 'TS_1', 'TS_2', 'TS_3', 'TS_4', 'TS_5', 'WTD', 'G', 'H')

EC_TARGETS = ('NEE', 'GPP_DT', 'GPP_NT', 'RECO_DT', 'RECO_NT', 'FCH4', 'LE')

GEO_PREDICTORS = ('lat', 'lon', 'elev')

IGBP_CODES = ('ENF', 'MF', 'WET', 'CRO', 'GRA', 'WAT', 'SAV', 'DBF', 'CSH', 'OSH', 'EBF', 'WSA', 'BSV', 'URB',
              'DNF', 'CVM', 'SNO')

IGBP_ACRONYMS = {
    0: 'WAT', 1: 'ENF', 2: 'EBF', 3: 'DNF', 4: 'DBF', 5: 'MF', 6: 'CSH',
    7: 'OSH', 8: 'WSA', 9: 'SAV', 10: 'GRA', 11: 'WET', 12: 'CRO',
    13: None, 14: 'CVM', 15: 'SNO', 16: None,
}

DEFAULT_NORM = {
    'DOY': {'cyclic': True, 'norm_max': 366.0, 'norm_min': 0.0},
    'TOD': {'cyclic': True, 'norm_max': 24.0, 'norm_min': 0.0},
    'TA': {'cyclic': False, 'norm_max': 80.0, 'norm_min': -80.0},
    'P': {'cyclic': False, 'norm_max': 100.0, 'norm_min': 0.0},
    'RH': {'cyclic': False, 'norm_max': 100.0, 'norm_min': 0.0},
    'VPD': {'cyclic': False, 'norm_max': 110.0, 'norm_min': 0.0},
    'PA': {'cyclic': False, 'norm_max': 110.0, 'norm_min': 0.0},
    'CO2': {'cyclic': False, 'norm_max': 750.0, 'norm_min': 0.0},
    'SW_IN': {'cyclic': False, 'norm_max': 1500.0, 'norm_min': -1500.0},
    'SW_IN_POT': {'cyclic': False, 'norm_max': 1500.0, 'norm_min': -1500.0},
    'SW_OUT': {'cyclic': False, 'norm_max': 500.0, 'norm_min': -500.0},
    'LW_IN': {'cyclic': False, 'norm_max': 1000.0, 'norm_min': -1000.0},
    'LW_OUT': {'cyclic': False, 'norm_max': 1000.0, 'norm_min': -1000.0},
    'NETRAD': {'cyclic': False, 'norm_max': 1000.0, 'norm_min': -1000.0},
    'PPFD_IN': {'cyclic': False, 'norm_max': 2500.0, 'norm_min': -2500.0},
    'PPFD_OUT': {'cyclic': False, 'norm_max': 1000.0, 'norm_min': -1000.0},
    'WS': {'cyclic': False, 'norm_max': 100.0, 'norm_min': -100.0},
    'WD': {'cyclic': True, 'norm_max': 360.0, 'norm_min': 0.0},
    'USTAR': {'cyclic': False, 'norm_max': 4.0, 'norm_min': -4.0},
    'SWC_1': {'cyclic': False, 'norm_max': 0.0, 'norm_min': 100.0},
    'SWC_2': {'cyclic': False, 'norm_max': 0.0, 'norm_min': 100.0},
    'SWC_3': {'cyclic': False, 'norm_max': 0.0, 'norm_min': 100.0},
    'SWC_4': {'cyclic': False, 'norm_max': 0.0, 'norm_min': 100.0},
    'SWC_5': {'cyclic': False, 'norm_max': 0.0, 'norm_min': 100.0},
    'TS_1': {'cyclic': False, 'norm_max': 40.0, 'norm_min': -40.0},
    'TS_2': {'cyclic': False, 'norm_max': 40.0, 'norm_min': -40.0},
    'TS_3': {'cyclic': False, 'norm_max': 40.0, 'norm_min': -40.0},
    'TS_4': {'cyclic': False, 'norm_max': 40.0, 'norm_min': -40.0},
    'TS_5': {'cyclic': False, 'norm_max': 40.0, 'norm_min': -40.0},
    'WTD': {'cyclic': False, 'norm_max': -3.0, 'norm_min': 3.0},
    'G': {'cyclic': False, 'norm_max': 700.0, 'norm_min': -700.0},
    'H': {'cyclic': False, 'norm_max': 700.0, 'norm_min': -700.0},
    'LE': {'cyclic': False, 'norm_max': 700.0, 'norm_min': -700.0},
}