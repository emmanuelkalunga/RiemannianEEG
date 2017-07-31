"""
LISV SSVEP dataset.
"""
from .base import BaseDataset
#from moabb.datasets.base import BaseDataset
from mne.io import RawArray
from mne.channels import read_montage
from mne import create_info
import pickle
import gzip
import os
import pandas as pd
import numpy as np


class lisvSSVEP(BaseDataset):
    """LISV exoskeleton SSVEP dataset"""
    def __init__(self):
        self.subject_list = range(1, 13)
        self.name = 'lisv exoskeleton ssvep'
        self.tmin = 2
        self.tmax = 6
        self.paradigm = 'SSVEP'
        self.event_id = {'Resting':33024, '13Hz':33025, '21Hz':33026, '17Hz':33027}

    def get_data(self, subjects):
        """return data for a list of subjects."""
        data = []
        for subject in subjects:
            data.append(self._get_single_subject_data(subject))
        return data

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        data_path = '/Users/emmanuelkalunga/Documents/School/EEG_covariance_classification/data/dataset-ssvep-exoskeleton/'
        path_subject = data_path+'subject{0:0>2}/'.format(subject)

        raw_files = []
        sample_rate = 256
        ch_names = ['Oz','O1','O2','PO3','POz','PO7','PO8','PO4','Stim']
        ch_types = ['eeg'] * 8 + ['stim']
        montage = read_montage('standard_1005')
        info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=256., montage=montage)
        for fn in os.listdir(path_subject):
            if fn.endswith('.pz'):
                with gzip.open(path_subject+fn, 'rb') as f:
                    o = pickle.load(f, encoding='latin1')
                raw_signal = o['raw_signal']
                event_pos = o['event_pos'].reshape((o['event_pos'].shape[0]))
                event_type = o['event_type'].reshape((o['event_type'].shape[0]))
                data = pd.DataFrame(raw_signal)
                data['Stim'] = 0
                data.loc[event_pos[np.where(event_type >=33024)], 'Stim'] = event_type[np.where(event_type >=33024)]

                raw_files.append(RawArray(data=data.values.T, info=info, verbose=False))
        return raw_files
