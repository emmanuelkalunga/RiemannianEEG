"""SSVEP contexts"""

import numpy as np
from .base import WithinSubjectContext, BaseContext
from mne import Epochs, find_events
from mne.epochs import concatenate_epochs, equalize_epoch_counts
from mne import create_info
from mne.io import RawArray
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut, KFold
from scipy.signal import filtfilt, butter, buttord
from collections import defaultdict

class BaseSSVEP(BaseContext):
    """Base SSVEP context


    Parameters
    ----------
    datasets : List of Dataset instances.
        List of dataset instances on which the pipelines will be evaluated.
    pipelines : Dict of pipelines instances.
        Dictionary of pipelines. Keys identifies pipeline names, and values
        are scikit-learn pipelines instances.
    frequencies: list | None (default)
        list of stimuli frequencies

    See Also
    --------

    """

    def __init__(self, datasets, pipelines, frequencies=None):
        self.frequencies = frequencies
        super().__init__(datasets, pipelines)

    def _epochs(self, dataset, subjects, event_id):
        """epoch data"""
        raws = dataset.get_data(subjects=subjects)
        raws = raws[0]
        ep = []
        # we process each run independently
        for raw in raws:

            # find events
            events = find_events(raw, shortest_event=0, verbose=False)

            # pick some channels
            raw.pick_types(meg=False, eeg=True, stim=False,
                           eog=False, exclude='bads')

            # filter data
            #raw.filter(self.fmin, self.fmax, method='iir')

            # epoch data
            epochs = Epochs(raw, events, event_id, dataset.tmin, dataset.tmax,
                            proj=False, baseline=None, preload=True,
                            verbose=False)
            ep.append(epochs)
        return ep

    def prepare_data(self, dataset, subjects):
        """Prepare data for classification."""

        event_id = dataset.event_id
        epochs = self._epochs(dataset, subjects, event_id)
        groups = []
        full_epochs = []

        for ii, epoch in enumerate(epochs):
            epochs_list = [epoch[k] for k in event_id]
            # equalize for accuracy
            equalize_epoch_counts(epochs_list)
            ep = concatenate_epochs(epochs_list)
            groups.extend([ii] * len(ep))
            full_epochs.append(ep)

        epochs = concatenate_epochs(full_epochs)
        #X = epochs.get_data()*1e6
        X = epochs.get_data()
        y = epochs.events[:, -1]
        groups = np.asarray(groups)
        return X, y, groups

    def score(self, clf, X, y, groups, n_jobs=1):
        """get the score"""
        if len(np.unique(groups)) > 1:
            # if group as different values, use group
            cv = LeaveOneGroupOut()
        else:
            # else use kfold
            cv = KFold(5, shuffle=True, random_state=45)

        auc = cross_val_score(clf, X, y, groups=groups, cv=cv,
                              scoring='accuracy', n_jobs=n_jobs)
        return auc.mean()

class ExtentedSSVEP(BaseSSVEP, WithinSubjectContext):
    """Base SSVEP context

    Parameters
    ----------
    datasets : List of Dataset instances.
        List of dataset instances on which the pipelines will be evaluated.
    pipelines : Dict of pipelines instances.
        Dictionary of pipelines. Keys identifies pipeline names, and values
        are scikit-learn pipelines instances.
    frequencies: list | None (default)
        list of stimuli frequencies

    See Also
    --------

    """

    # def __init__(self, datasets, pipelines, frequencies=None):
    #     self.frequencies = frequencies
    #     super().__init__(datasets, pipelines)

    def _epochs(self, dataset, subjects, event_id):
        """epoch data"""
        raws = dataset.get_data(subjects=subjects)
        raws = raws[0]
        ep = []
        # we process each run independently
        for raw in raws:

            # find events
            events = find_events(raw, shortest_event=0, verbose=False)

            # pick some channels
            raw.pick_types(meg=False, eeg=True, stim=False,
                           eog=False, exclude='bads')

            # Get X_ext
            raw_data = raw.get_data()
            frequencies = self.frequencies
            filter_coef = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))# dict of sampling frequencies, passed frequency and coef
            filter_coef['256.0']['13.0']['a'] = [1.0,-1.883459171625896,0.983572253981791]
            filter_coef['256.0']['13.0']['b'] = [0.008213873009104,0,-0.008213873009104]
            filter_coef['256.0']['17.0']['a'] = [1.0,-1.812573732694827,0.982661267813152]
            filter_coef['256.0']['17.0']['b'] = [0.008669366093424,0,-0.008669366093424]
            filter_coef['256.0']['21.0']['a'] = [1.0,-1.725001953692769,0.982556658907297]
            filter_coef['256.0']['21.0']['b'] = [0.008721670546351,0,-0.008721670546351]
            freq_band = 0.1
            ch_names_init = raw.ch_names
            sfreq = raw.info['sfreq']
            for f in frequencies:
                a = filter_coef[str(float(raw.info['sfreq']))][str(float(f))]['a']
                b = filter_coef[str(float(raw.info['sfreq']))][str(float(f))]['b']
                data_f = self._butter_bandpass(raw_data, lowcut=f-freq_band, highcut=f+freq_band, fs=sfreq, a=a, b=b)
                ch_names = [x+'-{}Hz'.format(f) for x in ch_names_init]
                ch_types = ['eeg'] * len(ch_names)
                info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq, montage=None)
                addedChan = RawArray(data=data_f, info=info, verbose=False)
                raw.add_channels([addedChan])
            raw.drop_channels(ch_names_init)

            # epoch data
            epochs = Epochs(raw, events, event_id, dataset.tmin, dataset.tmax,
                            proj=False, baseline=None, preload=True,
                            verbose=False)
            ep.append(epochs)
        return ep

    def _butter_bandpass(self, signal, lowcut, highcut, fs, a=None, b=None):
        # Descrption:
        #    Band pass filter a signal between 2 frequecies
        # Parameters:
        #    - signal: array-like. n x c array with n samplesa and c variables
        #    - lowcut and highcut: cutting frequencies in Hz
        #    - fs: float. Sampling frequency
        #    - a, b: Denominator (`a`) and numerator (`b`) polynomials of the IIR filter
        if (a is None) | (b is None):
            Rp = 3
            Rs = 10
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            Wp = [low, high]
            Ws = [x-(1/nyq) for x in Wp]
            [order,Wn] = buttord(Wp,Ws,Rp,Rs)
            b, a = butter(order, Wn, 'band')
        filtered = filtfilt(b, a, signal, axis=1)
        return filtered
