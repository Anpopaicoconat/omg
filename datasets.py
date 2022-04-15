import pandas as pd
import torch
import torchaudio
import numpy as np
import sklearn
import sklearn.feature_extraction
import tqdm
import os

class text_dataset(torch.utils.data.Dataset):
    def __init__(self, x, y, vectorizer=None, le=None):
        self.x = x
        self.y = y.to_numpy()#np.expand_dims(, axis=1)
        if le and (self.labels is not None):
            self.labels = le.transform(self.labels)
        self.vectorizer = vectorizer
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.vectorizer is not None:
            x = self.vectorizer.transform([self.x[idx]]).toarray()[0]

        y = self.y[idx]
        return x, y


class audio_dataset1(torch.utils.data.Dataset):

    def __init__(self, data_dir=None, fnames=None, wavs=None, labels=None, 
                 pad_size=False, le=None, extractor=None, aggregator=None, 
                 as_numpy=False):
        '''
        data_dir: str - path to dir with .wav files
        fnames: list - every item is name of .wav file in data_dir
        labels: list - labels for files in data_dir in the same order as fnames
        pad_size: 'mean', 'max', 'min' , False or int size for pad and cut if 
                  False do not use padding and truncutting
        le: function to encode labels for lerning
        extractor: function for feature extruction from .wav files
        aggregator: function for aggregation features and reducing dims
        as_numpy: False - using for torch.dataloader, True - for getting np.array
        '''
        super().__init__()
        self.extractor = extractor
        self.aggregator = aggregator
        self.as_numpy = as_numpy
        self.data_dir = data_dir
        self.fnames = fnames
        self.wavs = wavs
        self.labels = labels
        if le and (self.labels is not None):
            self.labels = le.transform(self.labels)

        len_sum = 0
        len_max = 0
        len_min = np.inf
        if self.fnames is not None:
            self.fnames = [os.path.join(self.data_dir, fname) for fname in self.fnames]
            l = self.fnames
        else:
            l = wavs
        for fname in tqdm.tqdm(l):
            if type(fname) is str:
                wav =  torchaudio.load(fname)[0][0]
            else:
                wav = l[0][0]
            len_sum += len(wav)
            if len_max < len(wav):
                len_max = len(wav)
            if len_min > len(wav):
                len_min = len(wav)
        self.len_mean = len_sum/len(l)
        self.len_max = len_max
        self.len_min = len_min
        if pad_size == 'mean':
            self.pad_size = int(self.len_mean)
        elif pad_size == 'max':
            self.pad_size = self.len_max
        elif pad_size == 'max':
            self.pad_size = self.len_min
        else:
            self.pad_size = pad_size

    def __len__(self):
        if self.fnames is not None:
            return len(self.fnames)
        else:
            return len(self.wavs)
    
    def __getitem__(self, idx):
        if self.fnames is not None:
            wav, sr =  torchaudio.load(self.fnames[idx])
        else:
            wav, sr =  self.wavs[idx]
        if self.labels is not None:
            labels = self.labels[idx]
        else:
            labels = []
        if self.pad_size:
            p = (self.pad_size - len(wav.flatten())) // 2 + 1
            if p>0:
                wav =torch.nn.functional.pad(wav, (p, p), value=0.0)
            wav = wav[:, :self.pad_size]
        if self.extractor is not None:
            wav = self.extractor(wav)
        if self.aggregator is not None:
            wav = self.aggregator(wav)
        if self.as_numpy:
            wav = wav.numpy()
            wav = np.append(wav, labels)
            return wav
        else:
            return wav, labels

    def numpy(self):
        self.as_numpy = True
        return np.array(self)

class audio_dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir=None, fnames=None, wavs=None, labels=None, 
                 pad_size=False, le=None, extractor=None, aggregator=None, 
                 as_numpy=False):
        '''
        data_dir: str - path to dir with .wav files
        fnames: list - every item is name of .wav file in data_dir
        labels: list - labels for files in data_dir in the same order as fnames
        pad_size: 'mean', 'max', 'min' , False or int size for pad and cut if 
                  False do not use padding and truncutting
        le: function to encode labels for lerning
        extractor: function for feature extruction from .wav files
        aggregator: function for aggregation features and reducing dims
        as_numpy: False - using for torch.dataloader, True - for getting np.array
        '''
        super().__init__()
        self.extractor = extractor
        self.aggregator = aggregator
        self.as_numpy = as_numpy
        self.data_dir = data_dir
        self.fnames = fnames
        self.wavs = wavs
        self.labels = labels
        if le and (self.labels is not None):
            self.labels = le.transform(self.labels)
        len_sum = 0
        len_max = 0
        len_min = np.inf
        if self.fnames is not None:
            self.fnames = [os.path.join(self.data_dir, fname) for fname in self.fnames]
            l = self.fnames
        else:
            l = wavs
        for fname in tqdm.tqdm(l):
            if type(fname) is str:
                wav =  torchaudio.load(fname)[0][0]
            else:
                wav = l[0][0]
            len_sum += len(wav)
            if len_max < len(wav):
                len_max = len(wav)
            if len_min > len(wav):
                len_min = len(wav)
        self.len_mean = len_sum/len(l)
        self.len_max = len_max
        self.len_min = len_min
        if pad_size == 'mean':
            self.pad_size = int(self.len_mean)
        elif pad_size == 'max':
            self.pad_size = self.len_max
        elif pad_size == 'max':
            self.pad_size = self.len_min
        else:
            self.pad_size = pad_size

    def __len__(self):
        if self.fnames is not None:
            return len(self.fnames)
        else:
            return len(self.wavs)
    
    def __getitem__(self, idx):
        if self.fnames is not None:
            wav, sr =  torchaudio.load(self.fnames[idx])
        else:
            wav, sr =  self.wavs[idx]
        if self.labels is not None:
            labels = self.labels[idx]
        else:
            labels = []
        if self.pad_size:
            p = (self.pad_size - len(wav[0].flatten())) // 2 + 1
            if p>0:
                wav =torch.nn.functional.pad(wav, (p, p), value=0.0)
            wav = wav[:, :self.pad_size]
        if self.extractor is not None:
            wav = self.extractor(wav)
        if self.aggregator is not None:
            wav = self.aggregator(wav)
        if self.as_numpy:
            wav = wav.numpy()
            wav = np.append(wav, labels)
            return wav
        else:
            return wav, labels

    def numpy(self):
        self.as_numpy = True
        return np.array(self)