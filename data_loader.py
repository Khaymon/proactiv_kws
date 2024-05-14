import torch
from torch import nn
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter, rotate

import torchaudio

class ImageDatasetFromPt(Dataset):
    def __init__(self, data_path, freq_mask = 0, time_mask = 0, noise_level = 0, cnt = 20, cuda=True):
        super(ImageDatasetFromPt, self).__init__()
        self.device         = 'cuda' if cuda else 'cpu'
        self.data_dir       = data_path
        data                = torch.load(data_path, map_location=lambda storage, loc: storage)
        self.audios            = []
        self.label          = []

        self.get_mel = self._mel_transform = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(16000, n_mels=40)
        )
        self.preprocess = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask),
            torchaudio.transforms.TimeMasking(time_mask),
        )
        self.noise_level    = noise_level

        for i in range(1, 27):
            start_index     = min(np.where(data[1].data.numpy() == i)[0])
            self.audios.extend(data[0].data.numpy()[start_index:start_index + cnt].tolist())
            self.label.extend(data[1].data.numpy()[start_index:start_index + cnt].tolist())

    def get_transformer(self):
        return self.preproc

    def __getitem__(self, index):
        audio               = self.preprocess(self.get_mel(np.array(self.audios[index])))
        label               = self.label[index]

        if self.noise_level != 0:
            audio += self.noise_level * torch.tensor(np.random.normal(0, (audio.max() - audio.min())/6., audio.shape))

        label               = torch.tensor(label).to(self.device)
        sample              = {'input': audio, 'target': label, 'fname': str(index), 'params': ''}
        return sample

    def __len__(self):
        return len(self.label)
