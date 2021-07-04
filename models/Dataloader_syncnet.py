import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import librosa
import cv2

class Data(Dataset):
    def __init__(self, data_dir):
        super(Data, self).__init__()
        self.data_dir = data_dir
        self.T = 5
        self.img_size = 96
        self.sr = 16000
        self.allfiles = os.listdir(self.data_dir)

    def __getitem__(self,idx):
        
        while True:
            vid = self.allfiles[idx]
            path = os.path.join(self.data_dir, vid)
            frames = [frame for frame in sorted(os.listdir(path)) if frame.endswith('.jpg')]
            if len(frames) - self.T <=1:
                idx = random.choice(range(len(self.allfiles)))
            else:
                break
        
        right_frame_idx = random.choice(range(len(frames) - self.T))
        wrong_frame_idx = random.choice(range(len(frames) - self.T))
        while right_frame_idx == wrong_frame_idx:
            wrong_frame_idx = random.choice(range(len(frames) - self.T))

        audio_file = list(set(os.listdir(path)) - set(frames))[0]
        audio, _ = librosa.load(os.path.join(path, audio_file), sr = self.sr)
        
        if random.choice([True, False]):
            y = torch.ones(1).float()
            frames_chosen = []
            for frame in frames[right_frame_idx: right_frame_idx + self.T]:
                img = cv2.imread(os.path.join(path, frame))
                img = cv2.resize(img, (self.img_size, self.img_size))
                frames_chosen += [torch.Tensor(img/255)]
            x = torch.cat(frames_chosen, dim = 2).permute(2,0,1)
            x = x[:, self.img_size//2:, :]
        else:
            y = torch.ones(1).float()
            frames_chosen = []
            for frame in frames[wrong_frame_idx: wrong_frame_idx + self.T]:
                img = cv2.imread(os.path.join(path, frame))
                img = cv2.resize(img, (self.img_size, self.img_size))
                frames_chosen += [torch.Tensor(img/255)]
            x = torch.cat(frames_chosen, dim = 2).permute(2,0,1)
            x = x[:, self.img_size//2:, :]

        start = int(right_frame_idx*self.sr/25)
        end = start + int(self.T*self.sr/25)
        audio = audio[start:end]
        if audio.shape[0] != int(self.T*self.sr/25):
            audio = np.pad(audio, (0,int(self.T*self.sr/25) - audio.shape[0]))
        spec = librosa.feature.melspectrogram(audio, sr = self.sr, n_fft = 800, win_length = 800, hop_length = 200, n_mels = 80)
        spec = spec/np.max(spec)
        spec = torch.Tensor(spec).unsqueeze(0)
        

        return x, spec, y
    
    def __len__(self):
        return len(self.allfiles)

