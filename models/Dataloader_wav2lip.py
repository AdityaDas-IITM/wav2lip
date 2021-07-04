import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
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
        self.trans = ToTensor()
        self.allfiles = os.listdir(self.data_dir)
    
    def get_spec(self, audio, index):
        start = int(index*self.sr/25)
        end = start + int(self.T*self.sr/25)
        audio = audio[start:end]
        if audio.shape[0] != int(self.T*self.sr/25):
            audio = np.pad(audio, (0,int(self.T*self.sr/25) - audio.shape[0]))
        spec = librosa.feature.melspectrogram(audio, sr = self.sr, n_fft = 800, hop_length = 200, n_mels = 80)
        spec = self.trans(spec)
        spec = spec.unsqueeze(0)
        return spec

    def get_indiv_spec(self, audio, index):
        start = int(index*self.sr/25) - int(2*self.sr/25)
        mels = []
        for i in range(self.T):
            end = start + int(self.T*self.sr/25)
            audio_crop = audio[start:end]
            if audio_crop.shape[0] != int(self.T*self.sr/25):
                audio_crop = np.pad(audio_crop, (0,int(self.T*self.sr/25) - audio_crop.shape[0]))
            spec = librosa.feature.melspectrogram(audio_crop, sr = self.sr, n_fft = 800, hop_length = 200, n_mels = 80)
            spec = self.trans(spec).unsqueeze(0)
            mels += [spec]
            start = start + int(self.sr/25)

        return np.array(mels)


    def __getitem__(self,idx):
        
        while True:
            vid = self.allfiles[idx]
            path = os.path.join(self.data_dir, vid)
            frames = [frame for frame in sorted(os.listdir(path)) if frame.endswith('.jpg')]
            if len(frames) - self.T <=1:
                idx = random.choice(range(len(self.allfiles)))
            else:
                break
        
        right_frame_idx = random.choice(range(2, len(frames) - self.T))
        wrong_frame_idx = random.choice(range(len(frames) - self.T))
        while right_frame_idx == wrong_frame_idx:
            wrong_frame_idx = random.choice(range(len(frames) - self.T))

        audio_file = list(set(os.listdir(path)) - set(frames))[0]
        audio, _ = librosa.load(os.path.join(path, audio_file), sr = self.sr)
        
        right_frames = []
        for frame in frames[right_frame_idx: right_frame_idx + self.T]:
            img = cv2.imread(os.path.join(path, frame))
            img = cv2.resize(img, (self.img_size, self.img_size))
            right_frames += [(img/255)]
        
        wrong_frames = []
        for frame in frames[wrong_frame_idx: wrong_frame_idx + self.T]:
            img = cv2.imread(os.path.join(path, frame))
            img = cv2.resize(img, (self.img_size, self.img_size))
            wrong_frames += [(img/255)]
        #right_frames = wrong_frames = T X H X W X 3
        right_frames = np.transpose(np.array(right_frames), (3,0,1,2)) 
        y = right_frames.copy()
        right_frames[:,:,self.img_size//2:] = 0
        wrong_frames = np.transpose(np.array(wrong_frames), (3,0,1,2)) 

        x = np.concatenate([right_frames, wrong_frames], axis = 0)
        x = torch.Tensor(x)
        spec = self.get_spec(audio, right_frame_idx)
        indiv_spec = self.get_indiv_spec(audio, right_frame_idx)
        indiv_spec = torch.Tensor(indiv_spec)
        y = torch.Tensor(y)
        
        return x, indiv_spec, spec, y

    def __len__(self):
        return len(self.allfiles)