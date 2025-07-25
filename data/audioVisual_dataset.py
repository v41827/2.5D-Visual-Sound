#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os.path
import time
import librosa
import h5py
import random
import math
import numpy as np
import glob
import torch
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset

def normalize(samples, desired_rms = 0.1, eps = 1e-4):
  rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
  samples = samples * (desired_rms / rms)
  return samples

def generate_spectrogram(audio):
    spectro = librosa.core.stft(audio, n_fft=512, hop_length=160, win_length=400, center=True)
    real = np.expand_dims(np.real(spectro), axis=0)
    imag = np.expand_dims(np.imag(spectro), axis=0)
    spectro_two_channel = np.concatenate((real, imag), axis=0)
    return spectro_two_channel

def process_image(image, augment):
    image = image.resize((480,240))
    w,h = image.size
    w_offset = w - 448
    h_offset = h - 224
    left = random.randrange(0, w_offset + 1)
    upper = random.randrange(0, h_offset + 1)
    image = image.crop((left, upper, left+448, upper+224))

    if augment:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.random()*0.6 + 0.7)
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(random.random()*0.6 + 0.7)
    return image

class AudioVisualDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.audios = []

        #load hdf5 file here
        if opt.hdf5FolderPath is None:
            raise ValueError("--hdf5FolderPath is required but not provided. Please specify the path to the folder containing train.h5, val.h5 and test.h5 files.")
        
        h5f_path = os.path.join(opt.hdf5FolderPath, opt.mode+".h5")
        if not os.path.exists(h5f_path):
            raise FileNotFoundError(f"HDF5 file not found: {h5f_path}. Please check if the file exists and the path is correct.")
        
        h5f = h5py.File(h5f_path, 'r')
        #self.audios = h5f['audio'][:] 
        self.audios = [p.decode() for p in h5f['audio'][:]]

        normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )
        vision_transform_list = [transforms.ToTensor(), normalize]
        self.vision_transform = transforms.Compose(vision_transform_list)

    def __getitem__(self, index):
        #load audio
        audio, audio_rate = librosa.load(self.audios[index], sr=self.opt.audio_sampling_rate, mono=False)
        #audio = self.audios[index]  # its already waveform
        #randomly get a start time for the audio segment from the 10s clip
        audio_start_time = random.uniform(0, 9.9 - self.opt.audio_length) 
        audio_end_time = audio_start_time + self.opt.audio_length #Calculates the end time of the audio segment
        audio_start = int(audio_start_time * self.opt.audio_sampling_rate) #Converts the start time (in seconds) to a sample index
        audio_end = audio_start + int(self.opt.audio_length * self.opt.audio_sampling_rate) #Calculates the end sample index for the segment
        audio = audio[:, audio_start:audio_end] #Extracts the audio segment (for both channels) between the calculated start and end indices.
        audio = normalize(audio) #normalise the audio segment to a fixed rms value for consistent volume
        audio_channel1 = audio[0,:] #separate two channels
        audio_channel2 = audio[1,:]

        #get the frame dir path based on audio path
        path_parts = self.audios[index].strip().split('/')
        path_parts[-1] = path_parts[-1][:-4] + '.mp4'
        path_parts[-2] = 'frames'
        frame_path = '/'.join(path_parts)

        # get the closest frame to the audio segment
        frame_index = int(round((audio_start_time + audio_end_time) / 2.0 + 0.5))  #1 frame extracted per second
        #frame_index = int(round(((audio_start_time + audio_end_time) / 2.0 + 0.05) * 10))  #10 frames extracted per second
        frame = process_image(Image.open(os.path.join(frame_path, str(frame_index).zfill(6) + '.png')).convert('RGB'), self.opt.enable_data_augmentation)
        frame = self.vision_transform(frame)

        #passing the spectrogram of the difference
        audio_diff_spec = torch.FloatTensor(generate_spectrogram(audio_channel1 - audio_channel2))
        audio_mix_spec = torch.FloatTensor(generate_spectrogram(audio_channel1 + audio_channel2))

        return {'frame': frame, 'audio_diff_spec':audio_diff_spec, 'audio_mix_spec':audio_mix_spec}

    def __len__(self):
        return len(self.audios)

    def name(self):
        return 'AudioVisualDataset'



