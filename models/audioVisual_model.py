#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from . import networks,criterion
from torch.autograd import Variable

class AudioVisualModel(torch.nn.Module): # defines the audio-visual model for training, inherits from torch.nn.Module
    def name(self):
        return 'AudioVisualModel'

    def __init__(self, nets, opt):
        super(AudioVisualModel, self).__init__()
        self.opt = opt
        #initialize model
        self.net_visual, self.net_audio, self.net_text = nets #Unpacks nets (a tuple of 2 networks) into self.net_visual (instance of VisualNet) and self.net_audio (instance of AudioNet)

    def forward(self, input, volatile=False):
        visual_input = input['frame']
        audio_diff = input['audio_diff_spec']
        audio_mix = input['audio_mix_spec']
        text_input = input['text']
        audio_gt = Variable(audio_diff[:,:,:-1,:], requires_grad=False)

        input_spectrogram = Variable(audio_mix, requires_grad=False, volatile=volatile)
        visual_feature = self.net_visual(Variable(visual_input, requires_grad=False, volatile=volatile))
        text_feature = self.net_text(text_input)
        mask_prediction = self.net_audio(input_spectrogram, visual_feature, text_feature)

        #complex masking to obtain the predicted spectrogram
        spectrogram_diff_real = input_spectrogram[:,0,:-1,:] * mask_prediction[:,0,:,:] - input_spectrogram[:,1,:-1,:] * mask_prediction[:,1,:,:]
        spectrogram_diff_img = input_spectrogram[:,0,:-1,:] * mask_prediction[:,1,:,:] + input_spectrogram[:,1,:-1,:] * mask_prediction[:,0,:,:]
        binaural_spectrogram = torch.cat((spectrogram_diff_real.unsqueeze(1), spectrogram_diff_img.unsqueeze(1)), 1)

        output =  {'mask_prediction': mask_prediction, 'binaural_spectrogram': binaural_spectrogram, 'audio_gt': audio_gt}
        return output


if __name__ == "__main__": #text script 
    import torch
    import torchvision
    from models.networks import VisualNet, TextNet, AudioNet

    # Dummy VisualNet
    resnet = torchvision.models.resnet18(weights=None)
    visual_net = VisualNet(resnet)

    # Dummy TextNet
    class DummyCLAP:
        def get_text_features(self, **x):
            return torch.randn(2, 512)
    clap = DummyCLAP()
    text_net = TextNet(clap)

    # Dummy AudioNet
    audio_net = AudioNet()

    # 組合成 AudioVisualModel
    nets = (visual_net, audio_net, text_net)
    model = AudioVisualModel(nets, opt=None)

    # 準備 dummy input
    dummy_input = {
        'frame': torch.randn(2, 3, 224, 448),
        'audio_diff_spec': torch.randn(2, 2, 257, 256),
        'audio_mix_spec': torch.randn(2, 2, 257, 256),
        'text': {}
    }

    output = model(dummy_input)
    print("Output keys:", output.keys())
    print("mask_prediction shape:", output['mask_prediction'].shape)
    print("binaural_spectrogram shape:", output['binaural_spectrogram'].shape)
    print("audio_gt shape:", output['audio_gt'].shape)