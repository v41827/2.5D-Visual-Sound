#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchvision
from transformers import ClapProcessor, ClapModel
from .networks import VisualNet, AudioNet,TextNet, weights_init

class ModelBuilder():
    # builder for visual stream
    def build_visual(self, weights=''):
        pretrained = True
        original_resnet = torchvision.models.resnet18(pretrained)
        net = VisualNet(original_resnet)

        if len(weights) > 0:
            print('Loading weights for visual stream')
            net.load_state_dict(torch.load(weights))
        return net

    #builder for text stream
    def build_text(self, weights='', freeze=True):
        # Always load the base pretrained CLAP model
        clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        net = TextNet(clap_model)

        if not freeze and len(weights) > 0:
            print('Loading custom weights for text stream (for fine-tuning)')
            net.load_state_dict(torch.load(weights))

        if freeze:
            print('Freezing text encoder (using pretrained CLAP weights)')
            for param in net.parameters():
                param.requires_grad = False
        return net

    #builder for audio stream
    def build_audio(self, ngf=64, input_nc=2, output_nc=2, weights=''):
        #AudioNet: 5 layer UNet
        net = AudioNet(ngf, input_nc, output_nc)

        net.apply(weights_init)
        if len(weights) > 0:
            print('Loading weights for audio stream') ## Load CLAP pretrained model from Hugging Face
            net.load_state_dict(torch.load(weights))
        return net
