#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.utils.data
from data.base_data_loader import BaseDataLoader
from transformers import ClapProcessor

processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

def CreateDataset(opt):
    dataset = None
    if opt.model == 'audioVisual':
        from data.audioVisual_dataset import AudioVisualDataset
        dataset = AudioVisualDataset()
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.model)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

def custom_collate_fn(batch):
    batch_texts = [item['text'] for item in batch]
    text_inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True) # Process text inputs
    text_inputs = {key: value.to(batch[0]['frame'].device) for key, value in text_inputs.items()}
    return {
        'frame': torch.stack([item['frame'] for item in batch]),
        'text': text_inputs,
        'audio_diff_spec': torch.stack([item['audio_diff_spec'] for item in batch]),
        'audio_mix_spec': torch.stack([item['audio_mix_spec'] for item in batch])
    }

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=True,
            num_workers=int(opt.nThreads),
            collate_fn=custom_collate_fn)

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data
