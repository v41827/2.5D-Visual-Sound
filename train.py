#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import torch
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import ModelBuilder
from models.audioVisual_model import AudioVisualModel
from torch.autograd import Variable
import wandb


def create_optimizer(nets, opt):
    (net_visual, net_audio, net_text) = nets
    param_groups = [{'params': net_visual.parameters(), 'lr': opt.lr_visual},
                    {'params': net_audio.parameters(), 'lr': opt.lr_audio},]

    if opt.optimizer == 'sgd':
        return torch.optim.SGD(param_groups, momentum=opt.beta1, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'adam':
        return torch.optim.Adam(param_groups, betas=(opt.beta1,0.999), weight_decay=opt.weight_decay)

def decrease_learning_rate(optimizer, decay_factor=0.94): #decrease learning rate by 6% at specified intervals
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_factor

#used to display validation loss
def display_val(model, loss_criterion, writer, index, dataset_val, opt):
    losses = []
    with torch.no_grad():
        for i, val_data in enumerate(dataset_val):
            if i < opt.validation_batches:
                output = model.forward(val_data)
                loss = loss_criterion(output['binaural_spectrogram'], output['audio_gt'])
                losses.append(loss.item()) 
            else:
                break
    avg_loss = sum(losses)/len(losses)
    return avg_loss 

def main():
    #parse arguments
    opt = TrainOptions().parse()
    
    # Debug: Print the hdf5FolderPath to see what's being passed
    print(f"DEBUG: hdf5FolderPath = {opt.hdf5FolderPath}")
    print(f"DEBUG: type of hdf5FolderPath = {type(opt.hdf5FolderPath)}")
    
    if opt.wandb:
        wandb.init(project=opt.wandb_project, config=vars(opt)) #Track all training hyperparameters from my opt object
    opt.device = torch.device("cuda")

    #construct data loader
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training clips = %d' % dataset_size)

    #create validation set data loader if validation_on option is set
    if opt.validation_on:
        #temperally set to val to load val data
        opt.mode = 'val'
        data_loader_val = CreateDataLoader(opt)
        dataset_val = data_loader_val.load_data()
        dataset_size_val = len(data_loader_val)
        print('#validation clips = %d' % dataset_size_val)
        opt.mode = 'train' #set it back

    else:
        writer = None
    writer = None

    # network builders
    builder = ModelBuilder()
    net_visual = builder.build_visual(weights=opt.weights_visual)
    net_audio = builder.build_audio(
            ngf=opt.unet_ngf,
            input_nc=opt.unet_input_nc,
            output_nc=opt.unet_output_nc,
            weights=opt.weights_audio)
    net_text = builder.build_text(weights=opt.weights_text, freeze=opt.freeze_text)

    nets = (net_visual, net_audio, net_text)

    # construct our audio-visual model
    model = AudioVisualModel(nets, opt) #前一行的nets pass進來這裡
    model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
    model.to(opt.device)

    # set up optimizer 
    optimizer = create_optimizer(nets, opt)

    # set up loss function
    loss_criterion = torch.nn.MSELoss()
    if(len(opt.gpu_ids) > 0):
        loss_criterion.cuda(opt.gpu_ids[0])

    # initialization
    total_steps = 0
    data_loading_time = []
    model_forward_time = []
    model_backward_time = []
    batch_loss = []
    best_err = float("inf")

    for epoch in range(1, opt.niter+1):
            torch.cuda.synchronize()
            epoch_start_time = time.time()

            if(opt.measure_time):
                    iter_start_time = time.time()
            for i, data in enumerate(dataset): #在train.py(here)，每次訓練迭代時一次輸入一個 batch 的多模態資料，這些資料是由 custom_collate_fn() 預先對齊和打包好的
                    if(opt.measure_time):
                        torch.cuda.synchronize()
                        iter_data_loaded_time = time.time()

                    total_steps += opt.batchSize

                    # forward pass
                    model.zero_grad()
                    output = model(data) # forward pass through the model

                    # compute loss
                    loss = loss_criterion(output['binaural_spectrogram'], Variable(output['audio_gt'], requires_grad=False))
                    batch_loss.append(loss.item())

                    if(opt.measure_time):
                        torch.cuda.synchronize()
                        iter_data_forwarded_time = time.time()

                    # update optimizer
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if(opt.measure_time):
                            iter_model_backwarded_time = time.time()
                            data_loading_time.append(iter_data_loaded_time - iter_start_time)
                            model_forward_time.append(iter_data_forwarded_time - iter_data_loaded_time)
                            model_backward_time.append(iter_model_backwarded_time - iter_data_forwarded_time)

                    if(total_steps // opt.batchSize % opt.display_freq == 0):
                            print('Display training progress at (epoch %d, total_steps %d)' % (epoch, total_steps))
                            avg_loss = sum(batch_loss) / len(batch_loss)
                            print('Average loss: %.3f' % (avg_loss))
                            batch_loss = []
                            wandb.log({"train/loss": avg_loss, "steps": total_steps})
                            if(opt.measure_time):
                                    print('average data loading time: ' + str(sum(data_loading_time)/len(data_loading_time)))
                                    print('average forward time: ' + str(sum(model_forward_time)/len(model_forward_time)))
                                    print('average backward time: ' + str(sum(model_backward_time)/len(model_backward_time)))
                                    data_loading_time = []
                                    model_forward_time = []
                                    model_backward_time = []
                            print('end of display \n')

                    if(total_steps // opt.batchSize % opt.save_latest_freq == 0):
                            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                            torch.save(net_visual.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'visual_latest.pth'))
                            torch.save(net_audio.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'audio_latest.pth'))
                            torch.save(net_text.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'text_latest.pth'))

                    if(total_steps // opt.batchSize % opt.validation_freq == 0 and opt.validation_on):
                            model.eval()
                            opt.mode = 'val'
                            print('Display validation results at (epoch %d, total_steps %d)' % (epoch, total_steps))
                            val_err = display_val(model, loss_criterion, writer, total_steps, dataset_val, opt)
                            wandb.log({"val/loss": val_err, "steps": total_steps})
                            print('end of display \n')
                            model.train()
                            opt.mode = 'train'
                            #save the model that achieves the smallest validation error
                            if val_err < best_err:
                                best_err = val_err
                                print('saving the best model (epoch %d, total_steps %d) with validation error %.3f\n' % (epoch, total_steps, val_err))
                                torch.save(net_visual.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'visual_best.pth'))
                                torch.save(net_audio.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'audio_best.pth'))
                                torch.save(net_text.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'text_best.pth'))


                    if(opt.measure_time):
                            iter_start_time = time.time()

            if(epoch % opt.save_epoch_freq == 0):
                    print('saving the model at the end of epoch %d, total_steps %d' % (epoch, total_steps))
                    torch.save(net_visual.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, str(epoch) + '_visual.pth'))
                    torch.save(net_audio.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, str(epoch) + '_audio.pth'))
                    torch.save(net_text.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, str(epoch) + '_text.pth'))

            #decrease learning rate 6% every opt.learning_rate_decrease_itr epochs
            if(opt.learning_rate_decrease_itr > 0 and epoch % opt.learning_rate_decrease_itr == 0):
                decrease_learning_rate(optimizer, opt.decay_factor)
                print('decreased learning rate by ', opt.decay_factor)
                # Log updated learning rates to wandb
                for i, param_group in enumerate(optimizer.param_groups):
                    wandb.log({f"lr/group_{i}": param_group['lr'], "epoch": epoch})

if __name__ == '__main__':
    main()
