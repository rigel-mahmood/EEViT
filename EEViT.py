from unittest.mock import patch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import sys

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pandas as pd
import csv
import time
import timm
from timm.models import create_model
from randomaug import RandAugment
from models.vit import ViT
from EEViT_IPTransfomer import EEViT_IPTransformer
from PARTransfomer import PARTransformer
from EEViT_PARTransfomer import EEViT_PARTransformer
from models.swin import swin_t
from models.cait import CaiT
import twins

import Utils
import UtilsTinyImagenet

import time
from tqdm import tqdm
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--dataset', default='CIFAR10') # options: CIFAR10, CIFAR100, SVHN, Imagenet,TINYIMAGENET
parser.add_argument('--dataset_classes', default='10') # options: 10 for CIFAR10, 100 fr CIFAR100, 10 for SVHN, 1000 for Imagenet, 200 for TinyImagenet
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
parser.add_argument('--opt', default="adam")
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
parser.add_argument('--resume', default=False, help='resume from checkpoint')
parser.add_argument('--noaug', default=True, help='disable use randomaug')
parser.add_argument('--nowandb', action='store_true', help='disable wandb')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='EEViT_PAR') # options: vit, swin, cait, twins, par (perceiverAR), EEViT_IP, EEViT_PAR
parser.add_argument('--heads', default='6')
parser.add_argument('--layers', default='8')  # depth
parser.add_argument('--bs', default='64')  # was 512
parser.add_argument('--image_size', default="32") #Imagesize: 32 for CIFAR, SVHN; 224 for imagenet, 64 for TinyImageNet
parser.add_argument('--n_epochs', type=int, default='200')
parser.add_argument('--patch_size', default='4', type=int, help="patch size for ViT") #4; 14 for imagenet, 8 for TinyImageNet
parser.add_argument('--dim', default="384", type=int) # or 512
parser.add_argument('--use_cls_token', default=False, type=bool, help="use cls token")
parser.add_argument('--use_cls_token_last_n_layers', default='2', type=int, help="use cls token for last n layers")

args = parser.parse_args()
best_acc = 0

def count_parameters(model): # count number of trainable parameters in the model
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    mainDir_TinyImageNet = "c:/tiny-imagenet-200" # tinyimagenet data folder

    bs = int(args.bs)
    imsize = int(args.image_size)

    use_amp = not args.noamp
    aug = args.noaug

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    if args.net=="vit_timm":
        size = 384
    else:
        size = imsize 
    if args.dataset == "CIFAR10":
        trainloader, testloader = Utils.get_loaders_CIFAR10(size, bs, aug=args.noaug)
    if args.dataset == "CIFAR10" and size == 224:
        trainloader, testloader = Utils.get_loaders_CIFAR10_224(size, bs, aug=args.noaug)
    if args.dataset == "CIFAR100":
        trainloader, testloader = Utils.get_loaders_CIFAR100(size, bs, aug=args.noaug)
    if args.dataset == "SVHN":
        trainloader, testloader = Utils.get_loaders_SVHN(size, bs, aug=args.noaug)
    if args.dataset == "TINYIMAGENET":
        trainloader, testloader = UtilsTinyImagenet.get_loaders_tiny_imagenet(mainDir_TinyImageNet, bs, aug=args.noaug)
    if args.dataset == "Imagenet":
        trainloader, testloader = Utils.get_loaders_imagenet(size, bs, aug=args.noaug)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print('==> creating model..')
    if args.net=="vit":
        net = ViT(
        image_size = size,
        patch_size = args.patch_size,
        num_classes = int(args.dataset_classes),
        dim = int(args.dim),
        depth = int(args.layers),
        heads = int(args.heads),
        mlp_dim = int(args.dim)*4, 
        dropout = 0.1,
        emb_dropout = 0.1
    )
    
    if args.net == "EEViT_IP":
        net = EEViT_IPTransformer(
            dim = int(args.dim), 
            num_tokens = int(args.dataset_classes),   
            num_layers = int(args.layers), 
            heads = int(args.heads), 
            sequence_len = int(size/int(args.patch_size)) * int(size/int(args.patch_size)), 
            segment_len = 8, #16; 32 for imagenet, 16 for tinyimagenet, 64 for imagenet
            num_segments = 8, #4; 8 for imagenet
            ff_dropout=0.1, 
            attn_dropout = 0.1,
            apply_pe_all_layers = True,
            image_size= int(args.image_size), 
            patch_size=int(args.patch_size), 
         ).cuda()

    latent_len = 32 # perceiver ar, EEViT_IP latent length
    if args.net == "par":
        net = PARTransformer(
            latent_len = latent_len,
            dim = int(args.dim), 
            num_tokens = int(args.dataset_classes),   
            num_layers = int(args.layers), 
            heads = int(args.heads), 
            sequence_len = 64, #int(1024/(args.patch*args.patch)), # for CIFAR-10/100
            segment_len = 16,
            num_segments = 4,
            ff_dropout=0.1, 
            attn_dropout = 0.1,
            apply_pe_all_layers = True
         ).cuda()
    if args.net == "EEViT_PAR":
        net = EEViT_PARTransformer(
            latent_len = latent_len,
            dim = int(args.dim), 
            num_tokens = int(args.dataset_classes),   
            num_layers = int(args.layers), 
            heads = int(args.heads), 
            sequence_len = int(1024/(args.patch_size*args.patch_size)), # for CIFAR-10/100
            segment_len = 16,
            num_segments = 4,
            ff_dropout=0.1, 
            attn_dropout = 0.1,
            apply_pe_all_layers = True,
            k_context_layers = 2,
            use_cls = bool(args.use_cls_token),
            batch_size = bs,
            patch_size = int(args.patch_size),
            use_cls_token_last_n_layers = int(args.use_cls_token_last_n_layers)
         ).cuda()
    if args.net=="cait":
        model = CaiT(
            image_size = size,
            patch_size = args.patch_size,
            num_classes = int(args.dataset_classes),
            dim = int(args.dim),
            depth = int(args.layers),   # depth of transformer for patch to patch attention only
            cls_depth=2, # depth of cross attention of CLS tokens to patch
            heads = int(args.heads),
            mlp_dim = int(args.dimh)*4, 
            dropout = 0.1,
            emb_dropout = 0.1,
            layer_dropout = 0.05
        )
    if args.net == "twins":
        model_names = timm.list_models()
        model = twins.Twins(
            num_classes= int(args.dataset_classes),
            img_size=size, patch_size=args.patch_size,
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1,
        )
    elif args.net=="swin":
        model = swin_t(window_size=args.patch_size,
                    num_classes=int(args.dataset_classes),
                    downscaling_factors=(2,2,2,1))

    net = net.cuda()
 
    pcount = count_parameters(net)
    print("count of parameters in the model = ", pcount/1e6, " million")
    
    criterion = nn.CrossEntropyLoss()  # loss function
    if args.opt == "adam":
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    elif args.opt == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=args.lr)  
    scaler = torch.amp.GradScaler('cuda',enabled=use_amp)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)
    
    if args.resume:  # load checkpoint
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/{}-{}-ckpt.t7'.format(args.net,args.patch_size),weights_only=False)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        #print(scaler)
        val_loss, acc = test(start_epoch, net, testloader, criterion, optimizer, scaler, scheduler, save_best=False)
 
    list_loss = []
    list_acc = []
    list_val_acc = []

    for epoch in range(start_epoch, args.n_epochs):
        start = time.time()
        trainloss = train(epoch, net, trainloader, criterion, scaler, optimizer, use_amp)
        val_loss, acc = test(epoch, net, testloader, criterion, optimizer, scaler, scheduler)
        list_val_acc.append(acc)
    
        scheduler.step() # step cosine scheduling
    
        list_loss.append(val_loss)
        list_acc.append(acc)

    df = pd.DataFrame(list_val_acc)
    file_name = str(args.net) + '_'+ str(args.dataset) + '_' + str(int(args.n_epochs)) + '.csv'
    if args.net == "par_enhanced":
        if bool(args.use_cls_token) == True:
            file_name = str(args.net) + '_CLS_token_used_last_' + str(int(args.use_cls_token_last_n_layers))+ '_layers_' + str(args.dataset) + '_' + str(int(args.n_epochs)) + '.csv'
        else:
            file_name = str(args.net) + '_NO_CLS_token_' + str(args.dataset) + '_' + str(int(args.n_epochs)) + '.csv'
    df.to_csv('D:/Rigel/results/' + file_name, index=True)

    args_dict = vars(args)
    with open('D:/Rigel/results/ARGS_' + file_name + '.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=args_dict.keys())
        writer.writeheader()
        writer.writerow(args_dict)
    
 # train for one epoch
def train(epoch, net, trainloader, criterion, scaler, optimizer, use_amp):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{args.n_epochs}", leave=False)
    batch_idx = 0
    for batch in progress_bar:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.amp.autocast('cuda',enabled=use_amp):
            outputs = net(inputs)  
            loss = criterion(outputs, targets) 
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        batch_idx = batch_idx + 1
        progress_bar.set_postfix(loss=train_loss/batch_idx)
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1}/{args.n_epochs} completed in {epoch_time/60:.2f} minutes")
    print(batch_idx, len(trainloader), 'Loss: %.3f | TrainAcc: %.3f%% (%d/%d)'
        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1)


# for testing model
def test(epoch, net, testloader, criterion, optimizer, scaler, scheduler, save_best=True):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print(batch_idx, len(testloader), 'Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # save checkpoint if accuracy is better than before
    acc = 100.*correct/total
    if acc > best_acc:
        if save_best:
            print('Saving..')
            state = {"net": net.state_dict(),
                  "optimizer": optimizer.state_dict(),
                  "scaler": scaler.state_dict(),
                  "best_acc": acc,
                  "epoch": epoch,
                  "scheduler": scheduler.state_dict()}
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/'+args.net+'-{}-ckpt.t7'.format(args.patch_size))
        best_acc = acc
    return test_loss, acc



if __name__ == "__main__":
    sys.exit(int(main() or 0))

