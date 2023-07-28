""" Full assembly of the parts to form the complete network """
import os,sys,inspect
import torch.nn.functional as F
import numpy as np

import torch
from models.unet.src.unet_parts import *
import torch.nn as nn
import pdb
import models.unet.src.utils as utils
import wandb
import matplotlib.pyplot as plt
from meddlr.ops import complex as cplx

class UNet(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels_in = n_channels_in
        self.n_channels_middle = 32
        self.n_channels_out = n_channels_out
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        # path 1
        self.inc = DoubleConv(n_channels_in, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        # joined path
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.out = OutConv(64, self.n_channels_out)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)

        return x 

    def loss(self,y_hat, y):
        criterion = torch.nn.MSELoss(reduction='none')
        
        y_flat = y.reshape((y.shape[0], -1))
        y_hat_flat = y_hat.reshape((y.shape[0], -1))
        recon_loss = criterion(input=y_hat_flat, target=y_flat).sum(dim=1)


        return torch.mean(recon_loss)
    
    def make_prediction(self, img):
        """Performs a prediction given an undersampled image

        Args:
            img (torch.Tensor): The undersampled image

        Returns:
            torch.Tensor: The reconstruction estimation
        """
        out = self.forward(img)

        return out.unsqueeze(0)


def compute_train_loss_and_train(train_loader, model, optimizer, use_gpu, epoch):

    model.train()

    running_loss = 0.0

    for x,y,_,_ in train_loader:
        if use_gpu:
            x = x.cuda()
            y = y.cuda()
        
        # compute forward pass
        output = model(x)

        loss = model.loss(output + x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss * train_loader.batch_size
        torch.cuda.empty_cache()
    

    epoch_loss = running_loss / len(train_loader.dataset)

    return epoch_loss



def compute_eval_loss(test_loader, model, use_gpu, epoch):
    """
    computes the evaluation epoch loss on the evaluation set
    """
    model.eval()

    running_loss = 0.0
    with torch.no_grad():
        for x,y,_,_ in test_loader:
            if use_gpu:
                x = x.cuda()
                y = y.cuda()
            
            # compute forward pass
            output = model(x)

            loss = model.loss(output + x, y)


            running_loss += loss * test_loader.batch_size
    torch.cuda.empty_cache()
    
    epoch_loss = running_loss / len(test_loader.dataset)
    return epoch_loss


def train_model(model, train_loader, eval_loader, optim, epochs=1, save_model=None, save_path=None, continue_training_path=None, eval_metric=None):
    end_epoch = 0
    use_gpu = torch.cuda.is_available()

    if continue_training_path:
        checkpoint = torch.load(continue_training_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if use_gpu:
            model.cuda()
        optim = torch.optim.Adam(model.parameters(), lr=0.0001)
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        end_epoch = checkpoint['epoch']
    if use_gpu:
        model.cuda()
    
    # define current best losses 
    best_total_eval_loss = np.inf
    best_ssim = -np.inf



    for epoch in range(end_epoch, epochs):
        print('Epoch:', epoch)

        # train the model
        train_running_loss = compute_train_loss_and_train(train_loader, model, optim, use_gpu, epoch=epoch)

        # compute evaluation loss
        eval_running_loss = compute_eval_loss(eval_loader, model, use_gpu, epoch)

        if eval_metric:
            if epoch % 50 == 0: # compute only every 50 epochs
                psnr, ssim = utils.eval_ssim_psnr(model, eval_loader)
                print('psnr:',psnr)
                print('ssim:',ssim)
        
        if save_model==True:
            if eval_running_loss < best_total_eval_loss:
                best_total_eval_loss = eval_running_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'loss': train_running_loss,
                    }, f'{save_path}unet_best_eval_epoch{epoch}.pth')
                print('saving best eval model')
            if eval_metric:
                if epoch % 10 == 0:
                    if ssim > best_ssim:
                        best_ssim = ssim
                        torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optim.state_dict(),
                        'loss': train_running_loss,
                        }, f'{save_path}unet_best_ssim_epoch{epoch}.pth')
                        print('saving best GED model')
            
        print('training loss:', train_running_loss)
        print('evaluation loss:', eval_running_loss)

    return