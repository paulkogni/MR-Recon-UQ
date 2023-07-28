""" Full assembly of the parts to form the complete network """
import os,sys,inspect
# sys.path.insert(1, os.path.join(sys.path[0], '../../../'))
import torch.nn.functional as F
import numpy as np

import torch
from models.unet_het.src.unet_parts import *
import torch.nn as nn
import pdb
import models.unet_het.src.utils as utils
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
        self.dropout = nn.Dropout(p=0.2)

        # path 1
        self.inc = DoubleConv(n_channels_in, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        # joined path
        self.up1_mu = Up(1024, 512 // factor, bilinear)
        self.up2_mu = Up(512, 256 // factor, bilinear)
        self.up3_mu = Up(256, 128 // factor, bilinear)
        self.up4_mu = Up(128, 64, bilinear)
        self.out_mu = OutConv(64, self.n_channels_out)

        self.up1_logvar = Up(1024, 512 // factor, bilinear)
        self.up2_logvar = Up(512, 256 // factor, bilinear)
        self.up3_logvar = Up(256, 128 // factor, bilinear)
        self.up4_logvar = Up(128, 64, bilinear)
        self.out_logvar = OutConv(64, self.n_channels_out)

    def forward(self, x):

        # encode path (shared)
        x1 = self.inc(self.dropout(x))
        x2 = self.down1(self.dropout(x1))
        x3 = self.down2(self.dropout(x2))
        x4 = self.down3(self.dropout(x3))
        x5 = self.down4(self.dropout(x4))

        # decode mu 
        mu = self.up1_mu(self.dropout(x5), x4)
        mu = self.up2_mu(self.dropout(mu), x3)
        mu = self.up3_mu(self.dropout(mu), x2)
        mu = self.up4_mu(mu, x1)
        mu = self.out_mu(mu)

        # decode logvar
        logvar = self.up1_logvar(self.dropout(x5), x4)
        logvar = self.up2_logvar(self.dropout(logvar), x3)
        logvar = self.up3_logvar(self.dropout(logvar), x2)
        logvar = self.up4_logvar(logvar, x1)
        logvar = self.out_logvar(logvar)

        return mu, logvar

    def loss(self,mu, logvar, y):
        # compute the NLL criterion according to https://github.com/mlaves/well-calibrated-regression-uncertainty/blob/master/utils.py
        
        mu_flat = mu.reshape((mu.shape[0], -1))
        logvar_flat = logvar.reshape((logvar.shape[0], -1))
        y_flat = y.reshape((y.shape[0], -1))
        
        loss = (torch.exp(-logvar_flat) * torch.pow(y_flat-mu_flat, 2) + logvar_flat).sum(dim=1)


        return torch.mean(loss)

    def enable_dropout(self):
        """ Function to enable the dropout layers during test-time """
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def make_prediction(self, img):
        """Performs a prediction to obtain a reconstruction

        Args:
            img (torch.Tensor): The undersampled image to reconstruct with shape (n_batch, n_channel, width, height)

        Returns:
            torch.Tensor: The reconstruction from multiple samples with same shape as input 
        """
        if torch.cuda.is_available():
            self.cuda()
            img = img.cuda()

        self.enable_dropout()
        mus = []
        vars = []
        
        with torch.no_grad():
            for i in range(20):
                mu, logvar = self.forward(img)
                var = torch.exp(logvar)
                mus.append(mu)
                vars.append(var)
        
        mus = torch.stack(mus)
        vars = torch.stack(vars)

        # compute means
        mean_mu = mus.mean(dim=0)
        mean_var = vars.mean(dim=0)

        # compute stuff for predictive variance
        square_of_mean = mean_mu**2
        mean_of_squares = (mus**2).mean(dim=0)

        # compute predictive variance
        pred_var = torch.sqrt(mean_var + mean_of_squares - square_of_mean)

        samples = []
        
        with torch.no_grad():
            for i in range(20):
                output = torch.distributions.Normal(mean_mu, pred_var).sample()
                samples.append(output)

        return torch.stack(samples)



def compute_train_loss_and_train(train_loader, model, optimizer, use_gpu, epoch):

    model.train()

    running_loss = 0.0

    for x,y,_,_ in train_loader:
        if use_gpu:
            x = x.cuda()
            y = y.cuda()
        
        # compute forward pass
        mu, logvar = model(x)

        loss = model.loss(mu + x, logvar, y)

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
            mu, logvar = model(x)

            loss = model.loss(mu + x, logvar, y)


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
                psnr, ssim, _ = utils.eval_ssim_psnr(model, eval_loader)
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
                    }, f'{save_path}unet_het_dr_best_eval_epoch{epoch}.pth')
                print('saving best eval model')
            if eval_metric:
                if epoch % 50 == 0:
                    if ssim > best_ssim:
                        best_ssim = ssim
                        torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optim.state_dict(),
                        'loss': train_running_loss,
                        }, f'{save_path}unet_het_dr_best_ssim_epoch{epoch}.pth')
                        print('saving best GED model')
            
        print('training loss:', train_running_loss)
        print('evaluation loss:', eval_running_loss)

    return