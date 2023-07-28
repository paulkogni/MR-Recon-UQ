from random import sample
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from medpy.metric import jc
import logging
import nibabel as nib

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Optional

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from meddlr.ops import complex as cplx


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.normal_(m.weight, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)
        truncated_normal_(m.bias, mean=0, std=0.001)


def init_weights_orthogonal_normal(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)


def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg


def normalise_image(image):
    '''
    make image zero mean and unit standard deviation
    '''

    img_o = np.float32(image.copy())
    m = np.mean(img_o)
    s = np.std(img_o)
    return np.divide((img_o - m), s + 1e-6)


def normalise_images(X):
    '''
    Helper for making the images zero mean and unit standard deviation i.e. `white`
    '''

    X_white = np.zeros(X.shape, dtype=np.float32)

    for ii in range(X.shape[0]):

        Xc = X[ii,...]
        X_white[ii,...] = normalise_image(Xc)

    return X_white.astype(np.float32)

def ncc(a,v, zero_norm=True):

    a = a.flatten()
    v = v.flatten()

    if zero_norm:

        a = (a - np.mean(a)) / (np.std(a) * len(a))
        v = (v - np.mean(v)) / np.std(v)

    else:

        a = (a) / (np.std(a) * len(a))
        v = (v) / np.std(v)

    return np.correlate(a, v)

def convert_nhwc_to_nchw(tensor):
    result = tensor.transpose(1, 3).transpose(2, 3)
    return result


def convert_nchw_to_nhwc(tensor):
    result = tensor.transpose(1, 3).transpose(1, 2)
    assert torch.equal(tensor, convert_nhwc_to_nchw(result))
    return result

def generate_n_samples_old(model, x, y, model_type, n_samples=100):
    """
    performs n times a forward pass and saves the returned segmentations in a list (the accumulated outputs)
    params:
        model: torch.nn.module
        x: torch Tensor with shape (n_batch,n_channels,d1,d2). The input data to the model
        y: torch Tensor with shape (n_batch,n_channels,d1,d2). The corresponding segmentation
        model_type: String, one of {phiseg, punet}
    returns: torch tensor with shape (n_samples, n_batch, n_categories, d_1, d_2)
    """
    samples = []
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
        x = x.cuda()
    with torch.no_grad():
        if model_type == 'phiseg':
            for i in range(n_samples):
                # encode
                prior_latent_space, _, _ = model.prior(x, training_prior=False)
                # decode
                s_out_list = model.likelihood(prior_latent_space)
                accumulated = model.accumulate_output(s_out_list, use_softmax=False)
                samples.append(accumulated)
                del(accumulated)
                del(s_out_list)
        else:
            model(x, y)
            # sample
            with torch.no_grad():
                for i in range(n_samples):
                    sample = model.sample(testing=True)
                    samples.append(sample)
        return torch.stack(samples)



def psnr(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = gt.max()
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)



def mse_error_map(gt, samples):
    # shape of gt: (1,1,d1,d2)
    # shape of samples: (n_samples, 1,1,d1,d2)
    gt_reshaped = gt[0,0,:,:]
    sampes_reshaped = samples[:,0,0,:,:]

    mse_errors = []
    loss_fn = torch.nn.MSELoss(reduction='none')

    with torch.no_grad():
        for i in range(len(sampes_reshaped)):
            err = loss_fn(gt_reshaped, sampes_reshaped[i])
            mse_errors.append(err.cpu().numpy())
    mse_errors = np.array(mse_errors)

    return np.mean(mse_errors, axis=0)

def eval_ssim_psnr_ncc(model, loader, n_samples, model_type):
    """
    computes mean SSIM and PSNR on a data loader
    """
    model.eval()
    psnr_list = []
    ssim_list = []
    ncc_list = []
    with torch.no_grad():
        for x, y, _, _ in loader:
            # sample here first
            if model_type == 'phiseg':
                samples = generate_n_samples_old(model, x, y, model_type, n_samples=n_samples) + model.transform_to_complex_abs(x).cuda()
            else:
                samples = generate_n_samples_old(model, x, y, model_type, n_samples=n_samples)
            
            # compute complex absolute
            samples = cplx.abs(samples.permute((0,1,3,4,2))).unsqueeze(1)
            y = cplx.abs(y.permute((0,2,3,1))).unsqueeze(1)
            x = cplx.abs(x.permute((0,2,3,1))).unsqueeze(1)
            
            # compute the mse error map
            err_map = mse_error_map(y, samples.cpu())

            # go back to cpu 
            samples = np.asarray(samples.cpu())
            x = np.asarray(x.cpu())
            # compute mean sample and variance
            print(x.shape)
            mean_sample = (np.mean(samples, axis=0) + x).reshape((x.shape[-2], x.shape[-1]))
            var = np.var(samples, axis=0).reshape((x.shape[-2], x.shape[-1]))
            # reshape gt for computations
            gt_reshaped = y.cpu().numpy().reshape((y.shape[-2], y.shape[-1]))
            # calculate ssim + append to list
            ssim_elem = structural_similarity(gt_reshaped, mean_sample, data_range=gt_reshaped.max()-gt_reshaped.min() )
            ssim_list.append(ssim_elem)

            # do the same for psnr
            psnr_elem = psnr(gt_reshaped, mean_sample)
            psnr_list.append(psnr_elem)

            # and now the ncc 
            ncc_elem = ncc(var, err_map)
            ncc_list.append(ncc_elem)


    mean_psnr = np.mean(np.asarray(psnr_list))
    mean_ssim = np.mean(np.asarray(ssim_list))
    ncc_mean = np.mean(np.asarray(ncc_list))

    return mean_psnr, mean_ssim, ncc_mean