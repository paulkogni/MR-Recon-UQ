import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import h5py
import torch.distributions as dist
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from typing import Optional
import pandas as pd

import meddlr.ops.complex as cplx

def psnr(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = gt.max()
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)


def make_prediction_on_volume(volume_us, model, n_samples):
    """Given a model, predictions for an undersampled volume is being performed

    Args:
        volume_us (numpy array): undersampled volume with shape (x,y,z,2)
        model (torch.module): the model that we want to use for prediction

    Returns:
        numpy array: samples for the reconstructed volume with shape (20,x,y,z,2)
    """
    final_shape = (n_samples,) + volume_us.shape # because we always have 20 samples
    reconstructed_volume = np.zeros(final_shape)

    z_dim = final_shape[-2] 

    for z in range(z_dim):
        slice = volume_us[:,:,z,:]
        slice = np.transpose(slice, axes=(2,0,1)) # move complex value channel to the beginning

        # convert to tensor 
        slice = torch.Tensor(slice)

        # normalize
        mean = slice.mean()
        std = slice.std()
        eps = 1e-5
        slice = ((slice - mean) / (std + eps)).unsqueeze(0) # fake batch dimension


        # perform prediction
        with torch.no_grad():
            predicted_slice = model.make_prediction(slice).squeeze().cpu() # has shape (20, 2, x, y)
        
        # add residual to slice
        if n_samples == 2:
            predicted_slice[0] = predicted_slice[0] + slice
        else:
            predicted_slice = predicted_slice + slice

        # unnormalize, go back to numpy 
        predicted_slice = (predicted_slice * std + mean).numpy()

        # get axes in correct order
        predicted_slice = np.transpose(predicted_slice, axes=(0,2,3,1))

        reconstructed_volume[:,:,:,z,:] = predicted_slice


    return reconstructed_volume

def ssim_psnr_on_h5_skmtea(pred_dir, val_dir):
    ssim_list = []
    psnr_list = []



    for tgt_file in os.listdir(pred_dir):
        one_vol_ssim = []
        one_vol_psnr = []
        with h5py.File(os.path.join(val_dir, tgt_file)) as target, h5py.File(os.path.join(pred_dir, tgt_file)) as recons:

            target = target['img_fs'][()]
            recons = recons['recon'][()]

            if recons.shape[0] == 2:
                mean_recons = recons[0]
            else:
                mean_recons = np.mean(recons, axis=0)
            
            mean_recons = cplx.abs(torch.Tensor(mean_recons)).numpy()
            target = cplx.abs(torch.Tensor(target)).numpy()

            assert mean_recons.shape == target.shape

            for i in range(target.shape[-1]):
                ssim_elem = structural_similarity(target[:,:,i], mean_recons[:,:,i], data_range=target.max())
                one_vol_ssim.append(ssim_elem)

                psnr_elem = psnr(target[:,:,i], mean_recons[:,:,i], maxval=target.max())
                one_vol_psnr.append(psnr_elem)

            ssim_list.append(np.mean(np.array(one_vol_ssim)))
            psnr_list.append(np.mean(np.array(one_vol_psnr)))

    return np.mean(np.array(ssim_list)), np.mean(np.array(psnr_list))



def eval_ssim_psnr_big(us_factors, base_data_origin, model_names, settings, base_save_path):

    # define the name for the indices in pandas dataframe 
    indis = [setting + ' ' + us_factor for setting in settings for us_factor in us_factors]

    # define the pandas dataframe where to store the results
    df = pd.DataFrame(index=indis, columns=model_names, dtype=object)

    for us_factor in us_factors:
        # create input data list
        files_origin = os.path.join(base_data_origin, us_factor)
        print(files_origin)

        for model_name in model_names:
            print('doing evaluation for', model_name)
            for setting in settings:
                print('doing evaluation for', setting)
                recon_path = os.path.join(base_save_path, us_factor[8:], model_name, setting)
                # recon_path = os.path.join(base_save_path, '4x', model_name, setting)

                ssim, psnr = ssim_psnr_on_h5_skmtea(recon_path, files_origin)

                df[model_name][setting + ' ' + us_factor] = (ssim, psnr)
    
    print(df)


def mse_error_map(gt, samples):
    """Computes the MSE error map between a 3D ground truth slice and reconstructed samples

    Args:
        gt (torch.Tensor): GT slice with shape (x,y)
        samples (torch.Tensor): samples with shape (n_samples,x,y)

    Returns:
        np.array: the pixel-wise error map between the GT and samples
    """
    # shape of gt: (x,y)
    # shape of samples: (n_samples,x,y)

    loss_fn = torch.nn.MSELoss(reduction='none')
    mse_errors = []
    # loop over samples and 
    with torch.no_grad():
        for i in range(len(samples)):
            err = loss_fn(gt, samples[i])
            mse_errors.append(err.numpy())
    
    mse_errors = np.array(mse_errors)
    return np.mean(mse_errors, axis=0)

def ncc(a,v, zero_norm=True):
    """Computes the normalized cross correaltion between two arrays

    Args:
        a (np.array): first array
        v (np.array): second array

    Returns:
        float: the normalized cross correlation between arrays a and v
    """
    a = a.flatten()
    v = v.flatten()
    eps = 1e-15
    if zero_norm:

        a = (a - np.mean(a)) / (np.std(a) * len(a) + eps)
        v = (v - np.mean(v)) / (np.std(v) + eps)

    else:

        a = (a) / (np.std(a) * len(a) + eps)
        v = (v) / (np.std(v) + eps)

    return np.correlate(a, v)


def ncc_on_h5_skmtea(pred_dir, val_dir):
    ncc_list = []

    for tgt_file in os.listdir(pred_dir):
        one_vol_ncc = []
        with h5py.File(os.path.join(val_dir, tgt_file)) as target, h5py.File(os.path.join(pred_dir, tgt_file)) as recons:

            target = target['img_fs'][()]
            recons = recons['recon'][()]

            if recons.shape[0] == 2:
                var_recons = recons[1]**2
                recons = recons[0:1,...]
            else:
                var_recons = np.var(recons, axis=0)
            
            var_recons = cplx.abs(torch.Tensor(var_recons)).numpy() # has shape (x,y,z)
            target = cplx.abs(torch.Tensor(target)) # has shape (x,y,z)
            recons = cplx.abs(torch.Tensor(recons)) # has shape (n_samples,x,y,z)

            for i in range(target.shape[-1]):
                err_map = mse_error_map(target[:,:,i], recons[:,:,:,i])
                assert err_map.shape == var_recons[:,:,i].shape
                ncc_val = ncc(var_recons[:,:,i], err_map)
                one_vol_ncc.append(ncc_val)


            ncc_list.append(np.mean(np.array(one_vol_ncc)))

    return np.mean(np.array(ncc_list))


def eval_ncc_big(us_factors, base_data_origin, model_names, settings, base_save_path):

    # define the name for the indices in pandas dataframe 
    indis = [setting + ' ' + us_factor for setting in settings for us_factor in us_factors]

    # define the pandas dataframe where to store the results
    df = pd.DataFrame(index=indis, columns=model_names, dtype=object)

    for us_factor in us_factors:
        # create input data list
        files_origin = os.path.join(base_data_origin, us_factor)
        print(files_origin)

        for model_name in model_names:
            print('doing evaluation for', model_name)
            for setting in settings:
                print('doing evaluation for', setting)
                recon_path = os.path.join(base_save_path, us_factor[8:], model_name, setting)
                # recon_path = os.path.join(base_save_path, '4x', model_name, setting)

                ncc_val = ncc_on_h5_skmtea(recon_path, files_origin)

                df[model_name][setting + ' ' + us_factor] = ncc_val
    
    print(df)



def var_on_h5_skmtea(pred_dir):
    var_list = []

    for tgt_file in os.listdir(pred_dir):
        with h5py.File(os.path.join(pred_dir, tgt_file)) as recons:

            recons = recons['recon'][()]
            print('recon shape', recons.shape)
            if recons.shape[0] == 2:
                var_recons = recons[1]**2
                recons = recons[0:1,...]
            else:
                var_recons = np.var(recons, axis=0)
            
            print('variance shape', var_recons.shape)
            mean_var_recons_one_vol = np.mean(cplx.abs(torch.Tensor(var_recons)).numpy())


            var_list.append(mean_var_recons_one_vol)

    return np.mean(var_list)


def eval_var_big(us_factors, model_names, settings, base_save_path):

    # define the name for the indices in pandas dataframe 
    indis = [setting + ' ' + us_factor for setting in settings for us_factor in us_factors]

    # define the pandas dataframe where to store the results
    df = pd.DataFrame(index=indis, columns=model_names, dtype=object)

    for us_factor in us_factors:

        for model_name in model_names:
            print('doing evaluation for', model_name)
            for setting in settings:
                print('doing evaluation for', setting)
                recon_path = os.path.join(base_save_path, us_factor[8:], model_name, setting)
                # recon_path = os.path.join(base_save_path, '4x', model_name, setting)

                ncc_val = var_on_h5_skmtea(recon_path)

                df[model_name][setting + ' ' + us_factor] = ncc_val
    
    print(df)



# Segmentation stuff here
def make_prediction_on_volume_segmentation(volume_fs, model):
    """Given a model, predictions for an undersampled volume is being performed

    Args:
        volume_fs (numpy array): undersampled volume with shape (n_samples, x,y,z,2)
        model (torch.module): the model that we want to use for prediction

    Returns:
        numpy array: segmentations for the sampled reconstructions with shape (20,x,y,z)
    """

    # make exception for heteroscedastic model
    if volume_fs.shape[0] == 2:
        final_shape = (1,) + volume_fs.shape[1:-1] 
    else:
        final_shape = volume_fs.shape[:-1] 
    segmentation_volume = np.zeros(final_shape)

    z_dim = final_shape[-1] 

    if torch.cuda.is_available():
        model.cuda()

    for i in range(len(segmentation_volume)):
        for z in range(z_dim):
            slice = volume_fs[i,:,:,z,:]
            slice = np.transpose(slice, axes=(2,0,1)) # move complex value channel to the beginning

            # convert to tensor 
            slice = torch.Tensor(slice)

            # normalize
            mean = slice.mean()
            std = slice.std()
            eps = 1e-5
            slice = ((slice - mean) / (std + eps)).unsqueeze(0) # fake batch dimension
            if torch.cuda.is_available():
                slice = slice.cuda()


            # perform prediction
            with torch.no_grad():
                predicted_slice = model.make_prediction(slice).squeeze().cpu() # has shape (x, y)
            
            
            segmentation_volume[i,:,:,z] = predicted_slice
            
        if segmentation_volume.shape[0] == 2:
            break


    return segmentation_volume

import torch.nn.functional as F

def ce_error_map_samples(gt_annot):
    """
    computes error map between mean_gt annotation and individual segmentations
    params:
        gt_annot: torch tensor in shape (n_annotations, n_channels, d1, d2)
    returns: numpy array with shape (n_annotations, n_channels, d1, d2). Contains the cross entropy errors between mean gt annotation and individual gt annotations
    """

    cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
    
    gt_annot_onehot = F.one_hot(torch.Tensor(gt_annot).long(), num_classes=7).permute((0,3,1,2))

    mean_annot = torch.mean(gt_annot_onehot.float(), dim=0).unsqueeze(0)#.softmax(dim=0).unsqueeze(0)
    

    ce_errors = []
    for i in range(len(gt_annot)):
        ce_loss = cross_entropy(input=mean_annot, target=torch.Tensor(gt_annot[i]).unsqueeze(0).long()).squeeze().numpy()
        ce_errors.append(ce_loss)
    return np.mean(np.array(ce_errors), axis=0)


def ce_error_map_gts(gt, samples):
    """
    computes error map between mean_gt annotation and individual segmentations
    params:
        gt_annot: torch tensor in shape (n_annotations, n_channels, d1, d2)
    returns: numpy array with shape (n_annotations, n_channels, d1, d2). Contains the cross entropy errors between mean gt annotation and individual gt annotations
    """

    cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
    
    annot_samples = F.one_hot(torch.Tensor(samples).long(), num_classes=7).permute((0,3,1,2))

    mean_annot = torch.mean(annot_samples.float(), dim=0).unsqueeze(0)#.softmax(dim=0).unsqueeze(0)
    


    ce_loss = cross_entropy(input=mean_annot, target=torch.Tensor(gt).unsqueeze(0).long()).squeeze().numpy()

    return np.array(ce_loss)

def ncc_on_h5_skmtea_segm(pred_dir, val_dir):
    ncc_list = []

    for tgt_file in os.listdir(pred_dir):
        one_vol_ncc = []
        with h5py.File(os.path.join(val_dir, tgt_file)) as target, h5py.File(os.path.join(pred_dir, tgt_file)) as recons:

            target = target['segm'][()] # has shape (x,y,z)
            recons = recons['segm'][()] # has shape (x,y,z)
            

            for i in range(target.shape[-1]):
                err_map_pred_gt = ce_error_map_gts(target[:,:,i], recons[:,:,:,i])
                err_map_preds = ce_error_map_samples(recons[:,:,:,i])
                assert err_map_pred_gt.shape == err_map_preds.shape
                ncc_val = ncc(err_map_pred_gt, err_map_preds)
                one_vol_ncc.append(ncc_val)


            ncc_list.append(np.mean(np.array(one_vol_ncc)))

    return np.mean(np.array(ncc_list))

def eval_ncc_big_segmentations(us_factors, base_data_origin, model_names, settings, base_save_path):

    # define the name for the indices in pandas dataframe 
    indis = [setting + ' ' + us_factor for setting in settings for us_factor in us_factors]

    # define the pandas dataframe where to store the results
    df = pd.DataFrame(index=indis, columns=model_names, dtype=object)

    for us_factor in us_factors:
        # create input data list
        # files_origin = os.path.join(base_data_origin, us_factor)
        files_origin = os.path.join(base_data_origin, 'skm-tea-4x')
        print(files_origin)

        for model_name in model_names:
            print('doing evaluation for', model_name)
            for setting in settings:
                print('doing evaluation for', setting)
                recon_path = os.path.join(base_save_path, us_factor[8:], model_name, setting)
                # recon_path = os.path.join(base_save_path, '4x', model_name, setting)

                # ncc_val = ncc_on_h5_skmtea(recon_path, files_origin)
                ncc_val = ncc_on_h5_skmtea_segm(recon_path, files_origin)

                df[model_name][setting + ' ' + us_factor] = ncc_val
    
    print(df)
    return df