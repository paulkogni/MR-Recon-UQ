#Useful data utils
import numpy as np
import pickle
from os.path import join
import pdb
from tqdm import tqdm
import torch
import os 
import h5py


from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from typing import Optional
from meddlr.ops import complex as cplx



def normalize(x, type, per_pixel, input_output):
    if type == 'standard':
        # code
        if per_pixel:
            mean_val = x.mean(dim=0)[:, None, :, :]
            std_val = x.std(dim=0)[:, None, :, :]
        else:
            mean_val = x.mean()
            std_val = x.std()
        params = {'mean_'+input_output: mean_val, 'std_'+input_output: std_val}
        x = (x - mean_val) / std_val

    elif type == 'min-max':
        if per_pixel:
            max_val = x.max(dim=0)[0][:, None, :, :]
            min_val = x.min(dim=0)[0][:, None, :, :]
        else:
            max_val = x.max()
            min_val = x.min()
        params = {'max_'+input_output: max_val, 'min_'+input_output: min_val}
        x = (x - min_val) / (max_val - min_val)

    else:
        raise NotImplementedError

    return x, params

def normalize_dataset(dataset):
  param_path = join(dataset.cache_path, 'norm_params.pickle')
  try:
    with open(param_path, 'rb') as handle:
      dataset.norm_params= pickle.load(handle)
    print('normalized with parameters from cache')
  except:
    print('Computing normalization parameters')
    running_max_in = dataset[0][0].max()
    running_min_in = dataset[0][0].min()
    running_max_out = dataset[0][1].max()
    running_min_out = dataset[0][1].min()
    stat_in = RunningStats()
    stat_out = RunningStats()
    for data_point in tqdm(dataset):
      if data_point[0].max() >= running_max_in:
        running_max_in = data_point[0].max()
      if data_point[1].max() >= running_max_out:
        running_max_out = data_point[1].max()
      if data_point[0].min() <= running_min_in:
        running_min_in = data_point[0].min()    
      if data_point[1].min() <= running_min_out:
        running_min_out = data_point[1].min()

      stat_in.push(data_point[0])
      stat_out.push(data_point[1])
         
    dataset.norm_params = {'input_max': running_max_in.item(), 'input_min': running_min_in.item(), 'input_mean': stat_in.mean().item(), 
              'input_std': np.sqrt(stat_in.variance().mean().item()), 'output_max': running_max_out.item(), 'output_min': running_min_out.item(), 
              'output_mean': stat_out.mean().item(), 'output_std': np.sqrt(stat_out.variance().mean()).item()}

    with open(param_path, 'wb') as handle:
      pickle.dump(dataset.norm_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

  return dataset

class RunningStats:

    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

    def clear(self):
        self.n = 0
    
    def push(self, x):
        self.n += 1
    
        if self.n == 1:
            self.old_m = self.new_m = x.mean()
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x.mean() - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)
        
            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0
    
    def standard_deviation(self):
        return np.sqrt(self.variance())


def psnr(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = gt.max()
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)


def mse_error_map(gt, mu):
    # shape of gt: (1,1,d1,d2)
    # shape of samples: (n_samples, 1,1,d1,d2)
    gt_reshaped = gt[0,0,:,:]
    mu_reshaped = mu[0,0,:,:]

    loss_fn = torch.nn.MSELoss(reduction='none')

    mse = loss_fn(gt_reshaped, mu_reshaped)

    return mse.cpu().numpy()


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


def eval_ssim_psnr(model, loader):
    """
    computes mean SSIM and PSNR on a data loader
    """
    model.eval()
    psnr_list = []
    ssim_list = []
    ncc_list = []
    with torch.no_grad():
        for x, y, _, _ in loader:
            
            mu, logvar = model(x.cuda())
            mu = mu.cpu() + x.cpu()

            # compute the complex absolute
            y = cplx.abs(y.permute((0,2,3,1))).unsqueeze(1)
            mu = cplx.abs(mu.permute((0,2,3,1))).unsqueeze(1)
            logvar = cplx.abs(logvar.permute((0,2,3,1))).unsqueeze(1)

            # compute the variance
            var = torch.exp(logvar).reshape((x.shape[-2], x.shape[-1])).cpu().numpy()

            # compute the mse error map
            err_map = mse_error_map(y.cuda(), mu.cuda())
            
            gt_reshaped = y.cpu().numpy().reshape((y.shape[-2], y.shape[-1]))
            mu = mu.cpu().numpy().reshape((y.shape[-2], y.shape[-1]))
            # calculate ssim + append to list
            ssim_elem = structural_similarity(gt_reshaped, mu, data_range=gt_reshaped.max()-gt_reshaped.min())
            ssim_list.append(ssim_elem)

            # do the same for psnr
            
            psnr_elem = psnr(gt_reshaped, mu)
            psnr_list.append(psnr_elem)

            # compute the ncc
            ncc_elem = ncc(var, err_map)
            ncc_list.append(ncc_elem)


    mean_psnr = np.mean(np.asarray(psnr_list))
    mean_ssim = np.mean(np.asarray(ssim_list))
    mean_ncc = np.mean(np.asarray(ncc_list))

    return mean_psnr, mean_ssim, mean_ncc
