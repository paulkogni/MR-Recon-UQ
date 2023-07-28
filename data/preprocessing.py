import numpy as np 
import h5py
import torch
import meddlr.ops as oF
from os import walk
import os
from meddlr.forward import SenseModel
import nibabel as nib



def to_tensor(data: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.

    For complex arrays, the real and imaginary parts are stacked along the last
    dimension.

    Args:
        data: Input numpy array.

    Returns:
        PyTorch version of data.
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data)




def convert_k2i(kspace_slice, mask, map_slice):
    """Converts one multi-coil kspace slice into image space using the SENSE algorithm

    Args:
        kspace_slice (numpy.array): kspace slice with shape (H, W, #coils)
        mask (numpy.array): undersampling mask with shape (H_m, W_m)
        map_slice (numpy.array): sensitivity map with shape (H, w, #coils, #maps (usually 1))

    Returns:
        torch.Tensor: reconstructed undersampled image 
    """
    kspace_t = np.transpose(kspace_slice, (2,0,1))
    kspace_tensor = to_tensor(kspace_t).unsqueeze(0) # has shape (1, n_coils, H, W, 2)
    
    mask_slice = torch.as_tensor(mask).unsqueeze(0) 
    mask = oF.zero_pad(mask_slice, kspace_slice.shape[0:2]) # shape (1, H, W)

    # undersample the kspace
    us_kspace = kspace_tensor * mask.unsqueeze(0).unsqueeze(-1).type(kspace_tensor.dtype) # has shape (1, n_coils, H, W, 2)

    # apply the sense model 
    map_slice_t = to_tensor(map_slice).unsqueeze(0) # has shape (1, H, W, #coils, #maps, 2)
    A = SenseModel(map_slice_t, mask.unsqueeze(-1).unsqueeze(-1))
    image_sense = A(us_kspace.permute(0,2,3,1,4), adjoint=True)

    
    return image_sense


def compute_k2i_volume(kspace, maps, mask):
    """Convert a whole volume into an undersampled image

    Args:
        kspace: whole kspace from h5 file 
        maps: whole maps from h5 file
        mask: mask from h5 file
    """
    image_shape = [*kspace.shape[:-2]] + [2]
    recon_img = np.zeros(image_shape)
    for i in range(len(kspace)):
        kspace_slice_orig = kspace[i,:,:,0,:]
        map_slice = maps[i]
        us_mask = mask
        img_us = convert_k2i(kspace_slice_orig, us_mask, map_slice)[0,:,:,0,:]
        recon_img[i] = img_us
    return recon_img


def generate_dataset(path_to_h5s, path_to_segmentations, save_path):
    """Generate the dataset that should be used for training later

    Args:
        path_to_h5s (str): path to where the raw h5 files from the dataset are
        path_to_segmentations (str): path to where the raw segmentation files from the dataset are
        save_path (str): path to where the images should be stored
    """
    file_names = []
    for (dirpath, dirnames, filenames) in walk(path_to_h5s):
        file_names.extend(filenames)
        break
    
    for file_name in file_names:
        # define all relevant paths
        name_raw = os.path.join(path_to_h5s, file_name)
        name_segm = os.path.join(path_to_segmentations, file_name[:-3]+'.nii.gz')
        name_save = os.path.join(save_path, file_name)

        # extract necessary informations from h5 file 
        f = h5py.File(name_raw, 'r')
        kspace, us_mask, maps, target = f['kspace'], f['masks']['poisson_16.0x'], f['maps'], f['target']

        # apply undersampling and go back to image space
        img = compute_k2i_volume(kspace, maps, us_mask)
        target = to_tensor(target[:,:,:,0,0]).numpy()
        assert img.shape == target.shape
        f.close()

        # load the segmentation
        segm = nib.load(name_segm).get_fdata()


        # store everything in a new h5 file
        new_f = h5py.File(name_save, 'w')
        new_f.create_dataset('img_us', data=img)
        new_f.create_dataset('img_fs', data=target)
        new_f.create_dataset('segm', data=segm)
        new_f.close()

path_orig = '/your_path_to/SKM-TEA/skm-tea/v1-release/files_recon_calib-24'
path_segm = '/your_path_to/SKM-TEA/skm-tea/v1-release/segmentation_masks/raw-data-track'
path_save = '/your_path_to/skm-tea-16x'

generate_dataset(path_orig, path_segm, path_save)