from __future__ import division
import numpy as np
from skimage import measure
from skimage.transform import resize

def norm(vol_src):
    '''
    This function normalise the intensities of a brain volume
    
    Input: a 3D brain volume in dtype of np.float32 or np.float64
    
    Ouput: the normalised brain volume
    '''
    tissue = vol_src[vol_src>0]
    tissue = np.sort(tissue)
    cutoff = np.int(0.99*len(tissue))
    tissue = tissue[:cutoff]
    mean = tissue.mean()
    std = tissue.std()
    vol_tar = vol_src*1.0
    vol_tar[vol_src>0] -= mean
    vol_tar /= std
    return vol_tar
    
def patch_extract(map,img,r,d):
    x,y = np.where(map==1)
    x_begin = x.mean()-r/2+1
    x_begin = max(x_begin.astype("uint8"),0)
    x_end = min(x_begin+r,map.shape[1])
    x_begin = x_end - r
    y_begin = y.mean()-r/2+1
    y_begin = max(y_begin.astype("uint8"),0)
    y_end = min(y_begin+r,map.shape[0])
    y_begin = y_end - r
    patch = img[x_begin:x_begin+r,y_begin:y_begin+r]
    patch = resize(patch,(d,d),mode="constant")
    patch = patch.reshape(1,d**2)
    return patch

def muscle_input(dwi,edd_prob):
    '''
    This function prepares input to the MUSCLE Net
    
    Input: the 2D slice from a normalised DWI volume 
               and its lesion segmentation from the EDD Net
               
    Output: a list of patch sets to input to the MUSCLE Net
    '''
    patch_size_thred = 60
    patch_diameter = 16
    edd_bin = (edd_prob>0.5).astype("uint8")
    edd_label = measure.label(edd_bin, background = 0)+1
    patch_ret = []
    for i in range(1, edd_label.max()+1):
        blob = (edd_label==i).astype("uint8")
        if(blob.sum() < patch_size_thred):
            pt_0 = patch_extract(blob,dwi,patch_diameter,patch_diameter)
            pt_1 = patch_extract(blob,dwi,patch_diameter+patch_diameter/2,patch_diameter)
            pt_2 = patch_extract(blob,dwi,patch_diameter*2,patch_diameter)
            pt_edd = patch_extract(blob,edd_prob,patch_diameter,patch_diameter)
            patch_ret.append(np.vstack([pt_0,pt_1,pt_2,pt_edd]))     
    return patch_ret
