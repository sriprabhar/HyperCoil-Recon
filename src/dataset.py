import pathlib
import random
import numpy as np
import h5py
from torch.utils.data import Dataset
import torch
import os 
from data import transforms as T

def combine_all_coils(image, sensitivity):
    """return sensitivity combined images from all coils"""
    combined = T.complex_multiply(sensitivity[...,0],
                                  -sensitivity[...,1],
                                  image[...,0],
                                  image[...,1])

    return combined.sum(dim = 0)



class KneeData(Dataset):

    def __init__(self, root, acc_factors,dataset_types,mask_types,num_coils,train_or_valid): 

        #files = list(pathlib.Path(root).iterdir())
        self.examples = []
        self.TotNumCoils = 15
        for ncoil in num_coils:
            for dataset_type in dataset_types:
                dataroot = os.path.join(root, dataset_type)
                for mask_type in mask_types:
                    newroot = os.path.join(dataroot, mask_type)
                    for acc_factor in acc_factors:
                        files = list(pathlib.Path(os.path.join(newroot,'acc_{}'.format(acc_factor),train_or_valid)).iterdir())
                        for fname in sorted(files):
                            self.examples.append([fname,ncoil,acc_factor,dataset_type])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        
        fname, ncoils, acc_factor,dataset_type = self.examples[i] 
        arr = np.zeros(self.TotNumCoils)
        ncoils = int(ncoils)
        acc_val = float(acc_factor[:-1].replace("_","."))
        arr[:ncoils]  = 1   
        np.random.shuffle(arr) # create random coil switches in which ncoils are binary 1
        #print("arr: ",arr)
        
        dataset_val = 1 if dataset_type=='anatomy_type_1' else 2 #anatomy_type_1 can be knee PD / PDFS or brain T1, T2 etc, that is dataset folder name
        arr_acc = np.multiply(arr,dataset_val)
        coilmasklist=[]
        
        with h5py.File(fname, 'r') as data:

            img_gt    = torch.from_numpy(data['img_gt'][:])
            img_und   = torch.from_numpy(data['img_und'][:])
            img_und_kspace = torch.from_numpy(data['img_und_kspace'][:])
            rawdata_und = torch.from_numpy(data['rawdata_und'][:])
            masks = torch.from_numpy(data['masks'][:])
            sensitivity = torch.from_numpy(data['sensitivity'][:])         
            #print(sensitivity.shape)   
            for i in range(sensitivity.shape[0]):
                coilmaskmtx=torch.full((sensitivity.shape[1],sensitivity.shape[2]),arr[i])
                coilmasklist.append(coilmaskmtx)
            coilmask = torch.stack(coilmasklist)
            coilmask1 = coilmask.unsqueeze(-1)
            kspace_und_mch_coilmasked = rawdata_und * coilmask1
            img_und_mch_coilmasked = T.ifft2(kspace_und_mch_coilmasked)
            img_und_mch_coilmasked = img_und_mch_coilmasked * coilmask1
            sensitivity_coilmasked = sensitivity * coilmask1
            img_und = combine_all_coils(img_und_mch_coilmasked, sensitivity_coilmasked)
            #print("img_gt: ",img_gt.shape, "img_und: ",img_und.shape)
            #print("arr_acc: ",arr_acc.shape)
            return img_gt,img_und,kspace_und_mch_coilmasked,masks,sensitivity,torch.from_numpy(arr_acc), coilmask


class KneeDataDev(Dataset):

    def __init__(self, root,num_coils,acc_factor,dataset_type):
       
        files = list(pathlib.Path(root).iterdir())
        self.examples = []
        self.TotNumCoils = 15
        for fname in sorted(files):
            self.examples.append([fname,num_coils,acc_factor,dataset_type])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        
        fname, ncoils,acc_factor,dataset_type = self.examples[i]
        arr = np.zeros(self.TotNumCoils)
        ncoils = int(ncoils)
        acc_val = float(acc_factor[:-1].replace("_","."))
        arr[:ncoils]  = 1
        np.random.shuffle(arr)
        dataset_val = 1 if dataset_type=='anatomy_type_1' else 2
        arr_acc = np.multiply(arr,dataset_val)
        coilmasklist=[]
    
        with h5py.File(fname, 'r') as data:

            img_gt    = torch.from_numpy(data['img_gt'][:])
            img_und   = torch.from_numpy(data['img_und'][:])
            img_und_kspace = torch.from_numpy(data['img_und_kspace'][:])
            rawdata_und = torch.from_numpy(data['rawdata_und'][:])
            masks = torch.from_numpy(data['masks'][:])
            sensitivity = torch.from_numpy(data['sensitivity'][:])
            for i in range(sensitivity.shape[0]):
                coilmaskmtx=torch.full((sensitivity.shape[1],sensitivity.shape[2]),arr[i])
                coilmasklist.append(coilmaskmtx)
            coilmask = torch.stack(coilmasklist)
            coilmask1 = coilmask.unsqueeze(-1)
            # find the img_und and img_und_kspace here and then return
            #print(coilmask1.shape) #torch.Size([15, 640, 368,1])
            kspace_und_mch_coilmasked = rawdata_und * coilmask1
            img_und_mch_coilmasked = T.ifft2(kspace_und_mch_coilmasked)
            img_und_mch_coilmasked = img_und_mch_coilmasked * coilmask1
            sensitivity_coilmasked = sensitivity * coilmask1
            img_und = combine_all_coils(img_und_mch_coilmasked, sensitivity_coilmasked)
            #print("img_gt: ",img_gt.shape, "img_und: ",img_und.shape)
       
        return  img_gt,img_und,rawdata_und,masks,sensitivity,str(fname.name),torch.from_numpy(arr_acc),coilmask



