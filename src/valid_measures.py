import pathlib
import sys
from collections import defaultdict
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import KneeDataDev
from architecture import networkwithunet
import h5py
from tqdm import tqdm
from data import transforms as T
from evaluate import Metrics,hfn,mse,nmse,psnr,ssim
import random
import pandas as pd 
METRIC_FUNCS = dict(
    MSE=mse,
    NMSE=nmse,
    PSNR=psnr,
    SSIM=ssim,
    HFN=hfn
)

def metrics_reconstructions(predictions,targets,metrics_info,acc_factor,num_coils):
    #out_dir.mkdir(exist_ok=True)
    metrics = Metrics(METRIC_FUNCS)
    #print("recons items: ", reconstructions.items())
    #print("recons items: ", reconstructions.keys())
    #for fname, [recons,target] in reconstructions.items():
    for fname in predictions.keys():
        recons = predictions[fname]
        target = targets[fname]
        recons = np.transpose(recons,[1,2,0])
        print("recons: ",recons.shape, " target:", target.shape)
        target = np.transpose(target,[1,2,0])
        if len(target.shape) == 2: 
            target = np.expand_dims(target,2) 
        no_slices = target.shape[-1]

        for index in range(no_slices):
            print(acc_factor, num_coils, fname, index)
            target_slice = target[:,:,index]
            recons_slice = recons[:,:,index]
            print("recons slice: ",recons_slice.shape, " target slice:", target_slice.shape)
            mse_slice  = round(mse(target_slice,recons_slice),5)
            nmse_slice = round(nmse(target_slice,recons_slice),5)
            psnr_slice = round(psnr(target_slice,recons_slice),2)
            ssim_slice = round(ssim(target_slice,recons_slice),4)

            metrics_info['MSE'].append(mse_slice)
            metrics_info['NMSE'].append(nmse_slice)
            metrics_info['PSNR'].append(psnr_slice)
            metrics_info['SSIM'].append(ssim_slice)
            metrics_info['VOLUME'].append(fname)
            metrics_info['SLICE'].append(index)
 
        #print (recons.shape,target.shape)
        #print (recons)
        #break
        metrics.push(target, recons)
    print ('end of metrics_reconstructions')
    return metrics, metrics_info 


def save_reconstructions(reconstructions, out_dir):
    """
    Saves the reconstructions from a model into h5 files that is appropriate for submission
    to the leaderboard.
    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
            corresponding reconstructions (of shape num_slices x height x width).
        out_dir (pathlib.Path): Path to the output directory where the reconstructions
            should be saved.
    """
    out_dir.mkdir(exist_ok=True)
    for fname, recons in reconstructions.items():
        with h5py.File(out_dir / fname, 'w') as f:
            f.create_dataset('reconstruction', data=recons)

def create_data_loaders(args):

    data = KneeDataDev(args.data_path,args.num_coils,args.acceleration_factor,args.dataset_type)
    data_loader = DataLoader(
        dataset=data,
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=True,)

    return data_loader


def load_model(checkpoint_file):

    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']

    dccoeff = 0.1
    wacoeff = 0.1
    cascade = 3

    model = networkwithunet(dccoeff,wacoeff,cascade).to(args.device)
    model.load_state_dict(checkpoint['model'])

    return model


def run_unet(args, model, data_loader):

    model.eval()
    reconstructions = defaultdict(list)
    with torch.no_grad():
        for (iter,data) in enumerate(tqdm(data_loader)):

            img_gt,img_und,rawdata_und,masks,sensitivity,fnames,gamma_input,coilmask = data

            #img_gt  = img_gt.to(args.device)
            img_und = img_und.to(args.device)
            rawdata_und = rawdata_und.to(args.device)
            masks = masks.to(args.device)
            sensitivity = sensitivity.to(args.device)
            coilmask = coilmask.to(args.device)
            gamma_input = gamma_input.to(args.device).float()
            
            output = model(img_und,rawdata_und,masks,sensitivity,gamma_input,coilmask)
            recons = T.complex_abs(output).to('cpu')
            target = T.complex_abs(img_gt)
            
            for i in range(recons.shape[0]):
                reconstructions[fnames[i]].append((recons[i].numpy(),target[i].numpy()))

        predictions = {
            fname: np.stack([pred for pred,_ in sorted(slice_preds)])
            for fname, slice_preds in reconstructions.items()
        }

        targets = {
            fname: np.stack([targ for _,targ in sorted(slice_preds)])
            for fname, slice_preds in reconstructions.items()
        }
             
    return predictions,targets


def main(args):
    
    data_loader = create_data_loaders(args)
    model = load_model(args.checkpoint)
    predictions, targets = run_unet(args, model, data_loader)
    #save_reconstructions(reconstructions, args.out_dir) # uncomment to save the prediction
    recons_key = 'volfs'

    metrics_info = {'VOLUME':[],'SLICE':[],'MSE':[],'NMSE':[],'PSNR':[],'SSIM':[]}

    metrics, metrics_info = metrics_reconstructions(predictions,targets,metrics_info, args.acceleration_factor, args.num_coils)
    metrics_report = metrics.get_report()
    with open(args.report_path / 'report_{}_{}_{}coils.txt'.format(args.dataset_type,args.acceleration_factor,args.num_coils),'w') as f:
        f.write(metrics_report)
    csv_path     = args.report_path / 'metrics_{}_{}_{}coils.csv'.format(args.dataset_type,args.acceleration_factor,args.num_coils)
    df = pd.DataFrame(metrics_info)
    df.to_csv(csv_path)



def create_arg_parser():

    parser = argparse.ArgumentParser(description="Valid setup for MR recon U-Net")
    parser.add_argument('--seed',default=42,type=int,help='Seed for random number generators')
    parser.add_argument('--checkpoint', type=pathlib.Path, required=True,
                        help='Path to the U-Net model')
    parser.add_argument('--out-dir', type=pathlib.Path, required=True,
                        help='Path to save the reconstructions to')
    parser.add_argument('--batch-size', default=16, type=int, help='Mini-batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Which device to run on')
    parser.add_argument('--data-path',type=str,help='path to validation dataset')
    parser.add_argument('--acceleration_factor',type=str,help='acceleration factors')
    parser.add_argument('--dataset_type',type=str,help='cardiac,kirby')
    parser.add_argument('--num_coils',type=str,help='number of coils')
    parser.add_argument('--report-path', type=pathlib.Path, required=True,
                        help='Path to save metrics')

    return parser

if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
 
    main(args)
