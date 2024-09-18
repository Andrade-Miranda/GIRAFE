
#Setup Imports
import os
import json
import argparse
from PIL import Image
from monai import transforms


import matplotlib.pyplot as plt
import numpy as np

from monai.inferers import SimpleInferer
from monai.transforms import (
    AsDiscrete,
    Activations,
    EnsureChannelFirstd,
    Compose
)

from monai.metrics import DiceMetric
from monai import data
from monai.data import decollate_batch
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.data import PILReader

import torch
from utils.swinv2Unet import SwinV2OneDecoder

#############################################################################
def restore_Model(file,model,device):
    checkpoint = torch.load(file,map_location=device,weights_only=True)
    if 'state_dict' in checkpoint:
        model.load_state_dict(
            checkpoint['state_dict'],strict=False)
    else:
        model.load_state_dict(
            checkpoint,strict=False)
    print("Replace Default initialization... LOAD TRAINED WEIGHTS")
    return model
#############################################################################

## Setup dataloader ####################################################################################################################   
def get_loader(datalist, roi): 
    data_partition = os.path.join(datalist,'Training','training.json') 
    with open(data_partition) as f:
        json_data = json.load(f)
    test_files= set_fullpath(datalist,json_data["test"])
  
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"],reader=PILReader(reverse_indexing=False,converter=lambda image: image.convert("RGB")),dtype='float',image_only=False),
            transforms.LoadImaged(keys=["label"],reader=PILReader(reverse_indexing=False,converter=lambda image: image.convert("1")),dtype='uint8',image_only=False),
            EnsureChannelFirstd(keys=["image","label"]),
            transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=1, upper=99, b_min=-1,b_max=1, clip=True),
            transforms.Resized(keys=["image"], spatial_size=(roi[0], roi[1]), mode="bilinear"),
            transforms.Resized(keys=["label"], spatial_size=(roi[0], roi[1]), mode="nearest-exact")
            ]
       )

    test_ds = data.Dataset(data=test_files, transform=val_transform)
    test_loader = data.DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return test_loader

def set_fullpath(data_dir,files):
    files_out=[]
    for i in range(len(files)):
        img_dir = {"image": [],"label": []}
        img_dir['image'] = os.path.join(data_dir,'Training','imagesTr',files[i])
        img_dir['label'] = os.path.join(data_dir,'Training','labelsTr',files[i])
    files_out.append(img_dir)
    return files_out

## Define test  ####################################################################################################
def test_data(model, loader, acc_func,model_name, nameRun,device,model_inferer=None, post_trans=None):
    
    pathmodel = os.path.join(os.getcwd(),'DL_code',"Results",model_name,nameRun,'model.pt')
    model=restore_Model(pathmodel,model,device)
    model.eval()
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"].to(device), batch_data["label"].to(device)
            logits = model_inferer(data.float(),model)
            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_trans(val_pred_tensor) for val_pred_tensor in val_outputs_list]

            acc_func(y_pred=val_output_convert, y=val_labels_list)
            filename=os.path.join(os.getcwd(),'DL_code',"Results",model_name,nameRun,batch_data['image_meta_dict']['filename_or_obj'][0].split('/')[-1].split('.')[0]+'.png')

            im = Image.fromarray(val_output_convert[0][0].detach().cpu().numpy().astype(np.uint8)*255)
            im.save(filename)
                
        average_dice = acc_func.aggregate()[0].item()    
        print(
                "Avg Dice Coeff: ", average_dice
            )

    acc_func.reset()

    return average_dice
######################################################################################################################


if __name__ == "__main__":

    #######################################################################################################
    parser = argparse.ArgumentParser(description='Command Line Arguments')
    parser.add_argument("--data_dir", type=str, required=False,default='../GIRAFE/')
    parser.add_argument("--model_dir", type=str, required=False,default='Unet_8_100_0.0002_256_Baseline')
    
    opt = parser.parse_args()
    ## Set dataset root directory and hyper-parameters
    
    roi = (256, 256)  
    batch_size = 8
    channels=(16,32,64,128)
    strides=(2,2,2)
    nameRun=opt.model_dir
    #######################################################################################################
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name=nameRun.split('_')[0]

#######Data Loader##################
    test_loader = get_loader(opt.data_dir, roi) 

#######Model##################
    if model_name=='Unet':
        model = UNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=1,
            channels=channels,
            strides=strides,
            num_res_units=2,
            norm=Norm.BATCH,
            ).to(device)
    elif model_name=='Swinv2':
        model = SwinV2OneDecoder(img_size=roi).to(device)
    else:
        raise ValueError("model not found")
########################################################################

## Optimizer and loss function
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    dice_Test = DiceMetric(include_background=True, reduction="mean", get_not_nans=True)

    model_inferer = SimpleInferer()

    test_data(model, test_loader, dice_Test,model_name, nameRun,device,model_inferer=model_inferer, post_trans=post_trans)
