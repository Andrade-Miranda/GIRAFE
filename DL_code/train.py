#Setup Imports
import os
import json
import argparse

import time
import random
from monai.utils import set_determinism
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

from monai.losses import DiceLoss
from monai.inferers import SimpleInferer
from monai import transforms
from monai.transforms import (
    AsDiscrete,
    Activations,
    EnsureChannelFirstd,
    Compose
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai import data
from monai.data import decollate_batch
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.data import PILReader

import torch
from utils.swinv2Unet import SwinV2OneDecoder
import wandb

print_config()



## Setup average meter, fold reader, checkpoint saver
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)

def restore_Model(file,model,opt):
    checkpoint = torch.load(file,map_location=opt.device,weights_only=True)
    if 'state_dict' in checkpoint:
        model.load_state_dict(
            checkpoint['state_dict'],strict=False)
    else:
        model.load_state_dict(
            checkpoint,strict=False)
    print("Replace Default initialization... LOAD TRAINED WEIGHTS")
    return model
#############################################################################

######Naive load data - only one fold #######################################
def datafold_read(datalist):
    with open(datalist) as f:
        json_data = json.load(f)

    train = json_data["Training"]
    val = json_data["Val"]
    test= json_data["test"]
    return train, val, test
##############################################################################


###################Save chckpoints############################################
def save_checkpoint(model, epoch,opt, filename="model.pt", best_acc=0):
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "model_state_dict": state_dict}
    pathmodel = os.path.join(os.getcwd(),"Results",opt.model_name,opt.nameRun,filename)
    torch.save(save_dict, pathmodel)
    print("Saving checkpoint", filename)
   
 
## Setup dataloader ####################################################################################################################   
def get_loader(batch_size, json_list, roi,MoreDA):  
    datalist_json = json_list
    train_files, validation_files,test_files = datafold_read(datalist=datalist_json)
  
    train_transform = transforms.Compose(
        [  
            transforms.LoadImaged(keys=["image"],reader=PILReader(reverse_indexing=False,converter=lambda image: image.convert("RGB")),dtype='float',image_only=False),
            transforms.LoadImaged(keys=["label"],reader=PILReader(reverse_indexing=False,converter=lambda image: image.convert("1")),dtype='uint8',image_only=False),
            EnsureChannelFirstd(keys=["image","label"]),
            transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=1, upper=99, b_min=-1,b_max=1, clip=True,dtype=np.float32),
            transforms.Resized(keys=["image"], spatial_size=(roi[0], roi[1]), mode="bilinear"),
            transforms.Resized(keys=["label"], spatial_size=(roi[0], roi[1]), mode="nearest-exact"),
        ])
    if MoreDA:
        extra_transform= transforms.Compose([
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                transforms.RandRotated(keys=["image", "label"],mode=("bilinear", "nearest"),range_x=np.pi/6,range_y=np.pi/6),

                transforms.RandAffined(keys=["image", "label"],prob=0.7,shear_range=(0.5,0.5),mode=['bilinear','nearest'],padding_mode='zeros'),            
                transforms.RandAdjustContrastd(keys=["image"],prob=1,gamma=(0.8,1.2)),
            
                transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ])
        train_transform=transforms.Compose([train_transform,extra_transform])
    
    
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

    train_ds = data.Dataset(data=train_files, transform=train_transform)
    train_loader = data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_ds = data.Dataset(data=validation_files, transform=val_transform)
    val_loader = data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    test_ds = data.Dataset(data=test_files, transform=val_transform)
    test_loader = data.DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
####################################################################################################################




## Define Train  ####################################################################################################
def train_epoch(model, loader, optimizer, epoch, loss_func,wandb_logger,dice_Train,opt):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        data, target = batch_data["image"].to(opt.device), batch_data["label"].to(opt.device)
        logits = model(data.float())

        val_labels_list = decollate_batch(target)
        val_outputs_list = decollate_batch(logits)
        val_output_convert = [post_trans(val_pred_tensor) for val_pred_tensor in val_outputs_list]
        dice_Train(y_pred=val_output_convert, y=val_labels_list)        

        loss = loss_func(logits, target)
        loss.backward()
        optimizer.step()
        run_loss.update(loss.item(), n=opt.batch_size)
        print(
            "Epoch {}/{} {}/{}".format(epoch, opt.max_epochs, idx, len(loader)),
            "loss: {:.4f}".format(run_loss.avg),
            "time {:.2f}s".format(time.time() - start_time),
        )
        start_time = time.time()

    average_diceTraining = dice_Train.aggregate()[0].item()
    print(
            "dice: {:.4f}".format(average_diceTraining),
        )  
    if  wandb_logger:
        wandb_logger.log({"train/loss_epoch":run_loss.avg},step=epoch)
        wandb_logger.log({"train/Dice_epoch":average_diceTraining},step=epoch)

    dice_Train.reset()
    return run_loss.avg

## Define val  ####################################################################################################
def val_epoch(model, loader, epoch, acc_func, model_inferer=None, post_trans=None,wandb_logger=None,opt=None):
    model.eval()
    start_time = time.time()

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"].to(opt.device), batch_data["label"].to(opt.device)
            logits = model_inferer(data.float(),model)
            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_trans(val_pred_tensor) for val_pred_tensor in val_outputs_list]

            acc_func(y_pred=val_output_convert, y=val_labels_list)

            if  opt.debug:
                # Create a figure and subplots
                fig, axs = plt.subplots(2, 2, figsize=(10, 10))
                # Plot each image in a different subplot
                axs[0,0].imshow(data[0][0].detach().cpu().numpy(),cmap="gray")
                axs[0,0].set_title('Aug Original')
                axs[0,1].imshow(target[0][0].detach().cpu().numpy(),cmap="gray")
                axs[0,1].set_title('Label')
                axs[1,0].imshow(val_outputs_list[0][0].detach().cpu().numpy(),cmap="gray")
                axs[1,0].set_title('Confidence map')
                axs[1,1].imshow(val_output_convert[0][0].detach().cpu().numpy(),cmap="gray")
                axs[1,1].set_title('Prediction')
                # Remove axis ticks
                for ax in axs.flat:
                    ax.axis('off')
                # Adjust layout
                plt.tight_layout()
                # Add a common title for all subplots
                fig.suptitle(batch_data['image_meta_dict']['filename_or_obj'][0].split('/')[-1], fontsize=16)
                #plt.show()
                plt.savefig(os.path.join(os.getcwd(),"Results",opt.model_name,opt.nameRun,str(idx)+'.pdf'))
            

        average_dice = acc_func.aggregate()[0].item()    
        print(
                "Val {}/{} {}/{}".format(epoch, opt.max_epochs, idx, len(loader)),
                ", Avg Dice Coeff:", average_dice,
                ", time {:.2f}s".format(time.time() - start_time),
            )
        if  wandb_logger:
            wandb_logger.log({"val/Dice_epoch":average_dice},step=epoch)

    
    acc_func.reset()
    return average_dice


## Define test  ####################################################################################################
def test_data(model, loader, acc_func, model_inferer=None, post_trans=None,wandb_logger=None):
    
    pathmodel = os.path.join(os.getcwd(),"Results",opt.model_name,opt.nameRun,'model.pt')
    model=restore_Model(pathmodel,model,opt)
    model.eval()
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"].to(device), batch_data["label"].to(device)
            logits = model_inferer(data.float(),model)
            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_trans(val_pred_tensor) for val_pred_tensor in val_outputs_list]

            acc_func(y_pred=val_output_convert, y=val_labels_list)
            filename=os.path.join(os.getcwd(),"Results",opt.model_name,opt.nameRun,batch_data['image_meta_dict']['filename_or_obj'][0].split('/')[-1].split('.')[0]+'.png')

            im = Image.fromarray(val_output_convert[0][0].detach().cpu().numpy().astype(np.uint8)*255)
            im.save(filename)
                
        average_dice = acc_func.aggregate()[0].item()    
        print(
                "Avg Dice Coeff: ", average_dice
            )
        if  wandb_logger:
            wandb_logger.log({"Test/Dice":average_dice})

    acc_func.reset()

    return average_dice
######################################################################################################################



## Define Trainer
def trainer(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    dice_Train,
    scheduler,
    model_inferer=None,
    start_epoch=0,
    post_trans=None,
    val_every=5,
    wandb_logger=None,
    opt=None
    ):

    val_acc_max = 0.0
    dices_avg = []
    loss_epochs = []
    trains_epoch = []
    
    for epoch in range(start_epoch, opt.max_epochs):
        print(time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            epoch=epoch,
            loss_func=loss_func,
            wandb_logger=wandb_logger,dice_Train=dice_Train,opt=opt
        )
        print(
            "Final training  {}/{}".format(epoch, opt.max_epochs),
            "loss: {:.4f}".format(train_loss),
            "time {:.2f}s".format(time.time() - epoch_time),
        )
        
        if (epoch ) % val_every == 0 :
            loss_epochs.append(train_loss)
            trains_epoch.append(int(epoch))
            epoch_time = time.time()
            
            # Calculate average Dice coefficient for validation
            val_avg_dice = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                post_trans=post_trans,
                wandb_logger=wandb_logger,opt=opt
            )
            print(
                "Final validation stats {}/{}".format(epoch, opt.max_epochs - 1),
                ", Dice_Avg:",
                val_avg_dice,
                ", time {:.2f}s".format(time.time() - epoch_time),
            )
            dices_avg.append(val_avg_dice)
            
            if val_avg_dice > val_acc_max:
                print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_dice))
                val_acc_max = val_avg_dice
                save_checkpoint(
                    model,
                    epoch,
                    opt,
                    best_acc=val_acc_max,
                )
            scheduler.step()
    
    print("Training Finished !, Best Accuracy: ", val_acc_max)
    return (
        val_acc_max,
        dices_avg,
        loss_epochs,
        trains_epoch,
    )



if __name__ == "__main__":

# acquire and parse input and output paths
    parser = argparse.ArgumentParser(description='Command Line Arguments')
    parser.add_argument("--json_list", type=str, required=False,default='../GIRAFE/Training/training.json',
                        help="Path of the json file used for training, validation and test")
    parser.add_argument("--roi", type=int, nargs=3, default= (256, 256) , help='Image Size')
    parser.add_argument('--batch_size', type=int, default=8, help='# the batchsize')
    parser.add_argument('--max_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--val_every', type=int, default=2, help='number of epochs to wait to perform validation')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--seed', type=int, default=1234, help='# of seed for deterministic training')
    parser.add_argument('--wandb_act', dest='wandb_act',action='store_true', default=False, help='activate wandb')  
    parser.add_argument('--debug', dest='debug',action='store_true', default=False, help='debugging during validation')  
    parser.add_argument("--project_name", type=str, required=False,default='Glottis',help="Wandb project name")
    parser.add_argument("--entity", type=str, required=False,default='xamus86',help="wandb project entity")      
    parser.add_argument("--model_name", type=str, default='Unet',help="DL model")    
    parser.add_argument('--channels', nargs='+', default=(16,32,64,128), help='filters for the CNN network')
    parser.add_argument('--strides', nargs='+', default=(2,2,2), help='strides for the CNN network')
    parser.add_argument("--nameRun_detail", type=str, required=False,default='Baseline',help="name of the current running")    
    parser.add_argument('--MoreDA', dest='MoreDA',action='store_true', default=False, help='include extra data augmentation') 
    parser.add_argument('--device', type=str, required=False,default='cuda',help="cehck cuda or cpu device")  


    opt = parser.parse_args()

    #######################################################################################################
    ## Set dataset root directory and hyper-parameters
    os.chdir(os.path.join(os.getcwd(),'DL_code'))
    opt.nameRun=opt.model_name+'_'+str(opt.batch_size)+'_'+str(opt.max_epochs)+'_'+str(opt.lr)+'_'+str(opt.roi[0])+'_'+opt.nameRun_detail
    opt.dir_wandb=os.path.join(os.getcwd(),'wandb')
    #######################################################################################################

    set_determinism(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a dictionary from local variables
    #opt = {k: v for k, v in locals().items() if k in ['json_list', 'roi', 'batch_size','max_epochs','val_every','lr','seed','wandb_act','project_name','model_name','nameRun_detail','dir_wandb','nameRun','debug','strides','channels']}
    opt.device=device

    if opt.wandb_act:
        wandb_logger = wandb.init(project=opt.project_name,
                                      entity=opt.entity,
                                      config=opt,
                                      name=opt.nameRun,
                                    dir=opt.dir_wandb)
    else:                                
        wandb_logger=None

    if not os.path.isdir(os.path.join(os.getcwd(),"Results",opt.model_name,opt.nameRun)):
        os.makedirs(os.path.join(os.getcwd(),"Results",opt.model_name,opt.nameRun))

#######Data Loader##################
    train_loader, val_loader, test_loader = get_loader(opt.batch_size, opt.json_list, opt.roi,opt.MoreDA) 

#######Model##################
    if opt.model_name=='Unet':
        model = UNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=1,
            channels=opt.channels,
            strides=opt.strides,
            num_res_units=2,
            norm=Norm.BATCH,
            ).to(opt.device)
    elif opt.model_name=='Swinv2':
        model = SwinV2OneDecoder(pretrained=False,img_size=opt.roi).to(opt.device)
    else:
        raise ValueError("model not found")
########################################################################

## Optimizer and loss function
    dice_loss = DiceLoss(to_onehot_y=False, sigmoid=True)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    dice_val = DiceMetric(include_background=True, reduction="mean", get_not_nans=True)
    dice_Train = DiceMetric(include_background=True, reduction="mean", get_not_nans=True)
    dice_Test = DiceMetric(include_background=True, reduction="mean", get_not_nans=True)

    model_inferer = SimpleInferer()

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=1e-5) 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.max_epochs) 

    ## Execute training
    start_epoch = 1
    (
    val_acc_max,
    dices_avg,
    loss_epochs,
    trains_epoch,
                ) = trainer(
                            model=model,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            optimizer=optimizer,
                            loss_func=dice_loss,
                            acc_func=dice_val,
                            dice_Train=dice_Train,
                            scheduler=scheduler,
                            model_inferer=model_inferer,
                            start_epoch=start_epoch,
                            post_trans=post_trans,
                            val_every=opt.val_every,
                            wandb_logger=wandb_logger,
                            opt=opt
                            )

    print(f"train completed, best average dice: {val_acc_max:.4f} ")


    ### Plot the loss and Dice metric
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    plt.xlabel("epoch")
    plt.plot(trains_epoch, loss_epochs, color="red")
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    plt.xlabel("epoch")
    plt.plot(trains_epoch, dices_avg, color="green")
    # Save the figure
    plt.savefig(os.path.join(os.getcwd(),"Results",opt.model_name,opt.nameRun,"PlotLoss.pdf"))
    #plt.show()

    test_data(model, test_loader, dice_Test, model_inferer=model_inferer, post_trans=post_trans,wandb_logger=wandb_logger)


