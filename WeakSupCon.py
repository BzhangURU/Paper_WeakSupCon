# python WeakSupCon.py

from __future__ import print_function

import os
import sys
import argparse
import time
from datetime import datetime
import math
import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
import torch.distributed as dist
from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.backbone_models import EncoderProjHead
from losses import SupConLoss#SupConLoss will be SimCLR loss if no labels are provided
from losses_SimilarityLoss import SimilarityLoss
import gc #gabrage collection
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image
import random
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES']='0'
count_GPUS_used=1

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass
batchsize=512 #hyperparameters #this is total batchsize
USE_RANDOM_SEED=False
random_seed=200 #hyperparameters, change it if exp_mode='resume_latest' AND USE_RANDOM_SEED is true
lr_after_warmup=0.2 #hyperparameters
count_GPUS_used_double_check=1 #hyperparameters #GPU_related
# '0Similarity1SimCLR' means the first label provided later is negative label, and Similarity Loss is used, the second label is positive label, SimCLR loss is used.
Con_method='0Similarity1SimCLR' #choices=['0Similarity1SimCLR', '0SimCLR1Similarity'] no other options
model_backbone='resnet18' #choices=['resnet18','resnet50','vit_tiny','vit_small','vit_base']
exp_dataset='Camelyon16' #choices=['Camelyon16', 'KidneyRVT', 'KidneyMeta415'] #hyperparameters
exp_mode='normal_training' #choices: 'normal_training','resume_latest' #hyperparameters: choose 'resume_latest' to resume latest training
SimLoss_weight=1.0# 1.0 weight of Similarity Loss #hyperparameters 
output_dir="./save/step1_"+exp_dataset+"_"+Con_method+"_"+model_backbone+"_"+"20250611_0Similarity_1SimCLRexp05_1" #hyperparameters 

iterations_per_epoch=5000#4000  #set to 10 for debugging #hyperparameters
loaded_initialized_model_path='N/A'
DEBUG_MODE=False
assert loaded_initialized_model_path=='N/A' or loaded_initialized_model_path.find(exp_dataset)>0

num_Camelyon_normal_samples_per_epoch = iterations_per_epoch * (batchsize//2)
num_Camelyon_tumor_samples_per_epoch = iterations_per_epoch * (batchsize//2)

num_KidneyRVT_normal_samples_per_epoch = iterations_per_epoch * (batchsize//2)
num_KidneyRVT_RVT_samples_per_epoch = iterations_per_epoch * (batchsize//2)

num_KidneyMeta415_nonMeta_samples_per_epoch = iterations_per_epoch * (batchsize//2)
num_KidneyMeta415_Meta_samples_per_epoch = iterations_per_epoch * (batchsize//2)

Camelyon_batchsizes=[(batchsize//2), (batchsize//2)]
KidneyRVT_batchsizes=[(batchsize//2), (batchsize//2)]
KidneyMeta415_batchsizes=[(batchsize//2), (batchsize//2)]

assert count_GPUS_used == count_GPUS_used_double_check
assert batchsize%2==0
assert (batchsize//2)%count_GPUS_used==0

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=200,#hyperparameters default: 200
                        help='print frequency')# print once for every print_freq batches in train
    parser.add_argument('--save_freq', type=int, default=5,
                        help='save frequency')#we save to checkpoint_latest.pth.tar, but we also save like checkpoint_30.pth.tar when %5==0
    parser.add_argument('--batch_size', type=int, default=batchsize,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=7,#hyperparameters # this is num_worker of ONE dataloader, there are 2(classes)xcount_GPUS_used dataloaders 
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=150,#100 #hyperparameters 
                        help='number of training epochs')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=lr_after_warmup,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='16,28,40,50,60,68,76,83,90,105,115',
                        help='where to decay lr, can be a list')#apply it if not using cosine annealing
    # parser.add_argument('--lr_decay_epochs', type=str, default='8,16,24,32,40,48,56,64,72,80,88,96',
    #                     help='where to decay lr, can be a list')#apply it if not using cosine annealing
    parser.add_argument('--lr_decay_rate', type=float, default=0.5,#hyperparameters
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')#used in SGD optimizer
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')#used in SGD optimizer

    # model dataset
    parser.add_argument('--model', type=str, default=model_backbone)
    parser.add_argument('--dataset', type=str, default=exp_dataset,
                        choices=['Camelyon16', 'KidneyRVT', 'KidneyMeta415'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=224, help='parameter for RandomResizedCrop')#RandomResizedCrop

    # method
    parser.add_argument('--method', type=str, default=Con_method,
                        choices=['0Similarity1SimCLR', '0SimCLR1Similarity'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.1,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.data_folder = output_dir
    opt.model_path = output_dir
    opt.tb_path = output_dir

    iterations = opt.lr_decay_epochs.split(',')#not used if you are using --cosine
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}_epoch_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial, opt.epochs)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.001
        opt.warm_epochs = 3
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = output_dir
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder, exist_ok=True)

    opt.save_folder = output_dir
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder, exist_ok=True)

    opt.txt_folder = output_dir
    if not os.path.isdir(opt.txt_folder):
        os.makedirs(opt.txt_folder, exist_ok=True)

    return opt

# for single GPU
class RandomSamplerWithReplacement(Sampler):
    def __init__(self, data_source, num_samples):
        """
        Args:
            data_source (Dataset): The dataset to sample from.
            num_samples (int): The number of samples to draw in an epoch.
        """
        self.data_source = data_source
        self.num_samples = num_samples

    def __iter__(self):
        # Yield indices sampled with replacement
        return iter(random.choices(range(len(self.data_source)), k=self.num_samples))

    def __len__(self):
        return self.num_samples

#for multiple GPUs
class DistributedRandomSamplerWithReplacement(Sampler):
    def __init__(self, data_source, num_samples, num_replicas=None, rank=None):
        """
        Distributed sampler with replacement.
        Args:
            data_source (Dataset): dataset to sample from
            num_samples (int): number of samples per epoch (total across all GPUs)
            num_replicas (int, optional): number of processes participating in distributed training
            rank (int, optional): rank of the current process
        """
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        
        self.data_source = data_source
        self.num_replicas = num_replicas
        self.rank = rank
        
        # Ensure each GPU gets equal samples
        self.num_samples_per_replica = int(math.ceil(num_samples / self.num_replicas))
        self.total_num_samples = self.num_samples_per_replica * self.num_replicas

    def __iter__(self):
        # Generate all samples with replacement for the entire epoch
        indices = random.choices(range(len(self.data_source)), k=self.total_num_samples)
        
        # Subset indices for current replica
        offset = self.rank
        indices_for_rank = indices[offset:self.total_num_samples:self.num_replicas]

        assert len(indices_for_rank) == self.num_samples_per_replica
        return iter(indices_for_rank)

    def __len__(self):
        return self.num_samples_per_replica

# Define a custom dataset with certain one label
# we could have multiple txts for one class/label if the number of images is huge. 
class ImageSingleLabelDataset(Dataset):
    def __init__(self, images_root_path_for_cur_category, list_txt_path, base_transform, dataset_label):
        self.base_transform = base_transform
        self.images_root_path_for_cur_category=images_root_path_for_cur_category
        self.img_and_label_list=[]#dim: num_images x 2(img_RELATIVE_path, label)
        print(f'Loading dataset class {dataset_label}...')
        for i in range(len(list_txt_path)):
            with open(list_txt_path[i], 'r') as file_txt:
                lines = file_txt.readlines()
            each_txt_list = [[os.path.join(self.images_root_path_for_cur_category,(line.strip())),dataset_label] for line in lines]
            self.img_and_label_list.extend(each_txt_list)
        print(f'Final total image number in dataset is {len(self.img_and_label_list)}')

        self.dataset_length = len(self.img_and_label_list)

        if images_root_path_for_cur_category.find('Camelyon16')>=0:
            self.dataset_name='Camelyon16'
        elif images_root_path_for_cur_category.find('20240601')>=0 or images_root_path_for_cur_category.find('Lab_kidney')>=0:
            self.dataset_name='Lab_kidney'
        else:
            raise ValueError(f'Dataset_name not recognized.')

        print(f'dataset name: {self.dataset_name}')

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        # Load the image at the specified index
        img_and_label = self.img_and_label_list[idx]
        image = Image.open(img_and_label[0]).convert("RGB")  # Ensure the image is in RGB format
        
        # Apply the transformation if provided
        if self.base_transform is not None:
            im1 = self.base_transform(image)
            im2 = self.base_transform(image)
            output = [im1, im2]
        return output, torch.tensor(img_and_label[1])


def set_loader_no_balance(opt, dataset_type):
    # construct data loader
    if opt.dataset == 'Camelyon16':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif opt.dataset == 'KidneyRVT':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif opt.dataset == 'KidneyMeta415':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    image_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])

    if opt.dataset == 'Camelyon16':
        print(f'Start loading {opt.dataset} dataset...')
        train_normal_txt_file="/your_own_path/202412_WeakSupCon/Camelyon16_dataset_txt/training_normal.txt"
        train_tumor_txt_file="/your_own_path/202412_WeakSupCon/Camelyon16_dataset_txt/training_tumor.txt"
        images_root_path_for_each_category=["/your_own_path2/public_datasets/Camelyon16/generated_patches",
                                            "/your_own_path2/public_datasets/Camelyon16/generated_patches"]
        list_of_list_txt_path=[[train_normal_txt_file],
                                    [train_tumor_txt_file]]
        print('Start ImageSingleLabelDataset...')
        Train_normal_dataset = ImageSingleLabelDataset(images_root_path_for_each_category[0], list_of_list_txt_path[0], image_transform, 0)
        Train_tumor_dataset = ImageSingleLabelDataset(images_root_path_for_each_category[1], list_of_list_txt_path[1], image_transform, 1)
    elif opt.dataset == 'KidneyRVT':
        print(f'Start loading {opt.dataset} dataset...')
        train_0_txt_file="/your_own_path/202404_MIL/croped_tiles/20240601/tiles_10X/No_RV_Thrombus/20250105_qualified_train_tiles_list.txt"
        train_1_txt_file="/your_own_path/202404_MIL/croped_tiles/20240601/tiles_10X/with_RV_Thrombus/20250105_qualified_train_tiles_list.txt"
        images_root_path_for_each_category=["/your_own_path2/Lab_kidney/cropped_tiles_20240601/tiles_10X",
                                            "/your_own_path2/Lab_kidney/cropped_tiles_20240601/tiles_10X"]
        list_of_list_txt_path=[[train_0_txt_file],
                                [train_1_txt_file]]
        print('Start ImageSingleLabelDataset...')
        Train_normal_dataset = ImageSingleLabelDataset(images_root_path_for_each_category[0], list_of_list_txt_path[0], image_transform, 0)
        Train_RVT_dataset = ImageSingleLabelDataset(images_root_path_for_each_category[1], list_of_list_txt_path[1], image_transform, 1)
    elif opt.dataset == 'KidneyMeta415':
        print(f'Start loading {opt.dataset} dataset...')
        train_nonmetastatic_txt_file="/your_own_path/202404_MIL/DTFD_MIL_code/DTFD-MIL/Lab_inputs_Metastatic_label_20250415_step2_kidney_tumor_slides_patches_list/20250415_qualified_nonmetastatic_train_tiles_list.txt"
        train_metastatic_txt_file="/your_own_path/202404_MIL/DTFD_MIL_code/DTFD-MIL/Lab_inputs_Metastatic_label_20250415_step2_kidney_tumor_slides_patches_list/20250415_qualified_metastatic_train_tiles_list.txt"
        images_root_path_for_each_category=["/your_own_path2/Lab_kidney_Metastasis/cropped_tiles_20240501/tiles_10X",
                                            "/your_own_path2/Lab_kidney_Metastasis/cropped_tiles_20240501/tiles_10X"]
        list_of_list_txt_path=[[train_nonmetastatic_txt_file],
                                [train_metastatic_txt_file]]
        print('Start ImageSingleLabelDataset...')
        Train_nonmetastatic_dataset = ImageSingleLabelDataset(images_root_path_for_each_category[0], list_of_list_txt_path[0], image_transform, 0)
        Train_metastatic_dataset = ImageSingleLabelDataset(images_root_path_for_each_category[1], list_of_list_txt_path[1], image_transform, 1) 
    else:
        raise ValueError(opt.dataset)

    if count_GPUS_used > 1:#GPU_related
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    if opt.dataset == 'Camelyon16':
        if count_GPUS_used > 1:#GPU_related
            train_val_sampler_Camelyon_normal = DistributedRandomSamplerWithReplacement(
                Train_normal_dataset, 
                num_samples=num_Camelyon_normal_samples_per_epoch,
                num_replicas=world_size,
                rank=rank
            )
            train_val_sampler_Camelyon_tumor = DistributedRandomSamplerWithReplacement(
                Train_tumor_dataset, 
                num_samples=num_Camelyon_tumor_samples_per_epoch,
                num_replicas=world_size,
                rank=rank
            )

        else:
            # Use the custom sampler directly for single-GPU
            train_val_sampler_Camelyon_normal = RandomSamplerWithReplacement(Train_normal_dataset, num_samples=num_Camelyon_normal_samples_per_epoch)
            train_val_sampler_Camelyon_tumor = RandomSamplerWithReplacement(Train_tumor_dataset, num_samples=num_Camelyon_tumor_samples_per_epoch)

        # Each process gets its own DataLoader, so set num_workers appropriately
        train_val_loader_normal = DataLoader(
            Train_normal_dataset, 
            batch_size=Camelyon_batchsizes[0]//count_GPUS_used, 
            shuffle=False,  # Disable shuffle as sampler handles randomness
            num_workers=opt.num_workers, 
            pin_memory=True, 
            sampler=train_val_sampler_Camelyon_normal, 
            drop_last=False
        )

        train_val_loader_tumor = DataLoader(
            Train_tumor_dataset, 
            batch_size=Camelyon_batchsizes[1]//count_GPUS_used, 
            shuffle=False,  # Disable shuffle as sampler handles randomness
            num_workers=opt.num_workers, 
            pin_memory=True, 
            sampler=train_val_sampler_Camelyon_tumor, 
            drop_last=False
        )
        return train_val_loader_normal, train_val_loader_tumor
    elif opt.dataset == 'KidneyRVT':
        if count_GPUS_used > 1:
            # Wrap the custom sampler with DistributedSampler for multi-GPU
            train_val_sampler_KidneyRVT_normal = DistributedRandomSamplerWithReplacement(
                Train_normal_dataset, 
                num_samples=num_KidneyRVT_normal_samples_per_epoch,
                num_replicas=world_size,
                rank=rank
            )
            train_val_sampler_KidneyRVT_RVT = DistributedRandomSamplerWithReplacement(
                Train_RVT_dataset, 
                num_samples=num_KidneyRVT_RVT_samples_per_epoch,
                num_replicas=world_size,
                rank=rank
            )
        else:
            # Use the custom sampler directly for single-GPU
            train_val_sampler_KidneyRVT_normal = RandomSamplerWithReplacement(Train_normal_dataset, num_samples=num_KidneyRVT_normal_samples_per_epoch)
            train_val_sampler_KidneyRVT_RVT = RandomSamplerWithReplacement(Train_RVT_dataset, num_samples=num_KidneyRVT_RVT_samples_per_epoch)

        train_val_loader_normal = DataLoader(
            Train_normal_dataset, 
            batch_size=KidneyRVT_batchsizes[0]//count_GPUS_used, 
            shuffle=False,  # Disable shuffle as sampler handles randomness
            num_workers=opt.num_workers, 
            pin_memory=True, 
            sampler=train_val_sampler_KidneyRVT_normal, 
            drop_last=False
        )

        train_val_loader_RVT = DataLoader(
            Train_RVT_dataset, 
            batch_size=KidneyRVT_batchsizes[1]//count_GPUS_used, 
            shuffle=False,  # Disable shuffle as sampler handles randomness
            num_workers=opt.num_workers, 
            pin_memory=True, 
            sampler=train_val_sampler_KidneyRVT_RVT, 
            drop_last=False
        )
        return train_val_loader_normal, train_val_loader_RVT
    elif opt.dataset == 'KidneyMeta415':
        if count_GPUS_used > 1:
            train_val_sampler_KidneyMeta_nonMeta = DistributedRandomSamplerWithReplacement(
                Train_nonmetastatic_dataset, 
                num_samples=num_KidneyMeta_nonMeta_samples_per_epoch,
                num_replicas=world_size,
                rank=rank
            )
            train_val_sampler_KidneyMeta_Meta = DistributedRandomSamplerWithReplacement(
                Train_metastatic_dataset, 
                num_samples=num_KidneyMeta_Meta_samples_per_epoch,
                num_replicas=world_size,
                rank=rank
            )
        else:
            # Use the custom sampler directly for single-GPU
            train_val_sampler_KidneyMeta_nonMeta = RandomSamplerWithReplacement(Train_nonmetastatic_dataset, num_samples=num_KidneyMeta_nonMeta_samples_per_epoch)
            train_val_sampler_KidneyMeta_Meta = RandomSamplerWithReplacement(Train_metastatic_dataset, num_samples=num_KidneyMeta_Meta_samples_per_epoch)

        train_val_loader_nonMeta = DataLoader(
            Train_nonmetastatic_dataset, 
            batch_size=KidneyMeta_batchsizes[0]//count_GPUS_used, 
            shuffle=False,  # Disable shuffle as sampler handles randomness
            num_workers=opt.num_workers, 
            pin_memory=True, 
            sampler=train_val_sampler_KidneyMeta_nonMeta, 
            drop_last=False
        )

        train_val_loader_Meta = DataLoader(
            Train_metastatic_dataset, 
            batch_size=KidneyMeta_batchsizes[1]//count_GPUS_used, 
            shuffle=False,  # Disable shuffle as sampler handles randomness
            num_workers=opt.num_workers, 
            pin_memory=True, 
            sampler=train_val_sampler_KidneyMeta_Meta, 
            drop_last=False
        )
        return train_val_loader_nonMeta, train_val_loader_Meta
    else:
        raise ValueError(opt.dataset)
        
def set_model(opt,local_rank):
    model = EncoderProjHead(name=opt.model)#use resnet18(standard version for 224x224)
    #SupConLoss will be SimCLR loss if no labels are provided
    criterionSimCLR = SupConLoss(temperature=opt.temp)
    criterionSimilarity = SimilarityLoss(temperature=opt.temp)
    
    # enable synchronized Batch Normalization, for running on multiple GPUs
    if opt.syncBN and count_GPUS_used > 1:#GPU_related
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if count_GPUS_used>1:
            model.to(local_rank) 
        else:
            model = model.cuda()

        if count_GPUS_used>1:
            criterionSimCLR = criterionSimCLR.to(local_rank)
            criterionSimilarity = criterionSimilarity.to(local_rank)
        else:
            criterionSimCLR = criterionSimCLR.cuda()
            criterionSimilarity = criterionSimilarity.cuda()
        
        if USE_RANDOM_SEED:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            
    return model, criterionSimCLR, criterionSimilarity

def save_checkpoint(state, is_best, filename=output_dir+'/checkpoint.pth.tar'):
    torch.save(state, filename)

def train(model, criterionSimCLR, criterionSimilarity, optimizer, scaler, epoch, opt, loader0, loader1):

    """one epoch training"""
    if DEBUG_MODE:
        print('This is DEBUG_MODE, num of iterations is set to very small!')
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    losses = AverageMeter()

    train_label0_iter = iter(loader0)
    train_label1_iter = iter(loader1)

    end = time.time()
    for iter_index in range(iterations_per_epoch):
        if DEBUG_MODE and iter_index>10:
            break
        
        images_label0, labels_label0=next(train_label0_iter)
        images_label1, labels_label1=next(train_label1_iter)
        bsz_label0=labels_label0.shape[0]
        bsz_label1=labels_label1.shape[0]
        
        #Camelyon, KidneyRVT, KidneyMeta415
        images = torch.cat([images_label0[0], images_label1[0], images_label0[1], images_label1[1]], dim=0)
        labels = torch.cat([labels_label0, labels_label1], dim=0)

        data_time.update(time.time() - end)

        if iter_index==0:
            print("Shape of labels in one GPU in a batch...")
            print(labels.shape)

        # concatenate two views. shape changes from [2, bsz, C, Height, Width] to [bsz*2, C, Height, Width] 
        if count_GPUS_used > 1:#GPU_related
            local_rank = dist.get_rank()
        else:
            local_rank=0
        if torch.cuda.is_available():
            if count_GPUS_used > 1:
                images = images.to(local_rank, non_blocking=True)
                images = images.to(local_rank, non_blocking=True)
            else:
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, iter_index, iterations_per_epoch, optimizer)

        # compute loss
        features, feat_after_encoder = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        features_label0, features_label1 = torch.split(features, [bsz_label0, bsz_label1], dim=0)
        with torch.cuda.amp.autocast(True):
            if Con_method=='0Similarity1SimCLR':
                loss1 = criterionSimilarity(features_label0)#no negative features
                loss2 = criterionSimCLR(features_label1) # this is actually Sim CLR  because we didn't provide labels
            elif Con_method=='0SimCLR1Similarity':
                loss2 = criterionSimCLR(features_label0)# this is actually Sim CLR because we didn't provide labels
                loss1 = criterionSimilarity(features_label1) #no negative features 
            loss=SimLoss_weight*loss1+loss2

        # update metric
        losses1.update(loss1.item(), bsz)#
        losses2.update(loss2.item(), bsz)
        losses.update(loss.item(), bsz)
        
        # SGD
        optimizer.zero_grad()#Gradients are reset to zero before computing the new gradients during backpropagation (backward()).
        
        scaler.scale(loss).backward()#Scales the loss by a scaling factor before calling .backward().
        scaler.step(optimizer)#Steps the optimizer but first unscales the gradients.
        scaler.update()#Updates the scaling factor dynamically based on gradient values.

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")

        if count_GPUS_used > 1:#GPU_related
            rank = dist.get_rank()
        else:
            rank=0

        # print info
        if (iter_index % opt.print_freq == 0 or iter_index==10) and rank==0:#GPU_related
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'loss1 {loss1.val:.3f} ({loss1.avg:.3f})\t'
                  'loss2 {loss2.val:.3f} ({loss2.avg:.3f})\t'
                  'current_time {current_time}'.format(
                   epoch, iter_index, iterations_per_epoch, batch_time=batch_time,
                   data_time=data_time, loss=losses, loss1=losses1, loss2=losses2, current_time=current_time))
            
            path_train_txt_record=opt.txt_folder+'/20250611_0Similarity_1SimCLRexp05_1_track_record.txt'
            f = open(path_train_txt_record, 'a+')
            f.write('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'loss1 {loss1.val:.3f} ({loss1.avg:.3f})\t'
                  'loss2 {loss2.val:.3f} ({loss2.avg:.3f})\t'
                  'current_time {current_time}\n'.format(
                   epoch, iter_index + 1, iterations_per_epoch, batch_time=batch_time,
                   data_time=data_time, loss=losses, loss1=losses1, loss2=losses2, current_time=current_time))
            f.close()
            sys.stdout.flush()

    return losses.avg, losses1.avg, losses2.avg

def main():
    

    opt = parse_option()

    seed = random_seed
    if count_GPUS_used > 1: #GPU_related
        # Initialize distributed training environment
        torch.distributed.init_process_group(backend='nccl')
        rank = torch.distributed.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        if USE_RANDOM_SEED:
            torch.manual_seed(seed + rank)
            random.seed(seed + rank)
    else:
        rank = 0 # id for all nodes and GPUs
        local_rank = 0 # id on local node 
        if USE_RANDOM_SEED:
            torch.manual_seed(seed)
            random.seed(seed)

    if count_GPUS_used==1 or rank == 0:#GPU_related
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # build data loader
    print('Start train_loader...')
    if opt.dataset == 'Camelyon16':
        train_val_loader_normal, train_val_loader_tumor = set_loader_no_balance(opt, "train")
    elif opt.dataset == 'KidneyRVT':
        train_val_loader_normal, train_val_loader_RVT = set_loader_no_balance(opt, "train")
    elif opt.dataset == 'KidneyMeta415':
        train_val_loader_nonMeta, train_val_loader_Meta = set_loader_no_balance(opt, "train")
    else:
        raise ValueError(f'Wrong {opt.dataset}!')

    print('Finished train_loader...\n\n')

    # build model and criterion
    # criterion is the loss function(need to load input later)
    model, criterionSimCLR, criterionSimilarity = set_model(opt,local_rank)#GPU_related

    # build optimizer
    optimizer = set_optimizer(opt, model)

    #search autocast
    scaler = torch.cuda.amp.GradScaler()

    # tensorboard logging (only on rank 0)
    if count_GPUS_used==1 or rank == 0:
        # tensorboard
        logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    start_epoch=1

    if exp_mode=='normal_training':
        if loaded_initialized_model_path!='N/A':
            model_path=loaded_initialized_model_path
            print("=> loading checkpoint '{}'".format(model_path))
            if count_GPUS_used>1:#2025 0526
                checkpoint = torch.load(model_path, map_location=f'cuda:{local_rank}')
            else:
                checkpoint = torch.load(model_path)
            print(checkpoint['state_dict'].keys())
            checkpoint_state_dict = checkpoint['state_dict']
            # Remove "module." prefix if present
            new_state_dict = {}
            for k, v in checkpoint_state_dict.items():
                new_key = k.replace("module.", "") if k.startswith("module.") else k
                new_state_dict[new_key] = v
            model.load_state_dict(new_state_dict)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            
            print("=> loaded checkpoint '{}' as initialization."
                .format(model_path))
            path_train_txt_record=opt.txt_folder+'/20250611_0Similarity_1SimCLRexp05_1_track_record.txt'
            f = open(path_train_txt_record, 'a+')
            f.write("\n=> loaded checkpoint '{}'  as initialization.\n"
                    .format(model_path))
            f.close()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            gc.collect()
    elif exp_mode=='resume_latest':
        model_path=output_dir+'/checkpoint_latest.pth.tar'
        if os.path.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            if count_GPUS_used>1:
                checkpoint = torch.load(model_path, map_location=f'cuda:{local_rank}')
            else:
                checkpoint = torch.load(model_path)
            
            start_epoch = checkpoint['epoch']+1

            checkpoint_state_dict = checkpoint['state_dict']
            # Remove "module." prefix if present
            new_state_dict = {}
            for k, v in checkpoint_state_dict.items():
                new_key = k.replace("module.", "") if k.startswith("module.") else k
                new_state_dict[new_key] = v
            model.load_state_dict(new_state_dict)
            
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(model_path, checkpoint['epoch']))

            path_train_txt_record=opt.txt_folder+'/20250611_0Similarity_1SimCLRexp05_1_track_record.txt'
            f = open(path_train_txt_record, 'a+')
            f.write("\n=> loaded checkpoint '{}' (epoch {})\n"
                  .format(model_path, checkpoint['epoch']))
            f.close()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            gc.collect()
        else:
            raise ValueError("=> no checkpoint found at '{}'".format(model_path))
    else:
        raise ValueError('exp_mode not supported: {}'.format(exp_mode))

    if count_GPUS_used > 1:#GPU_related
        # Wrap model with DistributedDataParallel
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    for epoch in range(start_epoch, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)
        # train for one epoch
        time1 = time.time()
        if opt.dataset == 'Camelyon16':
            loss, loss1, loss2 = train(model, criterionSimCLR, criterionSimilarity, optimizer, scaler, epoch, opt, train_val_loader_normal, train_val_loader_tumor)
        elif opt.dataset == 'KidneyRVT':
            loss, loss1, loss2 = train(model, criterionSimCLR, criterionSimilarity, optimizer, scaler, epoch, opt, train_val_loader_normal, train_val_loader_RVT)
        elif opt.dataset == 'KidneyMeta415':
            loss, loss1, loss2 = train(model, criterionSimCLR, criterionSimilarity, optimizer, scaler, epoch, opt, train_val_loader_nonMeta, train_val_loader_Meta)
        else:
            raise ValueError(f'Wrong {opt.dataset}')

        time2 = time.time()
        if count_GPUS_used==1 or rank == 0:#GPU_related
            print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

            # tensorboard logger
            logger.log_value('loss', loss, epoch)
            logger.log_value('loss1', loss1, epoch)
            logger.log_value('loss2', loss2, epoch)
            logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

            path_txt_record=opt.txt_folder+'/20250611_0Similarity_1SimCLRexp05_1_track_train.txt'
            now = datetime.now()
            current_time = now.strftime("%Y-%m-%d %H:%M:%S")
            f = open(path_txt_record, 'a+')
            f.write('epoch:{}, loss:{}, loss1:{},loss2:{}, lr:{}, current time:{}\n'.format(epoch,loss,loss1,loss2,optimizer.param_groups[0]['lr'],current_time))
            f.close()

            if epoch%opt.save_freq==0 or epoch==opt.epochs:
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.module.state_dict() if hasattr(model, "module") else model.state_dict(),#2025 0527
                    'optimizer' : optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                }, is_best=False, filename=output_dir+'/checkpoint_%04d.pth.tar' % epoch)
            
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.module.state_dict() if hasattr(model, "module") else model.state_dict(),#2025 0527
                'optimizer' : optimizer.state_dict(),
                'scaler': scaler.state_dict(),
            }, is_best=False, filename=output_dir+'/checkpoint_latest.pth.tar')

        optimizer.zero_grad()
        torch.cuda.empty_cache()
        gc.collect()

    # After all epochs finish:
    if count_GPUS_used > 1:
        # Cleanup distributed training environment
        torch.distributed.destroy_process_group()#GPU_related

if __name__ == '__main__':
    main()
