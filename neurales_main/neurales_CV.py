# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:01:39 2020

@author: XZ-WAVE
"""

import torch
import torch.nn as nn
import torch.utils as utils
import torchvision
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sklearn as sk
from neurales_vision_utils import *

#===============TRANSFORMS==========================
'''
Here we make a functional API for pytorch's transforms, and helpful utilities to combine transforms
'''
resize = lambda shape: transforms.Resize(size=shape)
color_jitter = lambda brightness,contrast,saturation,hue: transforms.ColorJitter(brightness=brightness,contrast=contrast,
                                                                                 saturation=saturation,hue=hue)
random_affine = lambda degrees,translate,scale=0,shear=0: transforms.RandomAffine(degrees=degrees,translate=translate,scale=scale,
                                                                              shear=shear)

def transform_pipeline(steps):
    transform_list = []
    for step in range(len(steps)):
        if steps[step] == 'resize':
            length = int(input('enter in an image length:\t'))
            width = int(input('enter in an image width:\t'))
            transform_list.append(resize(shape=(length,width)))
        if steps[step] == 'color_jitter':
            brightness = float(input('enter in a brightness:\t')) #add a range?
            contrast = float(input('enter in a contrast:\t')) #add a range?
            saturation = float(input('enter in a saturation:\t')) #add a range?
            hue = float(input('enter in a hue:\t')) #add a range?
            transform_list.append(color_jitter(brightness,contrast,saturation,hue))
        if steps[step] == 'random_affine':
            degrees = int(input('enter in the max rotation degree:\t')) #add a range?
            translate = int(input('enter in number of pixels to translate the images:\t')) #add a range?
            scale = float(input('enter in a scale:\t')) #add a range?
            shear = float(input('enter in a shear:\t')) #add a range?
            transform_list.append(random_affine(degrees,translate,scale,shear))
        transform_list.append(transforms.ToTensor())
    return transform_list
        

def make_transform_pipeline(steps):
    transform_list = []
    for transf in range(steps):
        if steps[transf] == 'resize':
            transform_list.append(transforms.Resize())
            pass
    
#=================DATASETS==============================
'''
Here we have APIs for custom image loading. Keep in mind we expect the user to already have folders that contain images
The folder structure will be ./class_1 ./class_2 , .... 
This API has to be maintained because Pytorch assigns a unique label for each unique folder, and all of the utils later use this structure
'''
def load_custom_image_dset(path,transform='default'):
    '''
    path: string - the root directory for the dataset
    transform: string if default, list of transforms otherwise (using the Pytorch API)
    '''
    if transform == 'default':
        transf = transforms.Compose([transforms.Resize((40,40)),transforms.ToTensor()])
    if transform != 'default':
        transf=transf
    dataloader=utils.data.DataLoader(dataset=torchvision.datasets.ImageFolder(root=path, 
                                     transform=transf, 
                                     ))
    return dataloader


    
def random_transform(img_size=(40,40)):
    '''
    img_size: a 2-tuple, default of 40 x 40
    creates a random transform for a given image size, this function can be improved, as I have yet to 
    write code for all of Pytorch's possible transforms
    '''
    hues = np.arange(0,0.5,step=0.01)
    saturations = np.arange(0,0.5,step=0.01)
    contrasts = np.arange(0,0.5,step=0.01)
    saturations = np.arange(0,0.05,step=0.01)
    translates = np.random.uniform(0,0.1)
    degrees = 180
    shears = 45
    default =  transforms.Compose([transforms.Resize(size=img_size),transforms.ToTensor()])
    affine = transforms.Compose([transforms.Resize(size=img_size),
                                 transforms.RandomAffine(degrees=degrees,translate=[translates,translates],shear=np.random.choice(a=shears)),
                                 transforms.ToTensor()])
    color_jitter = transforms.Compose([transforms.Resize(size=img_size),
                                 transforms.ColorJitter(hue=np.random.choice(a=hues),saturation=np.random.choice(a=saturations),contrast=np.random.choice(a=contrasts)),
                                 transforms.ToTensor()])
    transform = [default,affine,color_jitter]
    return transform
    
        
    
def plot_torch_samples(dataloader):
    '''
    dataloader: dataloader object from Pytorch
    plots the data samples in a dataloader, can be used to quickly visualize the dataset and see if a given transform was applied correctly
    '''
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    for i,sample in enumerate(dataloader,0):
#        import pdb
#        pdb.set_trace()
        plt.title('sample: {} | class: {}'.format(i+1,sample[1].item()+1))
        plt.imshow(sample[0][0].detach().numpy().transpose((1,2,0)))
        plt.axis('off')
        plt.show()

def plot_imgs_from_path(path,transform='default',figsize=(8,8)): 
    '''
    same as above, but also puts labels on the samples
    '''
    if transform == 'default':
        transf = transforms.Compose([transforms.Resize((40,40)),transforms.ToTensor()])
        fontsize = 15
    if transform != 'default':
        transf = transforms.Compose(transform)
    dataloader=utils.data.DataLoader(dataset=torchvision.datasets.ImageFolder(root=path, 
                                     transform=transf, 
                                     ))
    fontsize = 15
    
    
    classes = os.listdir(path)
    for i,sample in enumerate(dataloader,0):
        plt.title('sample: {} | class: {}'.format(i+1,classes[sample[1].item()]),fontsize=fontsize)
        plt.imshow(sample[0][0].detach().numpy().transpose((1,2,0)))
        plt.axis('off')
        plt.show()

def imgs_from_dataloader(dataloader): #still need to test
    imgs = []
    for i, img in enumerate(dataloader,0):
        imgs.append(img[0])
    return torch.stack(imgs).squeeze(1)

def targets_from_dataloader(dataloader):
    '''
    gets targets from dataloader
    '''
    target_list = []
    for i,targets in enumerate(dataloader,0):
        target_list.append(targets[1].item())
    return torch.Tensor(target_list)

def plot_manifold_embedding_from_path(path,transform='default',figsize=(10,10),embedding='tsne'):
    if transform == 'default':
        transf = transforms.Compose([transforms.Resize((40,40)),transforms.ToTensor()])
        fontsize = 15
    if transform != 'default':
        transf = transforms.Compose(transform)
    dataloader=utils.data.DataLoader(dataset=torchvision.datasets.ImageFolder(root=path, 
                                     transform=transf, 
                                     ))
    fontsize = 15
    classes = os.listdir(path)
    imgs = imgs_from_dataloader(dataloader)
    classes = os.listdir(path)
    class_idxs = targets_from_dataloader(dataloader)
    torch_imgs_class_embedding(imgs=imgs,targets=class_idxs,labels=classes,embedding=embedding)
    
def plot_img_manifolds_from_path(path,transform='default',figsize=(8,8),embedding='tsne'):
    '''
    this function plots a 2-d embedding of the data points (img tensors usually) and then plots the corresponding image on top of the 2-d cluster point
    '''
    if transform == 'default':
        transf = transforms.Compose([transforms.Resize((40,40)),transforms.ToTensor()])
        fontsize = 15
    if transform != 'default':
        transf = transforms.Compose(transform)
    dataloader=utils.data.DataLoader(dataset=torchvision.datasets.ImageFolder(root=path, 
                                     transform=transf, 
                                     ))
    fontsize = 15
    imgs = imgs_from_dataloader(dataloader)
    classes = os.listdir(path)
    class_idxs = targets_from_dataloader(dataloader)
    tensor_imgs_visualization_2d(imgs=imgs,targets=class_idxs,labels=classes,embedding='tsne',figsize=(12,12),
                                 image_zoom=1.75)

def learned_embedding(imgs,layer,model,path,targets,dim=2,plot_trustworthiness='off'):
    '''
    produces manifold separation for images in a batch
    '''
    
    feature_maps = list(model.children())[0][0:layer].forward(imgs) 
    labels = os.listdir(path) #gets class labels
    colors = np.random.choice(a=['red','green','blue','royalblue','magenta','pink','black','orange','purple','gold'],
                                  size=len(labels),replace=False)
    class_idxs = np.arange(0,len(labels))
    imgs = imgs.reshape(imgs.shape[0],imgs.shape[1]*imgs.shape[2]*imgs.shape[3])
    if len(feature_maps.shape) == 4: #if the layer is a convolutional layer
        feature_maps = feature_maps.reshape(feature_maps.shape[0],feature_maps.shape[1]*feature_maps.shape[2]*feature_maps.shape[3])
        embedding = tsne_transform(X=feature_maps.detach().numpy(),dim=dim)
        plot_class_scatter(X=imgs,X_embedded=embedding,y=targets,labels=labels,plot_trustworthiness='off')
    if len(feature_maps.shape) == 2: #if the layer is a FC layer
        embedding = tsne_transform(X=feature_maps.detach().numpy(),dim=dim)
        plot_class_scatter(X=imgs,X_embedded=embedding,y=targets,labels=labels,plot_trustworthiness='off')


        
def tensor_class_breakdown(path):
    '''
    class histogram from a directory
    '''
    class_labels = os.listdir(path)
    color_list = np.random.choice(a=['red','green','blue','royalblue','magenta','pink','black','orange','purple','gold'],
                                  size=len(class_labels),replace=False)
    transf = transforms.Compose([transforms.Resize((40,40)),transforms.ToTensor()])
    dataloader=utils.data.DataLoader(dataset=torchvision.datasets.ImageFolder(root=path, 
                                     transform=transf, 
                                     ))
    targets = targets_from_dataloader(dataloader)
    vals = sorted(pd.value_counts(targets.detach().numpy()))
    plt.title('Class breakdown',fontsize=15)
    if len(vals) <= 10:
       
        plt.bar(x=np.arange(1,len(class_labels)+1,1),height=vals,
                color=color_list,tick_label=class_labels)
        plt.xticks(np.arange(1,len(class_labels)+1,1))
        plt.show()
    if len(vals) > 10: 
        plt.bar(x=np.arange(1,len(class_labels)+1,1),height=vals,color=np.random.choice(color_list))
        plt.xticks(np.arange(1,len(class_labels)+1,1),fontsize=14)
        plt.show()
    return vals
    

def make_filters(num_layers,nc=3):
    '''
    helper function for creating convolutional architectures, not to be used separately
    '''
    filters = [nc] #only let users select [1,2,3,5] and every multiple of 2
    for layer in range(num_layers):
        channels = int(input('enter in the number of channels/filters for layer {}/{}:\t'.format(layer+1,num_layers)))
        filters.append(channels)
    return filters




def make_kernels(num_layers):
    '''
    helper function for creating convolutional architectures, not to be used seperately
    '''
    kernels = []
    for layer in range(num_layers):
        kernel = int(input('enter in the number of kernel size (beta only supports square kernels) for layer {}/{}:\t'.format(layer+1,num_layers)))
        kernels.append(kernel)
    return kernels

class custom_cnn(nn.Module):
    def __init__(self,num_conv_layers,nc=3,num_classes=10,
                 batch_norm=False,dropout=False,
                 adaptive_pool_type='max',adaptive_pool_size=5,
                 conv_dim=2):
        '''
        custom CNN for the users, only has custom convolutional layers, no pooling or custom activations yet
        note that the number of channels is either 1,2,3,5 or a multiple of 2 after that (this helps with plotting later)
        We need to limit the front-end API to only allow these for the beta
        '''
        super(custom_cnn,self).__init__()
        self.conv_dim = conv_dim 
        self.num_classes = num_classes
        self.conv_layers = []
        self.fc_layers = []
        self.filters = make_filters(nc=nc,num_layers=num_conv_layers)
        self.kernels = make_kernels(num_layers=num_conv_layers)
       
       
        for layer in range(num_conv_layers):
            if self.conv_dim == 1: 
                self.adaptive_pool_size = adaptive_pool_size
                self.conv_layers.append(nn.Conv1d(in_channels=self.filters[layer],out_channels=self.filters[layer+1],
                                                  kernel_size=self.kernels[layer]))
                self.conv_layers.append(nn.PReLU(num_parameters=self.filters[layer+1]))
                if batch_norm == True:
                    self.conv_layers.append(nn.BatchNorm1d(self.filters[layer+1]))
                if dropout == True:
                    prob = float(input('enter in a dropout prob: '))
                    self.conv_layers.append(nn.Dropout(p=prob))
                if layer == num_conv_layers-1:
                    if adaptive_pool_type == 'max':
                        self.adaptive_pool = self.conv_layers.append(nn.AdaptiveMaxPool1d(self.adaptive_pool_size))
                    if adaptive_pool_type == 'avg':
                        self.adaptive_pool = self.conv_layers.append(nn.AdaptiveAvgPool1d(self.adaptive_pool_size))
            
            if self.conv_dim == 2: 
                self.adaptive_pool_size = (adaptive_pool_size,adaptive_pool_size)
                self.conv_layers.append(nn.Conv2d(in_channels=self.filters[layer],out_channels=self.filters[layer+1],
                                                  kernel_size=self.kernels[layer]))
                
                self.conv_layers.append(nn.PReLU(num_parameters=self.filters[layer+1]))
                if batch_norm == True:
                    self.conv_layers.append(nn.BatchNorm2d(self.filters[layer+1]))
                if dropout == True:
                    prob = float(input('enter in a dropout prob: '))
                    self.conv_layers.append(nn.Dropout(p=prob))
                if layer == num_conv_layers-1:
                    if adaptive_pool_type == 'max':
                        self.adaptive_pool = self.conv_layers.append(nn.AdaptiveMaxPool2d(self.adaptive_pool_size))
                    if adaptive_pool_type == 'avg':
                        self.adaptive_pool = self.conv_layers.append(nn.AdaptiveAvgPool2d(self.adaptive_pool_size))
                        
            if self.conv_dim == 3: 
                self.adaptive_pool_size = (adaptive_pool_size,adaptive_pool_size,adaptive_pool_size)
                self.conv_layers.append(nn.Conv3d(in_channels=self.filters[layer],out_channels=self.filters[layer+1],
                                                  kernel_size=self.kernels[layer]))
                self.conv_layers.append(nn.PReLU(num_parameters=self.filters[layer+1]))
                if batch_norm == True:
                    self.conv_layers.append(nn.BatchNorm3d(self.filters[layer+1]))
                if dropout == True:
                    prob = float(input('enter in a dropout prob: '))
                    self.conv_layers.append(nn.Dropout(p=prob))
                if layer == num_conv_layers-1:
                    if adaptive_pool_type == 'max':
                        self.adaptive_pool = self.conv_layers.append(nn.AdaptiveMaxPool3d(self.adaptive_pool_size))
                    if adaptive_pool_type == 'avg':
                        self.adaptive_pool = self.conv_layers.append(nn.AdaptiveAvgPool3d(self.adaptive_pool_size))  
                        
        self.hidden_dim = int(input('enter in the size of the hidden layers:\t'))
        self.in_features = np.prod(self.adaptive_pool_size)*self.filters[-1]
        self.fc_layers.append(nn.Linear(in_features=self.in_features,
                                        out_features=self.hidden_dim))
        self.fc_layers.append(nn.ReLU())
        self.fc_layers.append(nn.Linear(self.hidden_dim,self.num_classes))
        self.conv_block = nn.Sequential(*self.conv_layers)
        self.fc_block = nn.Sequential(*self.fc_layers)
        
    def forward(self,x):
        self.conv_map = self.conv_block.forward(x)
        self.conv_map = self.conv_map.view(-1,
                                           self.filters[-1]*np.prod(self.adaptive_pool_size))
    
        self.fc_map = self.fc_block.forward(self.conv_map)
        #out = F.log_softmax(self.fc_map,dim=0)
        return self.fc_map
            
class custom_fcnn(nn.Module):
    def __init__(self,num_conv_layers,nc=3,num_classes=10,batch_norm=False,
                 adaptive_pool_type='max',adaptive_pool_size=(10,10)):
        super(custom_fcnn,self).__init__()
        self.num_classes = num_classes
        self.conv_layers = []
        self.fc_layers = []
        self.filters = make_filters(nc=nc,num_layers=num_conv_layers)
        self.kernels = make_kernels(num_layers=num_conv_layers)
        self.adaptive_pool_size = adaptive_pool_size
        for layer in range(num_conv_layers):
            
            self.conv_layers.append(nn.Conv2d(in_channels=self.filters[layer],
                                              out_channels=self.filters[layer+1],
                                              kernel_size=self.kernels[layer]))
            self.conv_layers.append(nn.PReLU(num_parameters=self.filters[layer+1]))
        
        if adaptive_pool_type == 'max':
            self.adaptive_pool = self.conv_layers.append(nn.AdaptiveMaxPool2d(self.adaptive_pool_size))
        if adaptive_pool_type == 'avg':
            self.adaptive_pool = self.conv_layers.append(nn.AdaptiveAvgPool2d(self.adaptive_pool_size))
        self.conv_block = nn.Sequential(*self.conv_layers)
    def forward(self,x):
        self.conv_map = self.conv_block(x)
        out = self.conv_map.view(-1,self.filters[-1]*np.prod(self.adaptive_pool_size))
        return out

class custom_ConvAE(nn.Module):
    def __init__(self,num_conv_layers,nc=3):
        '''
        custom autoencoder
        '''
        super(custom_ConvAE,self).__init__()
        self.num_conv_layers = num_conv_layers
        self.filters = make_filters(nc=nc,num_layers=self.num_conv_layers)
        self.kernels = make_kernels(num_layers=num_conv_layers)
        self.encoder_layers = []
        self.decoder_layers = []
        for layer in range(self.num_conv_layers):
            
            self.encoder_layers.append(nn.Conv2d(in_channels=self.filters[layer],
                                                 out_channels=self.filters[layer+1],
                                                 kernel_size=self.kernels[layer]))
            self.decoder_layers.append(nn.ConvTranspose2d(in_channels=self.filters[::-1][layer],
                                                          out_channels=self.filters[::-1][layer+1],
                                                          kernel_size=self.kernels[::-1][layer]))
            
            self.encoder_layers.append(nn.PReLU(num_parameters=self.filters[layer+1]))
            self.decoder_layers.append(nn.PReLU(num_parameters=self.filters[::-1][layer+1]))
        self.decoder_layers.append(nn.Sigmoid())
        self.encoder = nn.Sequential(*self.encoder_layers)
        self.decoder = nn.Sequential(*self.decoder_layers)
        
    def forward(self,x):
        self.encoded = self.encoder(x)
        self.decoded = self.decoder(self.encoded)
        return self.encoded,self.decoded

class custom_stacked_ConvAE(nn.Module):
    def __init__(self,num_conv_layers,nc=3,num_blocks=3):
        super(custom_stacked_ConvAE,self).__init__()
        self.blocks = []
        for block in range(num_blocks):
            pass
            
class custom_gan_beta(nn.Module):
    def __init__(self,num_conv_layers,nc=3,noise_dim=15):
        '''
        this class is NOT suitable for Neurales yet, don't use this one, I plan on repurposing this code later
        '''
        super(custom_gan_beta,self).__init__()
        self.num_conv_layers = num_conv_layers
        self.filters = make_filters(nc=nc,num_layers=self.num_conv_layers)
        self.kernels = make_kernels(num_layers=num_conv_layers)
        self.generator_layers = [nn.ConvTranspose2d(in_channels=noise_dim,out_channels=self.filters[0],kernel_size=1),
                                 nn.PReLU(num_parameters=self.filters[0])]
        self.discriminator_layers = [nn.Conv2d(in_channels=nc,out_channels=self.filters[-1],kernel_size=1),
                                     nn.PReLU(num_parameters=self.filters[-1])]
        for layer in range(self.num_conv_layers):
            
            self.generator_layers.append(nn.ConvTranspose2d(in_channels=self.filters[layer],
                                                 out_channels=self.filters[layer+1],
                                                 kernel_size=self.kernels[layer]))
            self.discriminator_layers.append(nn.Conv2d(in_channels=self.filters[::-1][layer],
                                                          out_channels=self.filters[::-1][layer+1],
                                                          kernel_size=self.kernels[::-1][layer]))
            
            self.generator_layers.append(nn.PReLU(num_parameters=self.filters[layer+1]))
            self.discriminator_layers.append(nn.PReLU(num_parameters=self.filters[::-1][layer+1]))
            
        self.generator_layers.append(nn.ConvTranspose2d(in_channels=self.filters[-1],out_channels=nc,kernel_size=1))
        self.G = nn.Sequential(*self.generator_layers)
        self.D = nn.Sequential(*self.discriminator_layers)
        
    def forward(self,z):
        self.g_out = self.G(z)
        self.d_out = self.D(self.g_out)
        self.conv_vector = self.d_out.view(self.d_out.shape[0],
                                              self.d_out.shape[1]*self.d_out.shape[2]*self.d_out.shape[3])
        self.fc_layer = nn.Linear(in_features=self.conv_vector.shape[1],out_features=2)
        return torch.sigmoid(self.fc_layer.forward(self.conv_vector))

class Gen(nn.Module):
    def __init__(self,num_conv_layers,nc=3,noise_dim=15):
        '''
        custom generator class, this is not used directly by the user, I construct a high level API from this class which
        serves as a building block
        '''
        super(Gen,self).__init__()
        self.nc = nc
        self.num_conv_layers = num_conv_layers
        self.filters = make_filters(nc=nc,num_layers=self.num_conv_layers)
        self.kernels = make_kernels(num_layers=num_conv_layers)
        self.generator_layers = [nn.ConvTranspose2d(in_channels=noise_dim,out_channels=self.filters[0],kernel_size=1),
                                 nn.PReLU(num_parameters=self.filters[0])]
        for layer in range(self.num_conv_layers):
            self.generator_layers.append(nn.ConvTranspose2d(in_channels=self.filters[layer],
                                                 out_channels=self.filters[layer+1],
                                                 kernel_size=self.kernels[layer]))
            self.generator_layers.append(nn.PReLU(num_parameters=self.filters[layer+1]))
        self.generator_layers.append(nn.ConvTranspose2d(in_channels=self.filters[-1],out_channels=nc,kernel_size=1))
        self.G = nn.Sequential(*self.generator_layers)
    def forward(self,x):
        return self.G.forward(x)

class Dis(nn.Module):
    def __init__(self,filters,kernels,nc=3,num_layers=3):
        '''
        like the Gen class, this makes custom discriminator model that mirrors the architecture of the Gen model
        This is a building block for the API for the user to make GANs by inputting some values. Note the filters and kernels passed 
        to this class need to be the same as in the Gen model, and make sure the number of layers match
        '''
        super(Dis,self).__init__()
        self.nc= nc
        self.filters = filters
        self.kernels = kernels
        self.discriminator_layers = [nn.Conv2d(in_channels=self.nc,out_channels=self.filters[-1],kernel_size=1),
                                     nn.PReLU(num_parameters=self.filters[-1])]
        self.num_layers = num_layers
        for layer in range(num_layers):
            self.discriminator_layers.append(nn.Conv2d(in_channels=self.filters[::-1][layer],
                                                          out_channels=self.filters[::-1][layer+1],
                                                          kernel_size=self.kernels[::-1][layer]))
            
            
            self.discriminator_layers.append(nn.PReLU(num_parameters=self.filters[::-1][layer+1]))
        self.D = nn.Sequential(*self.discriminator_layers)
        
    def forward(self,x):
        self.d_out = self.D(x)
        self.conv_vector = self.d_out.view(self.d_out.shape[0],
                                              self.d_out.shape[1]*self.d_out.shape[2]*self.d_out.shape[3])
        self.fc_layer = nn.Linear(in_features=self.conv_vector.shape[1],out_features=2)
        return F.sigmoid(self.fc_layer.forward(self.conv_vector))

def mk_custom_gan(num_conv_layers,nc=3,noise_dim=3):
    '''
    API for making custom Deep Convolutional GAN (DCGAN)
    note that this is 2-dimensional only 
    
    num_conv_layers: int - number of conv-prelu blocks to use
    nc: int - number of input channels
    noise_dim: int - number of noisy channels to feed into the gan 
    '''
    G = Gen(num_conv_layers=num_conv_layers,nc=nc,noise_dim=noise_dim)
    filters = G.filters
    kernels = G.kernels
    nc = G.nc
    
    model = {'G': G,
             'D': Dis(filters,kernels,nc,num_layers=num_conv_layers)}
    return model
        
    
def plot_class_confidence(prob_vec,class_labels,color='red'):
    '''
    takes a softmax vector output and uses that to visualize the prediction confidence for each class
    '''
    color_list = np.random.choice(a=['red','green','blue','royalblue','magenta','pink','black','orange','purple','gold'],
                                  size=len(class_labels),replace=False)
    prob_vec = prob_vec[0].detach().numpy()
    if len(class_labels) <= 10:
        plt.title('Class confidence',fontsize=15)
        plt.bar(x=np.arange(1,len(class_labels)+1,1),height=prob_vec,color=color_list,tick_label=class_labels)
        plt.xticks(np.arange(1,len(class_labels)+1,1))
        plt.show()
    if len(class_labels) > 10: 
        plt.bar(x=np.arange(1,len(class_labels)+1,1),height=prob_vec,color=np.random.choice(color_list))
        plt.xticks(np.arange(1,len(class_labels)+1,1),fontsize=14)
        plt.show()

def visualize_filters(layer,model,img,cmap='none'):
    '''
    visualizing the convolutional filters in a given layer, the dynamic size adjustment of the images still needs tweaking.
    Note that the model is assumed to be trained at this point
    '''
        
    feature_map = model.conv_block[0:int(2*layer)].forward(img.unsqueeze(0))
    num_channels = feature_map.shape[1]
    rows = 2
    cols = 1
    if num_channels % 4 == 0:
        rows = 2
        cols = int(num_channels/rows)
    if num_channels % 6 == 0:
        rows = 3
        cols = int(num_channels/rows)
    if num_channels % 8 == 0:
        rows = 4
        cols = int(num_channels/rows)
    if num_channels % 16 == 0:
        rows = 8
        cols = int(num_channels/rows)
    if num_channels % 32 == 0:
        rows = 16
        cols = int(num_channels/rows)
    fig = plt.figure(figsize=(18,18))
    for j in range(1,num_channels+1):
        fig.add_subplot(rows,cols,j)
        plt.title('Channel: {}'.format(j))
        plt.imshow(feature_map[0][j-1].detach().numpy(),cmap=cmap)
        plt.axis('off')
        plt.tight_layout()
    plt.show()


def visualize_filter(model,img,layer=1):
    '''
    plotting individual filters, still need to flesh this out
    '''
    feature_map = model.conv_block[0:int(2*layer)].forward(img.unsqueeze(0))
    pass

def train_cnn(train_path,val_path,model,transform='default',
              batch_size=10,epochs=5,
              optimizer='sgd',lr=0.005,momentum=0.9,weight_decay=0,
              img_size=(40,40),cmap=None,normalize_plot=False):
    '''
    trains cnn, stores metrics
    train_path: string - path to training data i
                       if train_path is fashion_mnist, mnist or cifar10
                       val_path must be set to "None".
                       
    val_path: string - path to testing data - unless train_path = mnist,fashion_mnist,cifar10
                       in this case val_path should be set to none 
                       
    model: object - an instance of the custom_CNN class
    transform: object - a PyTorch transform object 
    batch_size: int - batch size for the CNN
    lr: float - learning rate for model
    momentum: float - learning rate for model 
    weight_decay: float - weight_decay for model - not recommended if model uses a
                          PReLU activation function 
                          
    img_size: tuple - length x width of image
    augmentation: bool - whether or not to use data augmentation, currently only random transforms work
    '''
    if transform == 'default':
         transf = transforms.Compose([transforms.Resize(size=img_size),
                                      transforms.ToTensor()])
    if transform != 'default':
         transf=transf
         
    if train_path == 'mnist':
        transf = transforms.Compose([transforms.Resize(size=img_size),
                                     transforms.ToTensor(),
                                     #transforms.Normalize((0.1307,), (0.3081,))
                                     ])
        train_set = torchvision.datasets.MNIST(root='.',train=True,transform=transf,download=True)
        train_loader = utils.data.DataLoader(dataset=train_set,
                                             batch_size=batch_size,shuffle=True)
        val_set = torchvision.datasets.MNIST(root='.',train=False,
                                                      transform=transf,download=True)
        val_loader = utils.data.DataLoader(dataset=val_set,
                                             batch_size=batch_size,shuffle=True)
        
    if train_path == 'fashion_mnist':
        transf = transforms.Compose([transforms.Resize(size=img_size),
                                     transforms.ToTensor(),
                                     #transforms.Normalize((0.2860,), (0.3530,))
                                     ])
            
        train_set = torchvision.datasets.FashionMNIST(root='.',train=True,
                                                      transform=transf,download=True)
        train_loader = utils.data.DataLoader(dataset=train_set,
                                             batch_size=batch_size,shuffle=True)
        val_set = torchvision.datasets.FashionMNIST(root='.',train=False,
                                                      transform=transf,download=True)
        val_loader = utils.data.DataLoader(dataset=val_set,
                                             batch_size=batch_size,shuffle=True)
    if train_path == 'cifar10':
        transf = transforms.Compose([transforms.Resize(size=img_size),
                                     transforms.ToTensor(),
                                     #transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     #(0.247, 0.243, 0.261)),
                                     ])
        
        train_set = torchvision.datasets.CIFAR10(root='.',train=True,transform=transf,download=True)
        train_loader = utils.data.DataLoader(dataset=train_set,
                                             batch_size=batch_size,shuffle=True)
        
        val_set = torchvision.datasets.CIFAR10(root='.',train=False,
                                                      transform=transf,download=True)
        val_loader = utils.data.DataLoader(dataset=val_set,
                                             batch_size=batch_size,shuffle=True)
        
    if train_path not in ['cifar10','mnist','fashion_mnist']:
       train_loader = utils.data.DataLoader(dataset=torchvision.datasets.ImageFolder(root=train_path, 
                                     transform=transf),batch_size=batch_size,shuffle=True)
       val_loader = utils.data.DataLoader(dataset=torchvision.datasets.ImageFolder(root=val_path, 
                                     transform=transf),batch_size=batch_size,shuffle=True)
    if optimizer == 'sgd':
        opt = optim.SGD(model.parameters(),lr=lr,momentum=momentum,weight_decay=weight_decay)
    if optimizer == 'adam':
        opt = optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
    if optimizer == 'rmsprop':
        opt = optim.RMSprop(model.parameters(),lr=lr,momentum=momentum,weight_decay=weight_decay)
        

    train_loss = []
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
                batch_true = []
                batch_pred = []
                for i, (data,labels) in enumerate(train_loader):
                    opt.zero_grad()
                    output = model.forward(data)
                    predicted = torch.argmax(output,dim=1)
                    loss = criterion(output,labels)
                    loss.backward()
                    opt.step()
                    print('epoch: {}/{}'.format(epoch+1,epochs))
                    print('batch: {}/{}'.format(i+1,len(train_loader)))
                    print('Loss: {}'.format(loss.item()))
                    #print('predicted: {}'.format(predicted.tolist()))
                    #print('ground truth: {}'.format(labels.tolist()))
                    batch_true.extend(labels.tolist())
                    batch_pred.extend(predicted.tolist())
                    train_loss.append(loss.item())
                    correct = (np.array(batch_true) == np.array(batch_pred)).sum()
                    acc = 100*np.around(correct/len(batch_true),decimals=3)
                    acc = np.format_float_positional(acc,precision=3)
                    acc = float(acc)
                if epoch == epochs-1:
                    train_results = {'predicted': batch_pred,
                               'ground': batch_true,
                               'loss': train_loss,
                               'precision': sk.metrics.precision_score(y_true=batch_true,y_pred=batch_pred,average=None),
                               'recall': sk.metrics.recall_score(y_true=batch_true,y_pred=batch_pred,average=None),
                               'F1': sk.metrics.f1_score(y_true=batch_true,y_pred=batch_pred,average=None),
                               'correct':'{}/{}'.format(correct,len(batch_true)),
                               'accuracy':'{}'.format(acc),
                               'confusion_matrix': sk.metrics.confusion_matrix(y_true=batch_true,y_pred=batch_pred)}
    model = model.eval()
    batch_true = []
    batch_pred = []
    for i, (data,labels) in enumerate(val_loader):
        output = model.forward(data)
        predicted = torch.argmax(output,dim=1)
        loss = F.nll_loss(output,labels)
        print('batch: {}/{}'.format(i+1,len(val_loader)))
        print('Loss: {}'.format(loss.item()))
        print('predicted: {}'.format(predicted.tolist()))
        print('ground truth: {}'.format(labels.tolist()))
        batch_true.extend(labels.tolist())
        batch_pred.extend(predicted.tolist())
        train_loss.append(loss.item())
        correct = (np.array(batch_true) == np.array(batch_pred)).sum()
        acc = 100*np.around(correct/len(batch_true),decimals=3)
        acc = np.format_float_positional(acc,precision=3)
        acc = float(acc)
        test_results = {'predicted': batch_pred,
                               'ground': batch_true,
                               'loss': train_loss,
                               'precision': sk.metrics.precision_score(y_true=batch_true,y_pred=batch_pred,average=None),
                               'recall': sk.metrics.recall_score(y_true=batch_true,y_pred=batch_pred,average=None),
                               'F1': sk.metrics.f1_score(y_true=batch_true,y_pred=batch_pred,average=None),
                               'correct':'{}/{}'.format(correct,len(batch_true)),
                               'accuracy':'{}'.format(acc),
                               'confusion_matrix': sk.metrics.confusion_matrix(y_true=batch_true,y_pred=batch_pred)}
    results = {'train': train_results, 'test': test_results,'model': model.state_dict()}
    return results

def train_autoencoder(path,model,batch_size=10,transform='default',
                      epochs=10,optimizer='sgd',lr=0.001,momentum=0.5,weight_decay=0,
                      img_size=(40,40),cmap=None,normalize_plot=True,
                      save=False):
    if transform == 'default':
         transf = transforms.Compose([transforms.Resize(size=img_size),
                                      transforms.ToTensor()])
    if transform != 'default':
         transf=transf
         
    if path == 'mnist':
        transf = transforms.Compose([transforms.Resize(size=img_size),
                                     transforms.ToTensor(),
                                     #transforms.Normalize((0.1307,), (0.3081,))
                                     ])
        train_set = torchvision.datasets.MNIST(root='.',train=True,transform=transf,download=True)
        train_loader = utils.data.DataLoader(dataset=train_set,
                                             batch_size=batch_size,shuffle=True)
        
    if path == 'fashion_mnist':
        transf = transforms.Compose([transforms.Resize(size=img_size),
                                     transforms.ToTensor(),
                                     #transforms.Normalize((0.2860,), (0.3530,))
                                     ])
            
        train_set = torchvision.datasets.FashionMNIST(root='.',train=True,
                                                      transform=transf,download=True)
        train_loader = utils.data.DataLoader(dataset=train_set,
                                             batch_size=batch_size,shuffle=True)
        
    if path == 'cifar10':
        transf = transforms.Compose([transforms.Resize(size=img_size),
                                     transforms.ToTensor(),
                                     #transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     #(0.247, 0.243, 0.261)),
                                     ])
        
        train_set = torchvision.datasets.CIFAR10(root='.',train=True,transform=transf,download=True)
        train_loader = utils.data.DataLoader(dataset=train_set,
                                             batch_size=batch_size,shuffle=True)
        
    if path not in ['cifar10','mnist','fashion_mnist']:
       train_loader=utils.data.DataLoader(dataset=torchvision.datasets.ImageFolder(root=path, 
                                     transform=transf),batch_size=batch_size,shuffle=True)
    
    if optimizer == 'sgd':
        opt = optim.SGD(model.parameters(),lr=lr,momentum=momentum,weight_decay=weight_decay)
    if optimizer == 'adam':
        opt = optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
    if optimizer == 'rmsprop':
        opt = optim.RMSprop(model.parameters(),lr=lr,momentum=momentum,weight_decay=weight_decay)
        
    criterion = nn.MSELoss()
    decoded_samples = []
    for epoch in range(epochs):
        
        train_loss = []
        for i,(batch,_) in enumerate(train_loader,0):
            opt.zero_grad()
            encoded, decoded = model.forward(batch)
            loss = criterion(decoded,batch)
            loss.backward()
            opt.step()
            print('epoch: {}/{}'.format(epoch+1,epochs))
            print('batch: {}/{}'.format(i+1,len(train_loader)))
            print('loss: {}'.format(loss.item()))
            train_loss.append(loss)
            if save == False:
                plot_batch(tensor=decoded,title='decoded',cmap=cmap,normalize=normalize_plot,
                           save=False)
        if save == True:
            plot_batch(tensor=decoded,title='epoch_{}_decoded'.format(epoch+1),save=True)
            
        
        #decoded_samples.append(im)
    results = {#'decoded': decoded_samples,
               'loss': train_loss,'model': model.state_dict()}
    return results


def train_denoising_autoencoder(path,model,batch_size,epochs,transform='default',
                                optimizer='adam',lr=1e-3,momentum=0.8,weight_decay=0,
                                loss_fun='mse',mean=0,sigma=0.25,
                                img_size=(40,40),cmap=None,normalize_plot=True,
                                plot_samples=True):
    if transform == 'default':
         transf = transforms.Compose([transforms.Resize(size=img_size),
                                      transforms.ToTensor()])
    if transform != 'default':
         transf=transf
         
    if path == 'mnist':
        transf = transforms.Compose([transforms.Resize(size=img_size),
                                     transforms.ToTensor(),
                                     #transforms.Normalize((0.1307,), (0.3081,))
                                     ])
        train_set = torchvision.datasets.MNIST(root='.',train=True,transform=transf,download=True)
        train_loader = utils.data.DataLoader(dataset=train_set,
                                             batch_size=batch_size,shuffle=True)
        
    if path == 'fashion_mnist':
        transf = transforms.Compose([transforms.Resize(size=img_size),
                                     transforms.ToTensor(),
                                     #transforms.Normalize((0.2860,), (0.3530,))
                                     ])
            
        train_set = torchvision.datasets.FashionMNIST(root='.',train=True,
                                                      transform=transf,download=True)
        train_loader = utils.data.DataLoader(dataset=train_set,
                                             batch_size=batch_size,shuffle=True)
        
    if path == 'cifar10':
        transf = transforms.Compose([transforms.Resize(size=img_size),
                                     transforms.ToTensor(),
                                     #transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     #(0.247, 0.243, 0.261)),
                                     ])
        
        train_set = torchvision.datasets.CIFAR10(root='.',train=True,transform=transf,download=True)
        train_loader = utils.data.DataLoader(dataset=train_set,
                                             batch_size=batch_size,shuffle=True)
        
    if path not in ['cifar10','mnist','fashion_mnist']:
       train_loader=utils.data.DataLoader(dataset=torchvision.datasets.ImageFolder(root=path, 
                                     transform=transf),batch_size=batch_size,shuffle=True)
       
    
    if optimizer == 'sgd':
        opt = optim.SGD(model.parameters(),lr=lr,momentum=momentum,weight_decay=weight_decay)
    if optimizer == 'adam':
        opt = optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
    if optimizer == 'rmsprop':
        opt = optim.RMSprop(model.parameters(),lr=lr,momentum=momentum,weight_decay=weight_decay)
    
    if loss_fun == 'mse':
        criterion = nn.MSELoss()
        
    if loss_fun == 'mae':
        criterion = nn.L1Loss()
    
    if loss_fun == 'kl_div':
        criterion = nn.KLDivLoss()
    decoded_samples = []
    
    for epoch in range(epochs):
        train_loss = []
        for i,(batch,_) in enumerate(train_loader,0):
            opt.zero_grad()
            noisy_batch = add_gaussian_noise(batch,mean=mean,std=sigma)
            encoded, decoded = model.forward(noisy_batch) #inputs noisy samples
            loss = criterion(decoded,batch)
            loss.backward() 
            opt.step()
            print('epoch: {}/{}'.format(epoch+1,epochs))
            print('batch: {}/{}'.format(i+1,len(train_loader)))
            print('loss: {}'.format(loss.item()))
            train_loss.append(loss)
            if plot_samples == True:
                plot_batch(tensor=batch,title='samples',cmap=cmap,normalize=normalize_plot)
                plot_batch(tensor=noisy_batch,title='noisy_samples',cmap=cmap,normalize=normalize_plot)
                plot_batch(tensor=decoded,title='decoded_samples',cmap=cmap,normalize=normalize_plot)
        if plot_samples == False:
            plot_batch(tensor=batch,title='epoch_{}_samples'.format(epoch+1),
                       path='autoencoder_output/samples',
                       cmap=cmap,normalize=normalize_plot,
                       save=True)
            plot_batch(tensor=noisy_batch,title='epoch_{}_noisy'.format(epoch+1),
                       path='autoencoder_output/noisy',
                       cmap=cmap,normalize=normalize_plot,
                       save=True)
            plot_batch(tensor=decoded,title='epoch_{}_decoded'.format(epoch+1),
                       path='autoencoder_output/decoded',
                       cmap=cmap,normalize=normalize_plot,
                       save=True)

def train_gan(path,model,noise_dim=3,batch_size=8,epochs=25,img_size=(40,40),
              transform='default',opt_g='sgd',opt_d='sgd',
              lr_g=0.001,lr_d=0.0001,weight_decay_g=0,weight_decay_d=0,mom_g=0.9,mom_d=0.3,
              sigma=0.15,loss_function='bce',cmap=None,normalize_plot=True,plot_batches=False):
    
    if transform == 'default':
        transf = transforms.Compose([transforms.Resize(img_size),transforms.ToTensor()])
       
    if transform != 'default':
        transf=transf
        
    if path == 'mnist':
        train_set = torchvision.datasets.MNIST(root='.',train=True,transform=transf,download=True)
        train_loader = utils.data.DataLoader(dataset=train_set,
                                             batch_size=batch_size,shuffle=True)
    if path == 'fashion_mnist':
        train_set = torchvision.datasets.FashionMNIST(root='.',train=True,transform=transf,download=True)
        train_loader = utils.data.DataLoader(dataset=train_set,
                                             batch_size=batch_size,shuffle=True)
    if path == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(root='.',train=True,transform=transf,download=True)
        train_loader = utils.data.DataLoader(dataset=train_set,
                                             batch_size=batch_size,shuffle=True)
    if path == 'cifar100':
        train_set = torchvision.datasets.CIFAR100(root='.',train=True,transform=transf,download=True)
        train_loader = utils.data.DataLoader(dataset=train_set,
                                             batch_size=batch_size,shuffle=True)
    if path not in ['cifar10','cifar100','mnist','fashion_mnist']:
       train_loader=utils.data.DataLoader(dataset=torchvision.datasets.ImageFolder(root=path, 
                                     transform=transf),batch_size=batch_size,shuffle=True)
    
    D = model['D']
    G = model['G']
    if opt_d == 'sgd':
        optimizerD = optim.SGD(D.parameters(),
                               lr=lr_d,
                               momentum=mom_d,
                               weight_decay=weight_decay_d)
    if opt_d == 'adam':
        optimizerD = optim.Adam(D.parameters(),
                                lr=lr_d,
                                weight_decay=weight_decay_d)
    if opt_d == 'rmsprop':
        optimizerD = optim.RMSprop(D.parameters(),
                                   lr=lr_d,
                                   momentum=mom_d,
                                   weight_decay=weight_decay_d)
        
    if opt_g == 'sgd':
        optimizerG = optim.SGD(G.parameters(),
                               lr=lr_g,
                               momentum=mom_g,
                               weight_decay=weight_decay_g)
    if opt_g == 'adam':
        optimizerG = optim.Adam(G.parameters(),
                                lr=lr_g,
                                weight_decay=weight_decay_g)
    if opt_g == 'rmsprop':
        optimizerG = optim.RMSprop(G.parameters(),
                                   lr=lr_g,
                                   momentum=mom_g,
                                   weight_decay=weight_decay_g)
    
    D.train()
    G.train()
    D_loss_list = []
    G_loss_list = []
    samples = []
    if loss_function == 'mse':
        criterion = nn.MSELoss()
    if loss_function == 'bce':
        criterion = nn.BCELoss()
    if loss_function == 'kl_div':
        criterion = nn.KLDivLoss()
    if loss_function == 'mae':
        criterion = nn.L1Loss()
        
    for epoch in range(epochs):
        for i, (sample, _) in enumerate(train_loader):
            batch_size = sample.size(0)
            input_var = Variable(sample)
            label_real = torch.ones(batch_size)
            label_real_var = Variable(label_real)
            
            prob_vec = Variable(D(input_var))
            D_real_result = torch.max(prob_vec,dim=1)[0]
           
            D_real_loss = criterion(D_real_result,label_real_var.view(1,-1))
            label_fake = torch.zeros(batch_size)
            label_fake_var = Variable(label_fake)
            noise = truncated_noise(torch.randn(batch_size,noise_dim,32,32),mean=0.5,std=sigma) # for image size 40 x 40
                    
            noise_var = Variable(noise)
            G_result = G(noise_var)
            
            
            prob_vec = D(G_result)
            D_fake_result = torch.max(prob_vec,dim=1)[0]
            D_fake_loss = criterion(D_fake_result, label_fake_var.view(1,-1))
            D_train_loss = D_real_loss + D_fake_loss
            D.zero_grad()
            D_train_loss.backward()
            optimizerD.step()
          
            G_result = G(noise_var)
            prob_vec = D(G_result)
            D_fake_result = torch.max(prob_vec,dim=1)[0]
            G_train_loss = criterion(D_fake_result, label_real_var.view(1,-1))#+(loss_coeffs[chrom]*criterion_l1(G_result,input_var))
            D.zero_grad()
            G.zero_grad()
            G_train_loss.backward()
            optimizerG.step()
            D_loss_list.append(D_train_loss.item())
            G_loss_list.append(G_train_loss.item())
            print('epoch: [{}/{}] | batch: [{}/{}]\n'.format(epoch+1,epochs,i+1,len(train_loader)))
            print('G_Loss: {}  |  D_Loss: {}\n'.format(G_train_loss.item(),D_train_loss.item()))
            if plot_batches == True:
                plot_batch(tensor=G_result,title='Generator',cmap=cmap,save=False)
        if plot_batches == False:
            plot_batch(tensor=G_result,title='epoch_{}'.format(epoch+1),
                       path='gan_output',save=True)
    results = {'G_loss': G_loss_list, 'D_loss': D_loss_list,'G': G.state_dict(),'D': D.state_dict()}
    return results
    
def plot_clf_results(train_path,val_path,results,mode='train',metric='F1'):
    '''
    expects results as the same format as train_cnn_beta
    '''
     
    train_labels = os.listdir(train_path)
    val_labels = os.listdir(val_path)
    
    if mode == 'train':
        color_list = np.random.choice(a=['red','green','blue','royalblue','magenta','pink','black','orange','purple','gold'],
                                      size=len(train_labels),replace=False)
        if metric == 'loss':
            loss_plot = results['train']['loss']
            plt.title('{} Loss'.format(mode))
            plt.plot(np.arange(0,len(loss_plot)),loss_plot)
            plt.xlabel('epochs')
            plt.show()
            
        if metric == 'F1':
           
            if len(train_labels) <= 10:
                plt.title('{}-{} score'.format(mode,metric),fontsize=15)
                plt.bar(x=np.arange(1,len(train_labels)+1,1),height=results['train']['F1'],
                        color=color_list,tick_label=train_labels)
                plt.xticks(np.arange(1,len(train_labels)+1,1))
                plt.show()
            if len(train_labels) > 10: 
                plt.bar(x=np.arange(1,len(train_labels)+1,1),height=results['train']['F1'],color=np.random.choice(color_list))
                plt.xticks(np.arange(1,len(train_labels)+1,1),fontsize=14)
                plt.show()
        
        if metric == 'precision':
           
            if len(train_labels) <= 10:
                plt.title('{}-{} score'.format(mode,metric),fontsize=15)
                plt.bar(x=np.arange(1,len(train_labels)+1,1),height=results['train']['precision'],
                        color=color_list,tick_label=train_labels)
                plt.xticks(np.arange(1,len(train_labels)+1,1))
                plt.show()
            if len(train_labels) > 10: 
                plt.bar(x=np.arange(1,len(train_labels)+1,1),height=results['train']['precision'],color=np.random.choice(color_list))
                plt.xticks(np.arange(1,len(train_labels)+1,1),fontsize=14)
                plt.show()
                
        if metric == 'recall':
           
            if len(train_labels) <= 10:
                plt.title('{}-{} score'.format(mode,metric),fontsize=15)
                plt.bar(x=np.arange(1,len(train_labels)+1,1),height=results['train']['recall'],
                        color=color_list,tick_label=train_labels)
                plt.xticks(np.arange(1,len(train_labels)+1,1))
                plt.show()
            if len(train_labels) > 10: 
                plt.bar(x=np.arange(1,len(train_labels)+1,1),height=results['train']['recall'],color=np.random.choice(color_list))
                plt.xticks(np.arange(1,len(train_labels)+1,1),fontsize=14)
                plt.show()
        
    if mode == 'val':
        color_list = np.random.choice(a=['red','green','blue','royalblue','magenta','pink','black','orange','purple','gold'],
                                      size=len(val_labels),replace=False)
        if metric == 'F1':
           
            if len(val_labels) <= 10:
                plt.title('{}-{} score'.format(mode,metric),fontsize=15)
                plt.bar(x=np.arange(1,len(val_labels)+1,1),height=results['val']['F1'],
                        color=color_list,tick_label=val_labels)
                plt.xticks(np.arange(1,len(val_labels)+1,1))
                plt.show()
            if len(val_labels) > 10: 
                plt.bar(x=np.arange(1,len(val_labels)+1,1),height=results['val']['F1'],color=np.random.choice(color_list))
                plt.xticks(np.arange(1,len(val_labels)+1,1),fontsize=14)
                plt.show()
        
        if metric == 'precision':
           
            if len(val_labels) <= 10:
                plt.title('{}-{} score'.format(mode,metric),fontsize=15)
                plt.bar(x=np.arange(1,len(val_labels)+1,1),height=results['val']['precision'],
                        color=color_list,tick_label=val_labels)
                plt.xticks(np.arange(1,len(val_labels)+1,1))
                plt.show()
            if len(val_labels) > 10: 
                plt.bar(x=np.arange(1,len(val_labels)+1,1),height=results['val']['precision'],color=np.random.choice(color_list))
                plt.xticks(np.arange(1,len(val_labels)+1,1),fontsize=14)
                plt.show()
                
        if metric == 'recall':
           
            if len(val_labels) <= 10:
                plt.title('{}-{} score'.format(mode,metric),fontsize=15)
                plt.bar(x=np.arange(1,len(val_labels)+1,1),height=results['val']['recall'],
                        color=color_list,tick_label=val_labels)
                plt.xticks(np.arange(1,len(val_labels)+1,1))
                plt.show()
            if len(val_labels) > 10: 
                plt.bar(x=np.arange(1,len(val_labels)+1,1),height=results['val']['recall'],color=np.random.choice(color_list))
                plt.xticks(np.arange(1,len(val_labels)+1,1),fontsize=14)
                plt.show()
            
                
        if mode == 'loss':
            loss_plot = results['val']['loss']
            plt.title('{} Loss'.format(mode))
            plt.plot(np.arange(0,len(loss_plot)),loss_plot)
            plt.xlabel('epochs')
            plt.show()
            
def plot_gan_loss(results):
    plt.title('GAN loss',fontsize=15)
    plt.plot(np.arange(0,len(results['D_loss'])),results['D_loss'],'r')
    plt.plot(np.arange(0,len(results['G_loss'])),results['G_loss'],'b')
    plt.xlabel('epochs',fontsize=12)
    plt.legend(('Discriminator','Generator'))
    plt.show()

def transfer_learn_cnn(train_path,val_path,transform='default',
                   batch_size=10,epochs=5,
                   optimizer='sgd',lr=0.005,momentum=0.9,img_size=(40,40),augmentation=False,
                   backbone='vgg13',freeze_layers=20):
    num_classes = len(os.listdir(train_path)) #gets number of classes from training directory 
    
    model = make_transfer_module(num_classes=num_classes,backbone=backbone,num_freeze_layers=freeze_layers)
    results = train_cnn(train_path=train_path,val_path=val_path,
                        model=model,
                        transform=transform,
                        batch_size=batch_size,epochs=epochs,
                        optimizer=optimizer,lr=lr,momentum=momentum,img_size=img_size,augmentation=False)