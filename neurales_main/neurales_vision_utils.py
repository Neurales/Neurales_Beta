# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 22:30:03 2020

@author: XZ-WAVE
"""

import os
import sklearn as sk
from sklearn import manifold
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold
from sklearn import ensemble
import heapq
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.utils as utils
import pandas as pd
from Neurales_2020_main import *
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def add_gaussian_noise(batch,mean=0,std=0.5):
    '''
    expects a 4-D Tensor
    '''
    return batch + (torch.randn(batch.size())*std + mean)

def truncated_noise(tensor, mean=0, std=0.5):
    return (tensor*std + mean)

colors = ['r','g','b','c','m','y','k','orange','purple','brown','salmon','darkorchid']

make_color_list = lambda colors,num_colors: list(np.random.choice(colors,num_colors,replace=False))
get_feature_names = lambda df: list(df.keys())

drop_column = lambda df,name: df.drop(columns=name) #drops a feature/column
drop_row = lambda df,idx: df.drop([idx]) #drops a single row
load_csv = lambda csv_file: pd.read_csv(csv_file) 
load_excel = lambda excel_file: pd.read_excel(excel_file) #reads in excel file
load_json = lambda file: pd.read_json(file) #reads in json file, alpha version

def vgg19_classify(img,class_idx):
    prediction = F.softmax(torchvision.models.vgg19(pretrained=True).forward(img))
    pass
def vgg19_filters_vis(x,layer_choices=[1,2,10,34],cmap=None):
    model = torchvision.models.vgg19(pretrained=True)
    for choice in range(len(layer_choices)):
        print('layer: {}'.format(layer_choices[choice]))
        feature_map = list(model.children())[0][0:layer_choices[choice]].forward(x)
        if feature_map.shape[1] > 256:
            figsize=(50,50)
            rows = 32
            
        if 128 < feature_map.shape[1] <= 256:
            figsize=(40,40)
            rows = 32
            
        if 64 < feature_map.shape[1] <= 128:
            figsize=(26,26)
            rows = 16
            
        if 32 < feature_map.shape[1] <= 64:
            figsize=(18,18)
            rows = 8
                
        if feature_map.shape[1] < 32:
            figsize=(13,13)
            rows = 4
        fig = plt.figure(figsize=figsize)
        rows = rows
        cols = int(feature_map.shape[1]/rows)
        for channel in range(1,(cols*rows)+1):
            img = feature_map[0][channel-1].detach().numpy()
                
                    
            fig.add_subplot(rows,cols,channel)
            #plt.suptitle('Convolution Layer: {}'.format(layer))
            plt.title('channel: {}'.format(channel))
            plt.imshow(img,cmap)
            plt.axis('off')
        plt.tight_layout()
            #plt.savefig('./outputs/{}/epoch_{}_layer{}'.format(model_name,epoch,choice))
        plt.show()
    
def plot_multiple_imgs(imgs,figsize=(10,10),rows=2,cmap=None):
    channels = imgs.shape[1]
    if channels == 3:
        fig = plt.figure(figsize=figsize)
        rows = rows
        cols = int(imgs.shape[0]/rows)
        for channel in range(1,(cols*rows)+1):
            img = imgs[0].permute(1,2,0).numpy()
                    
            fig.add_subplot(rows,cols,channel)
            plt.title('img: {}'.format(channel))
            plt.imshow(img,cmap)
            plt.axis('off')
        plt.tight_layout()
        plt.show()
        
def drop_multiple_rows(df,idxs):
    '''
    this function probably isn't necessary, will get around to fixing
    '''
    indexes_to_keep = set(range(df.shape[0])) - set(idxs)
    df_sliced = df.take(list(idxs))
    return df_sliced

def data_target_parse(filepath,mode='clf'): 
    '''
    loads in a dataframe, and parses it depending on whether or not we are dealing with a classification or regression problem
    in regression, we have the data itself, the target, and the names of the features in the data
    with classification we have all of that PLUS the class names so we have a 4-tuple, whereas the regression mode
    returns a 3-tuple
    '''
    data = load_csv(filepath)
    data = drop_multiple_cols(data)
    df = data 
    df = df.dropna()    
    print(df.keys())
    target_name = str(input('enter in a variable you want to predict'))
    y = df[target_name.strip()] #target
    X = df.drop(columns=target_name.strip(),axis=0) #Data
    if mode == 'clf':
        return X,y, X.keys(), pd.unique(y)
    if mode == 'reg':
        return X,y, X.keys()

    #return tuple X,y X: data, y: target

def class_breakdown(df): #target column from data frame (needs to be in data frame format)
  
    df = df.dropna()
    print(df.keys())
    target_name = str(input('enter in a variable you want to predict'))
    y = df[target_name.strip()]
    X = df.drop(columns=target_name.strip(),axis=0)
    color_list = make_color_list(colors,num_colors=len(pd.unique(y)))
    plt.title('class breakdown')
    plt.xticks(np.arange(0,len(pd.unique(y)),1))
    plt.bar(x=pd.value_counts(y).keys(),height=pd.value_counts(y),align='center',color=color_list)

def class_breakdown_(y): #this one takes just the target column by itself
    color_list = make_color_list(colors,num_colors=len(pd.unique(y)))
    plt.title('class breakdown')
    plt.xticks(np.arange(0,len(pd.unique(y)),1))
    plt.bar(x=pd.value_counts(y).keys(),height=pd.value_counts(y),align='center',color=color_list)
    plt.show()

def img_tensor4plot(imgs,channels=3):
    
    if channels == 1:
        imgs_ = imgs.reshape(imgs.shape[0],imgs.shape[1]*imgs.shape[2]*imgs.shape[3]).detach().numpy()
        
    if channels == 3:
        imgs_ = imgs.reshape(imgs.shape[0],imgs.shape[2],imgs.shape[3],imgs.shape[1]).detach().numpy()
        
    return imgs_
#default MNIST AND CIFAR dataloaders, test out with the plots
#then test it out with convolutional filters in different layers
#pandas dataframe to extract feature names and class labels from data frames with loaded csv's
#---------already have these functions ----------------------------
    

flatten_img_tensor = lambda imgs: imgs.reshape(imgs.shape[0],imgs.shape[1]*imgs.shape[2]*imgs.shape[3]).detach().numpy()


sample_labels=['class 1','class 2','class 3','class 4']
'''
tsne_embedding = tsne_transform(X,dim=2)
isomap_embedding = isomap_transform(X,dim=2)
pca_embedding = pca_transform(X,n=3)
'''

def channel_tsne(imgs,tsne_dim=2):
    '''
    tsne visualization for single channel
    '''
    feature_maps = imgs[:,1].detach().numpy()
    feature_maps = feature_maps.reshape(feature_maps.shape[0],feature_maps.shape[1]*feature_maps.shape[2])
    channel_tsne = tsne_transform(feature_maps,dim=tsne_dim)
    return channel_tsne

def multi_channel_tsne(imgs,tsne_dim=2):
    '''
    tsne visualization for all channels
    '''
    feature_maps = imgs.reshape(imgs.shape[0],imgs.shape[1]*imgs.shape[2]*imgs.shape[3]).detach().numpy()
    multi_channel_tsne = tsne_transform(feature_maps,dim=tsne_dim)
    return multi_channel_tsne


def torch_imgs_class_embedding(imgs,targets,labels,embedding='tsne',dim=2,
                               title='Class Clusters',plot_trustworthiness='on'):
    '''
    torch tensors for images are 4-D so we combined all the channels into one block, we want to get them
    in a format that is suitable for plotting
    imgs: a 4-D Tensor of shape num_samples (i.e. batch size) x channels x height x width
    targets: the target dats
    labels: class labels
    '''
    flattened_imgs = flatten_img_tensor(imgs)
    targets = targets.detach().numpy()
    
    if embedding == 'tsne':
        imgs_embedding = tsne_transform(X=flattened_imgs,dim=dim)
        plot_class_scatter(X=flattened_imgs,X_embedded=imgs_embedding,y=targets,labels=labels)
        
    if embedding == 'lle':
        imgs_embedding = lle_transform(X=flattened_imgs,dim=dim)
        plot_class_scatter(X=flattened_imgs,X_embedded=imgs_embedding,y=targets,labels=labels)
        
    if embedding == 'isomap':
        imgs_embedding = isomap_transform(X=flattened_imgs,dim=dim)
        plot_class_scatter(X=flattened_imgs,X_embedded=imgs_embedding,y=targets,labels=labels)
        
    if embedding == 'pca':
        components = int(input('enter in the number of principal components'))
        pca_class_scatter(X=flattened_imgs,y=targets,dim=dim,n=components,labels=labels,colors=colors)
    
    
def plot_class_scatter(X,X_embedded,y,labels,colors=colors,title='Class Clusters',plot_trustworthiness='on'):
    '''
    plots a 2D or 3D embedding of the data. Each class has its own color, each data point is mapped
    to a color depending on which class the data point belongs to
    
    X: a 2D numpy array of num_samples x num_features
    X_embedded: a 2D numpy array of shape (num_samples x 2) for 2D embeddings and (num_samples x 3) for 3D embeddings
    y: the targets
    labels: the class labels corresponding to the targets
    '''
   
    dim = X_embedded.shape[1]
    y = encode_labels(y)
    if dim == 2:
        
        num_classes = len(labels)
        colors = make_color_list(colors,num_classes)
        for i,c,label in zip(range(len(labels)),colors,labels):
            plt.scatter(X_embedded[y == i, 0], X_embedded[y == i, 1], c=c, label=label)
            
        if plot_trustworthiness == 'on':
            t_score = sk.manifold.trustworthiness(X,X_embedded)
            plt.title('{}\n Trustworthiness: {:4f}'.format(title,t_score))
            
        if plot_trustworthiness == 'off':
            plt.title('{}'.format(title))
            
        plt.legend()
        plt.show()
    
    if dim == 3:
        
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111,projection='3d')
        for i,c,label in zip(range(len(labels)),colors,labels):
            ax.scatter(X_embedded[y == i,0],X_embedded[y == i,1],X_embedded[y == i,2],c=c,label=label)
        if plot_trustworthiness == 'on':
            t_score = sk.manifold.trustworthiness(X,X_embedded)
            plt.title('{}\n Trustworthiness: {:4f}'.format(title,t_score))
        if plot_trustworthiness == 'off':
            plt.title('{}'.format(title))
     
        plt.legend()
        plt.show()

def plot_batch(tensor,title='plot',
               path='.',figsize=(20,20),
               cmap=None,normalize=True,
               save=True):
    if os.path.exists(path) == False:
        os.makedirs('./{}'.format(path))
    
    batch_size = tensor.shape[0]
    channels = tensor.shape[1]
    fig = plt.figure(figsize=figsize)
    rows = 1
    if batch_size % 2 == 0:
        rows = 2
    if batch_size % 5 == 0:
        rows = 5
    if batch_size % 6 == 0:
        rows = 3
    if batch_size % 10 == 0:
        rows = 5
    if batch_size % 12 == 0:
        rows = 6
    if batch_size % 16 == 0:
        rows = 8
    cols = batch_size/rows
    cols = int(cols)
    for j in range(1,batch_size+1):
        fig.add_subplot(rows,cols,j)
        plt.title('{} sample:{}'.format(title,j))
        if channels == 3:
            img = tensor[j-1].permute(1,2,0).detach().numpy()
        if channels == 1:
            img = tensor[j-1][0].detach().numpy()
        if normalize == True:
            im = (img*255).astype(np.uint8)
        if normalize == False:
            im = img
        plt.imshow(im,cmap=cmap)
        plt.axis('off')
        plt.tight_layout()
    if save == True:
        plt.savefig('./{}/{}.png'.format(path,title))
        plt.close()
    if save == False:
        plt.show()
        plt.close()

def plot_fcnn_out(feature_map,batch_idx=0,title='feature map',cmap='gray',
                  figsize=(20,20),normalize=True):
    batch_size = feature_map.shape[0]
    channels = feature_map.shape[1]
    fig = plt.figure(figsize=figsize)
    rows = 1
    if channels % 2 == 0:
        rows = 2
    if channels % 5 == 0:
        rows = 5
    if channels % 6 == 0:
        rows = 3
    if channels % 10 == 0:
        rows = 5
    if channels % 12 == 0:
        rows = 6
    if channels % 16 == 0:
        rows = 8
    cols = channels/rows
    cols = int(cols)
    for j in range(1,channels+1):
        fig.add_subplot(rows,cols,j)
        plt.title('{} channel: {}'.format(title,j))
        img = F.softmax(feature_map[batch_idx][j-1],dim=0).detach().numpy()
        plt.imshow(img,cmap=cmap)
        plt.axis('off')
        plt.tight_layout()
    plt.show()
    plt.cla()
    
def pca_class_scatter(X,y,dim,n,labels):
    '''
    dim can only be 2 or 3, but the number of components has to dim <= pca_componentns <= X.shape[1] 
    We can't make an embedding if we have fewer components than the dimension, and the max number of components
    are equal to the number of possible features
    '''
    title = 'PCA Class clusters {}/{} components'.format(n,X.shape[1])
    pca_embedded = pca_transform(X,n)
    X_embedded = pca_embedded[:,0:dim]
    
    plot_class_scatter(X=X,X_embedded=X_embedded,y=y,labels=labels,colors=colors,title=title)
    pass



def visualize_scatter_with_images(X_2d_data, images, figsize=(12,12), image_zoom=1.75,title='Class Scatter'):
    '''
    X_2d_data is an embedding such as TSNE
    expects images in the following two shapes:
        
    3-D np.array of the form [num_imgs x height x  width] 
    4-D np.array of the form [num_imgs x height x width x 3] for three channel images
    '''
    fig, ax = plt.subplots(figsize=figsize)
    artists = []
    for xy, i in zip(X_2d_data, images):
        x0, y0 = xy
        img = OffsetImage(i, zoom=image_zoom)
        ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(X_2d_data)
    ax.autoscale()
    plt.title(title,fontsize=20)
    plt.show()

def tensor_imgs_visualization_2d(imgs,targets,labels,embedding='tsne',figsize=(12,12),
                                 image_zoom=1.75):
    
    flattened_imgs = flatten_img_tensor(imgs)
    targets = targets.detach().numpy()
    images = imgs.permute(0,2,3,1).detach().numpy()
    
    if embedding == 'tsne':
        imgs_embedding = tsne_transform(X=flattened_imgs,dim=2)
        visualize_scatter_with_images(X_2d_data=imgs_embedding,images=images,
                                      figsize=figsize,image_zoom=image_zoom)
    if embedding == 'lle':
        imgs_embedding = lle_transform(X=flattened_imgs,dim=2)
        visualize_scatter_with_images(X_2d_data=imgs_embedding,images=images,
                                      figsize=figsize,image_zoom=image_zoom)
    if embedding == 'isomap':
        imgs_embedding = isomap_transform(X=flattened_imgs,dim=2)
        visualize_scatter_with_images(X_2d_data=imgs_embedding,images=images,
                                      figsize=figsize,image_zoom=image_zoom)
    if embedding == 'pca':
        components = int(input('enter in the number of principal components'))
        imgs_embedding = pca_transform(X=flattened_imgs,n=components)
        visualize_scatter_with_images(X_2d_data=imgs_embedding[:,0:2],images=images,
                                      figsize=figsize,image_zoom=image_zoom)
def visualize_layers(x,model,model_choice='FCNN',epoch=None,cmap=None):
    if model_choice == 'FCNN':
       layer_choices = [0,2,4,6,8]
       model_name = model_choice
       
    if model_choice == 'CNN':
        layer_choices = [0,2,4,6]
        model_name = model_choice
        
    for choice in range(len(layer_choices)):
        print('layer: {}'.format(layer_choices[choice]))
        feature_map = list(model.children())[0][0:layer_choices[choice]].forward(x)
        if feature_map.shape[1] > 256:
            figsize=(50,50)
            rows = 32
            
        if 128 < feature_map.shape[1] <= 256:
            figsize=(40,40)
            rows = 32
            
        if 64 < feature_map.shape[1] <= 128:
            figsize=(26,26)
            rows = 16
            
        if 32 < feature_map.shape[1] <= 64:
            figsize=(18,18)
            rows = 8
                
        if feature_map.shape[1] < 32:
            figsize=(13,13)
            rows = 4
        fig = plt.figure(figsize=figsize)
        rows = rows
        cols = int(feature_map.shape[1]/rows)
        for channel in range(1,(cols*rows)+1):
            img = feature_map[0][channel-1].detach().numpy()
                
                    
            fig.add_subplot(rows,cols,channel)
            #plt.suptitle('Convolution Layer: {}'.format(layer))
            plt.title('channel: {}'.format(channel))
            plt.imshow(img,cmap)
            plt.axis('off')
        plt.tight_layout()
            #plt.savefig('./outputs/{}/epoch_{}_layer{}'.format(model_name,epoch,choice))
        plt.show()
    

def visualize_conv_layer(model,x,chrom,epoch,batch,layer=1,figsize=(13,13),name=None):
#    import pdb
#    pdb.set_trace()
    layers = [*model.children()][0:layer]
    feature_map = nn.Sequential(*layers).forward(x)
    fig = plt.figure(figsize=figsize)
    if feature_map.shape[1] % 2 == 0:
        cols = 2
    if feature_map.shape[1] % 4 == 0:
        cols = 4
    if feature_map.shape[1] % 6 == 0:
        cols = 6
    if feature_map.shape[1] % 8 == 0:
        cols = 8 
    
    
    rows = int(feature_map.shape[1]/cols)
    for channel in range(1,(cols*rows)+1):
        img = feature_map[0][channel-1].detach().numpy()
        fig.add_subplot(rows,cols,channel)
        #plt.suptitle('Convolution Layer: {}'.format(layer))
        plt.title('channel: {}'.format(channel))
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('./neuroevolution_outputs/GAN/filters/{}_chrom{}epoch{}batch{}.png'.format(name,chrom,epoch,batch))
    plt.clf()
    plt.close()




class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        self.conv_block = nn.Sequential(nn.Conv2d(in_channels=3,out_channels=10,kernel_size=3),
                                   nn.ReLU())
        
        self.fc_block = nn.Linear(20,4)
    def forward(self,x):
        self.conv_map = self.conv_block(x)
        self.conv_vector = self.conv_map.view(-1,self.conv_map.shape[1]*self.conv_map.shape[2]*self.conv_map.shape[3])
        self.feature_map = nn.Linear(in_features=self.conv_vector.shape[1],
                                         out_features=self.fc_block.in_features).forward(self.conv_vector)
        
        return F.softmax(self.feature_map,dim=1)

class custom_CNN(nn.Module): #full custom, with details, we have an alternative faster one
    def __init__(self,num_conv_layers,num_fc_layers,nc=3):
        super(custom_CNN,self).__init__()
        self.conv_layers = []
        self.fc_layers = []
        for i in range(num_conv_layers):
            filters = int(input('enter in the number of channels/filters for layer {}/{}'.format(i+1,num_conv_layers)))
            kernel = int(input('enter in a kernel size: for layer {}/{}'.format(i+1,num_conv_layers)))
            stride_opt = str(input('use default stride Y/N?')).strip()
            
            if stride_opt == 'N':
                stride = int(input('enter in a stride for layer {}/{}'.format(i+1,num_conv_layers)))
                
            if stride_opt == 'Y':
                stride = 1
            pad_opt = str(input('use default padding? Y/N?')).strip()
            
            if pad_opt == 'N':
                padding = int(input('enter in a padding for layer {}/{}').format(i+1,num_conv_layers))
            if pad_opt == 'Y':
                padding = 1
                if i == 0:
                   self.conv_layers.append(nn.Conv2d(in_channels=nc,out_channels=filters,kernel_size=kernel,
                                                  stride=stride,padding=padding))
                activation = str(input('Choose activation function: relu | prelu | sigmoid')).strip() #drop down menu? We can add all the torch activation functions
                if activation == 'relu':
                    self.conv_layers.append(nn.ReLU())
                if activation == 'prelu':
                    all_channels = str(input('use parameter for each channel Y/N?')).strip()
                    if all_channels == 'Y':
                        self.conv_layers.append(nn.PReLU(num_parameters=filters))
                    if all_channels == 'N':
                        self.conv_layers.append(nn.PReLU())
                if activation == 'sigmoid':
                    self.conv_layers.append(nn.Sigmoid())
                use_BN = str(input('use batch norm Y/N?')).strip()
                if use_BN == 'Y':
                    self.conv_layers.append(nn.BatchNorm2d(num_features=filters))
            if i > 0:
                if use_BN == 'N':
                    self.conv_layers.append(nn.Conv2d(in_channels=self.conv_layers[i-1].out_channels,
                                                      out_channels=filters,kernel_size=kernel,stride=stride,padding=padding))
                if use_BN == 'Y':
                    self.conv_layers.append(nn.Conv2d(in_channels=self.conv_layers[i-2].num_features,
                                                      out_channels=filters,kernel_size=kernel,stride=stride,padding=padding))
                    
                    self.conv_layers.append(nn.BatchNorm2d(num_features=self.conv_layers[i-1].out_channels))
                activation = str(input('Choose activation function: relu | prelu | sigmoid')).strip() #drop down menu? We can add all the torch activation functions
                if activation == 'relu':
                    self.conv_layers.append(nn.ReLU())
                if activation == 'prelu':
                    all_channels = str(input('use parameter for each channel Y/N?')).strip()
                    if all_channels == 'Y':
                        self.conv_layers.append(nn.PReLU(num_parameters=filters))
                    if all_channels == 'N':
                        self.conv_layers.append(nn.PReLU())
                if activation == 'sigmoid':
                    self.conv_layers.append(nn.Sigmoid())
                use_BN = str(input('use batch norm Y/N?')).strip()
        self.conv_block = nn.Sequential(*self.conv_layers)
        for j in range(num_fc_layers):
            pass
                    
        
    def forward(self,x):
        self.conv_map = self.conv_block.forward(x)
        self.conv_vector = self.conv_vector = self.conv_map.view(-1,
                                                                 self.conv_map.shape[1]*self.conv_map.shape[2]*self.conv_map.shape[3])
        
        self.feature_map = nn.Linear(in_features=self.conv_vector.shape[1],
                                         out_features=self.fc_block[0].in_features).forward(self.conv_vector)
        self.fc_map = self.fc_block(self.feature_map)
        return F.softmax(self.fc_map,dim=1)
                

def plot_imgs(data_loader,plot_batches=True):
    dataloader = list(enumerate(data_loader))
    for sample in range(len(dataloader)):
        plt.title('sample: {}'.format(sample+1))
        plt.imshow(dataloader[0][1][0][sample].permute(1,2,0).detach().numpy())
        plt.show()
    
    pass
    
def make_transfer_layer(backbone='vgg13',num_classes=10):
    if backbone == 'vgg13':
        model = models.vgg13(pretrained=True)
        model.classifier[-1] = nn.Linear(in_features=4096,out_features=num_classes)
    return model

def freeze_layers(model,num_layers):
    for param in model.parameters():
        for layer in range(num_layers):
            list(model.parameters())[layer].requires_grad = False
    return model 

def make_transfer_module(num_classes,backbone='vgg13',num_freeze_layers=20):
    transfer_model = make_transfer_layer(backbone=backbone,num_classes=num_classes)
    transfer_model = freeze_layers(model=transfer_model,num_layers=num_freeze_layers)
    return transfer_model 
    


    
    