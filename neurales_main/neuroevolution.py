# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 20:18:42 2020

@author: XZ-WAVE
"""

import tqdm
import time
import sklearn as sk
from sklearn import metrics
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import torchvision.transforms as transforms
import torch.optim as optim 
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import os
from abc import ABCMeta, abstractmethod 
from neurales_vision_utils import *
from Neurales_2020_main import *


def cnn_clf_fitness(model,f1_score):
    num_layers = len(model.conv_network)
    L_score = 1/num_layers
    fitness = np.tanh(f1_score+L_score)+0.095
    return fitness
    

def gan_fitness(G,D,D_loss,G_loss,loss_coeff=0.75,param_coeff=0.25):
    D_params = sum(p.numel() for p in D.parameters() if p.requires_grad)
    G_params = sum(p.numel() for p in G.parameters() if p.requires_grad)
    params = D_params + G_params
    p_score = (2*10*6)/params
    model_fitness = loss_coeff*np.tanh(0.01*((1/(G_loss+1e-7))+(1/(D_loss+1e-7))))
    param_fitness = param_coeff*np.tanh(p_score)
    fitness = model_fitness + param_fitness
    return fitness


def dcgan_fitness(genome,D_loss,G_loss,loss_coeff=0.75,param_coeff=0.25):
    g_filters = genome['G_filters']
    d_filters = genome['D_filters']
    num_g_layers = len(g_filters)
    num_d_layers = len(d_filters)
    sum_d_filters = sum(d_filters)
    sum_g_filters = sum(g_filters)
    params_score = np.tanh((100/(num_d_layers+sum_d_filters))+(100/(num_g_layers+sum_g_filters)))
    loss_score = np.tanh(0.01*((1/(G_loss+1e-7))+(1/(D_loss+1e-7))))
    fitness = (loss_coeff*loss_score)+(param_coeff*params_score)
    return fitness 

class GeneticCNN(nn.Module):
    def __init__(self,genome,nc=3,num_classes=10):
        super(GeneticCNN,self).__init__()
        self.genome = genome
        self.num_conv_layers = len(self.genome['filters'])
        self.num_fc_layers = len(self.genome['hidden_sizes'])
        self.conv_layers = [nn.Conv2d(in_channels=nc,out_channels=self.genome['filters'][0],
                                      kernel_size=self.genome['kernels'][0]),
                            nn.ReLU()]
        
        for layer in range(self.num_conv_layers-1):
            self.conv_layers.append(nn.Conv2d(in_channels=self.genome['filters'][layer],
                                              out_channels=self.genome['filters'][layer+1],
                                              kernel_size=self.genome['kernels'][layer]))
            if self.genome['norms'][layer] == 'batch':
                self.conv_layers.append(nn.BatchNorm2d(self.genome['filters'][layer+1]))
            if self.genome['norms'][layer] == 'instance':
                self.conv_layers.append(nn.InstanceNorm2d(self.genome['filters'][layer+1]))
            if self.genome['activations'][layer] == 'prelu':
                self.conv_layers.append(nn.PReLU(num_parameters=self.genome['filters'][layer+1]))
            if self.genome['activations'][layer] == 'relu':
                self.conv_layers.append(nn.ReLU())
            if self.genome['activations'][layer] == 'selu':
                self.conv_layers.append(nn.SELU())
            if self.genome['dropouts'][layer] != 'none':
                self.conv_layers.append(nn.Dropout(p=float(self.genome['dropouts'][layer])))
                
        self.width = self.genome['adaptive_size'][0]
        self.length = self.genome['adaptive_size'][1]
        if self.genome['adaptive_type'] == 'max':
            self.conv_layers.append(nn.AdaptiveMaxPool2d(output_size=(self.width,self.length)))
        if self.genome['adaptive_type'] == 'avg':
            self.conv_layers.append(nn.AdaptiveAvgPool2d(output_size=(self.width,self.length)))
        self.fc_layers = [nn.Linear(in_features=self.width*self.length*self.genome['filters'][-1],
                                    out_features=self.genome['hidden_sizes'][0]),nn.ReLU()]
        for layer in range(self.num_fc_layers-1):
            self.fc_layers.append(nn.Linear(in_features=self.genome['hidden_sizes'][layer],
                                            out_features=self.genome['hidden_sizes'][layer+1]))
            self.fc_layers.append(nn.ReLU())
        self.fc_layers.append(nn.Linear(in_features=self.genome['hidden_sizes'][-1],
                                        out_features=num_classes))
       
        self.conv_network = nn.Sequential(*self.conv_layers)
        self.fc_network = nn.Sequential(*self.fc_layers)
        
    def forward(self,x):
        self.conv_map = self.conv_network.forward(x)
        self.conv_map = self.conv_map.view(-1,
                                           self.genome['filters'][-1]*self.width*self.length)
    
        self.fc_map = self.fc_network.forward(self.conv_map)
        return self.fc_map 
            
def make_genetic_cnns(num_chroms=10,
                      nc=3,
                      num_classes=10,
                      max_epochs=50):
    chroms = []
    genomes = []
    activation_choices = ['relu','prelu','selu']
    norm_choices = ['batch','instance','none']
    filter_choices = [2,4,8,16,32,64,128,256,512]
    kernel_sizes = [1,3,5]
    dropouts = ['none',0.1,0.2,0.25,0.5]
    opt_choices = ['sgd','adam','rmsprop']
    lr_choices = [1e-4,2e-4,5e-4,1e-3,2e-3,1e-3,5e-3,1e-2]
    momentum_choices = [0.4,0.5,0.75,0.8,0.9]
    adaptive_choices = ['max','avg']
    adaptive_out_choices = [3,4,5,6,7,8,9,10]
    hidden_sizes = [25,50,100,250,500,1000]
    epoch_choices = [2,3,5,10,15,20,max_epochs]
    num_convs = [2,3,4,5,6,7,8,9]
    num_fcs = [2,3,4,5]
    for chrom in range(num_chroms):
        adaptive_size = np.random.choice(a=adaptive_out_choices)
        num_conv_layers = np.random.choice(a=num_convs)
        num_fc_layers = np.random.choice(a=num_fcs)
        genome = {'filters': list(np.random.choice(a=filter_choices,size=num_conv_layers)),
              'kernels': list(np.random.choice(a=kernel_sizes,size=num_conv_layers,p=[0.1,0.7,0.2])),
              'dropouts': list(np.random.choice(a=dropouts,size=num_conv_layers,p=[0.75,0.1,0.05,0.05,0.05])),
              'norms': list(np.random.choice(a=norm_choices,size=num_conv_layers,p=[0.375,0.125,0.5])),
              'adaptive_type': np.random.choice(a=adaptive_choices),
              'adaptive_size': (adaptive_size,adaptive_size),
              'hidden_sizes': list(np.random.choice(a=hidden_sizes,size=num_fc_layers)),
              'activations': list(np.random.choice(a=activation_choices,size=num_conv_layers,p=[0.1,0.8,0.1])),
              'optimizer': np.random.choice(a=opt_choices),
              'lr': np.random.choice(a=lr_choices),
              'momentum': np.random.choice(a=momentum_choices),
              'epochs': np.random.choice(a=epoch_choices)}
        genomes.append(genome)
    for chrom in range(num_chroms):
        chroms.append(GeneticCNN(genome=genomes[chrom],nc=nc,num_classes=num_classes))
    return chroms,genomes 

def crossover_and_mutate_cnn(genomes,
                         num_parents=4,
                         num_mutations=2,
                         num_classes=10,
                         nc=3,
                         max_epochs=50):
    #==========CROSSOVER=========================#
    parents = genomes[:num_parents]
    children = []
    chroms = []
    
    activation_choices = ['relu','prelu','selu']
    norm_choices = ['batch','instance','none']
    filter_choices = [2,4,8,16,32,64,128,256,512]
    kernel_sizes = [1,3,5]
    dropouts = ['none',0.1,0.2,0.25,0.5]
    opt_choices = ['sgd','adam','rmsprop']
    lr_choices = [1e-4,2e-4,5e-4,1e-3,2e-3,1e-3,5e-3,1e-2]
    momentum_choices = [0.4,0.5,0.75,0.8,0.9]
    adaptive_choices = ['max','avg']
    adaptive_out_choices = [3,4,5,6,7,8,9,10]
    hidden_sizes = [25,50,100,250,500,1000]
    epoch_choices = [2,3,5,10,15,20,max_epochs]
    num_convs = [2,3,4,5,6,7,8,9]
    num_fcs = [2,3,4,5]
    for p in range(len(parents)-1):
        offspring_filters = parents[p]['filters']
        offspring_kernels = parents[p]['kernels']
        offspring_dropouts = parents[p]['dropouts']
        offspring_norms = parents[p]['norms']
        offspring_adaptive_type = parents[p]['adaptive_type']
        offspring_adaptive_size = parents[p]['adaptive_size']
        offspring_hidden_sizes = parents[p]['hidden_sizes']
        offspring_activations = parents[p]['activations']
        offspring_optimizer = parents[p+1]['optimizer']
        offspring_lr = parents[p+1]['lr']
        offspring_mom = parents[p+1]['momentum']
        offspring_epochs = parents[p+1]['epochs']
        child = {'filters': offspring_filters,
                 'kernels': offspring_kernels,
                 'dropouts': offspring_dropouts,
                 'norms': offspring_norms,
                 'adaptive_type': offspring_adaptive_type,
                 'adaptive_size': offspring_adaptive_size,
                 'hidden_sizes': offspring_hidden_sizes,
                 'activations': offspring_activations,
                 'optimizer': offspring_optimizer,
                 'lr': offspring_lr,
                 'momentum': offspring_mom,
                 'epochs': offspring_epochs}
        children.append(child)
    #==========MUTATE============================#
    #this code is inefficient====================#
    for m in range(num_mutations):
        
        for c in range(len(children)):
            gene = np.random.choice(a=list(child.keys()))
            if gene == 'filters':
                children[c]['filters'] = list(np.random.choice(a=filter_choices,
                        size=len(children[c]['filters'])))
            if gene == 'kernels':
                children[c]['kernels'] == list(np.random.choice(a=kernel_sizes,
                        size=len(children[c]['kernels'])))
            if gene == 'dropouts':
                children[c]['dropouts'] == list(np.random.choice(a=dropouts,
                        size=len(children[c]['dropouts'])))
            if gene == 'norms':
                children[c]['norms'] == list(np.random.choice(a=norm_choices,
                        size=len(children[c]['norms'])))
            if gene == 'adaptive_type':
                if children[c]['adaptive_type'] == 'max':
                    children[c]['adaptive_type'] = 'avg'
                    
                if children[c]['adaptive_type'] == 'avg':
                    children[c]['adaptive_type'] = 'max'
            if gene == 'adaptive_size':
                size = np.random.choice(a=adaptive_out_choices)
                children[c]['adaptive_size'] == (size,size)
            if gene == 'hidden_sizes':
                children[c]['hidden_sizes'] == list(np.random.choice(a=hidden_sizes,
                        size=len(children[c]['hidden_sizes'])))
            if gene == 'activations':
                children[c]['activations'] = list(np.random.choice(a=activation_choices,
                        size=len(children[c]['activations'])))
            if gene == 'optimizer':
                children[c]['optimizer'] = np.random.choice(a=opt_choices)
            if gene == 'lr':
                children[c]['lr'] = np.random.choice(a=lr_choices)
            if gene == 'momentum':
                children[c]['momentum'] = np.random.choice(a=momentum_choices)
            if gene == 'epochs':
                children[c]['epochs'] = np.random.choice(a=epoch_choices)
    
    for c in range(len(children)):
        model = GeneticCNN(genome=children[c],nc=nc,num_classes=num_classes)
        chroms.append(model)
    new_chroms, new_genomes = make_genetic_cnns(num_chroms=len(genomes)-len(chroms),
                                                nc=nc,
                                                num_classes=num_classes,
                                                max_epochs=max_epochs)
    offspring = children
    chroms.extend(new_chroms) #always come at the "tail" of the list 
    children.extend(new_genomes) #always come at the "tail" of the list 
    parents_and_offspring = {'parents': parents, 'offspring': offspring}
    return chroms, children, parents_and_offspring
        
def genetic_train_cnn(train_path,val_path,
                      num_generations=20,
                      num_chroms=20,
                      elite_frac=0.2,
                      nc=3,
                      transform='default',
                      img_size=(40,40),
                      batch_size=10,
                      target_score=0.95,
                      metric='F1',
                      max_loss_thresh=1e7,
                      evolution_mode='elitist',
                      num_mutations=1,
                      num_parents=4,
                      max_epochs=50):
    
    
    if transform == 'default':
         transf = transforms.Compose([transforms.Resize(size=img_size),
                                      transforms.ToTensor()])
    if transform != 'default':
         transf=transf
         
    if train_path == 'mnist':
        nc = 1 #1 channel images
        num_classes = 10 
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
        nc = 1 #1 channel images
        num_classes = 10
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
        nc = 3 #3 channel images
        num_classes = 10 
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
    if train_path == 'cifar100':
        nc = 3 #3 channel images
        num_classes = 100
        transf = transforms.Compose([transforms.Resize(size=img_size),
                                     transforms.ToTensor(),
                                     #transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     #(0.247, 0.243, 0.261)),
                                     ])
        
        train_set = torchvision.datasets.CIFAR100(root='.',train=True,transform=transf,download=True)
        train_loader = utils.data.DataLoader(dataset=train_set,
                                             batch_size=batch_size,shuffle=True)
        
        val_set = torchvision.datasets.CIFAR100(root='.',train=False,
                                                      transform=transf,download=True)
        val_loader = utils.data.DataLoader(dataset=val_set,
                                             batch_size=batch_size,shuffle=True)
    if train_path == 'imagenet':
        nc = 3 #3 channel images
        num_classes = 1000
        transf = transforms.Compose([transforms.Resize(size=img_size),
                                     transforms.ToTensor(),
                                     #transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     #(0.247, 0.243, 0.261)),
                                     ])
        
        train_set = torchvision.datasets.ImageNet(root='.',train=True,transform=transf,download=True)
        train_loader = utils.data.DataLoader(dataset=train_set,
                                             batch_size=batch_size,shuffle=True)
        
        val_set = torchvision.datasets.ImageNet(root='.',train=False,
                                                      transform=transf,download=True)
        val_loader = utils.data.DataLoader(dataset=val_set,
                                             batch_size=batch_size,shuffle=True)
        
    if train_path not in ['imagenet','cifar100','cifar10','mnist','fashion_mnist']:
       nc=3 
       num_classes = len(os.listdir(train_path)) 
       
       train_loader = utils.data.DataLoader(dataset=torchvision.datasets.ImageFolder(root=train_path, 
                                     transform=transf),batch_size=batch_size,shuffle=True)
       val_loader = utils.data.DataLoader(dataset=torchvision.datasets.ImageFolder(root=val_path, 
                                     transform=transf),batch_size=batch_size,shuffle=True)
    chroms = make_genetic_cnns(num_chroms=num_chroms,
                               nc=nc,
                               num_classes=num_classes,
                               max_epochs=max_epochs)
    
    max_score = 0 
    max_fitness = []
    for gen in range(num_generations):
        fitness = []
        models_and_results = []
        for chrom in range(num_chroms):
            model = chroms[0][chrom]
            if chroms[1][chrom]['optimizer'] == 'sgd':
                optimizer = optim.SGD(model.parameters(),
                                      lr=chroms[1][chrom]['lr'],
                                      momentum=chroms[1][chrom]['momentum'])
            if chroms[1][chrom]['optimizer'] == 'adam':
                optimizer = optim.Adam(model.parameters(),
                                       lr=chroms[1][chrom]['lr'])
            if chroms[1][chrom]['optimizer'] == 'rmsprop':
                optimizer = optim.RMSprop(model.parameters(),
                                       lr=chroms[1][chrom]['lr'],
                                       momentum=chroms[1][chrom]['momentum'])
            train_loss = []
            criterion = nn.CrossEntropyLoss()
            epochs = chroms[1][chrom]['epochs']
            for epoch in range(epochs): #
                batch_true = []
                batch_pred = []
                for i, (data,labels) in enumerate(train_loader):
                    optimizer.zero_grad()
                    output = model.forward(data)
                    predicted = torch.argmax(output,dim=1)
                    loss = criterion(output,labels)
                    loss.backward()
                    optimizer.step()
                    print('evolution mode: {}'.format(evolution_mode))
                    if evolution_mode == 'elitist':
                        print('elite fraction: {}'.format(elite_frac))
                    if evolution_mode == 'crossover':
                        print('num parents: {}'.format(num_parents))
                        print('num mutations: {}'.format(num_mutations))
                    print('Generation: [{}/{}] | Chromosome: [{}/{}]'.format(gen+1,num_generations,chrom+1,num_chroms))
                    print('Epoch: [{}/{}] | Batch: [{}/{}]'.format(epoch+1,epochs,i+1,len(train_loader)))
                    print('Loss: {:4f}\n'.format(loss.item()))
                    #print('predicted: {}'.format(predicted.tolist()))
                    #print('ground truth: {}'.format(labels.tolist()))
                    batch_true.extend(labels.tolist())
                    batch_pred.extend(predicted.tolist())
                    train_loss.append(loss.item())
                    correct = (np.array(batch_true) == np.array(batch_pred)).sum()
                    acc = 100*np.around(correct/len(batch_true),decimals=3)
                    acc = np.format_float_positional(acc,precision=3)
                    acc = float(acc)
                if max(train_loss) >= abs(max_loss_thresh):
                    print('DANG IT BOBBY!!! The loss too high, aborting training...')
                    train_results = 'aborted'
                    break
                if epoch == epochs-1:
                    train_results = {'predicted': batch_pred,
                               'ground': batch_true,
                               'loss': train_loss,
                               'precision': sk.metrics.precision_score(y_true=batch_true,y_pred=batch_pred,average=None),
                               'recall': sk.metrics.precision_score(y_true=batch_true,y_pred=batch_pred,average=None),
                               'F1': sk.metrics.precision_score(y_true=batch_true,y_pred=batch_pred,average=None),
                               'correct': '{}/{}'.format(correct,len(batch_true)),
                               'accuracy': '{}'.format(acc),
                               'confusion_matrix': sk.metrics.confusion_matrix(y_true=batch_true,y_pred=batch_pred)}
            model = model.eval()
            batch_true = []
            batch_pred = []
            
            for i, (data,labels) in enumerate(val_loader):
                if train_results == 'aborted':
                    print('skipping testing...')
                    test_results = 'aborted'
                    fitness = [0]
                    break
                output = model.forward(data)
                predicted = torch.argmax(output,dim=1)
                loss = criterion(output,labels)
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
                                       'correct': '{}/{}'.format(correct,len(batch_true)),
                                       'accuracy':'{}'.format(acc),
                                       'confusion_matrix': sk.metrics.confusion_matrix(y_true=batch_true,y_pred=batch_pred)}
                
            results = {'train': train_results, 
                       'test': test_results,
                       'model': model,
                       'weights': model.state_dict()}
            if test_results == 'aborted':
                metric_score = 0
            if test_results != 'aborted':
                metric_score = np.mean(results['test']['{}'.format(metric)]) #pulls the appropriate metric
            if metric_score > max_score:
                max_score = metric_score  
            if test_results == 'aborted':
                f1_score = 0
            if test_results != 'aborted':
                f1_score = results['test']['F1']
            score = cnn_clf_fitness(model,f1_score=np.mean(f1_score))
            models_and_results.append({'fitness': score,'results': results,'genome': chroms[1][chrom]})
            fitness.append(score)
            
        max_fitness.append(max(fitness))
        print("Generation {} max {} score: {}".format(gen+1,metric,max_score))
       
        num_elites = int(np.ceil(elite_frac*num_chroms)) 
        fitness_sorted = np.argsort(fitness)
        chroms_sorted = [i for _,i in sorted(zip(fitness_sorted,chroms[0]))]
        genomes_sorted = [i for _,i in sorted(zip(fitness_sorted,chroms[1]))]
        if evolution_mode == 'elitist':
            chroms_sorted = chroms_sorted[::-1]
            genomes_sorted = genomes_sorted[::-1]
            chroms = chroms_sorted[:num_elites]
            genomes = genomes_sorted[:num_elites]
            num_new = num_chroms-num_elites
            new_pop = make_genetic_cnns(num_chroms=num_new,
                                           num_classes=num_classes,
                                           nc=nc,
                                           max_epochs=max_epochs) #returns a 2-element tuple: ([chroms],[genomes])
            
            new_chroms = new_pop[0] #gets the new chromosomes
            new_genomes = new_pop[1] #gets the new genomes 
            chroms.extend(new_chroms)
            genomes.extend(new_genomes)
            pop = (chroms,genomes) 
            chroms = pop
        if evolution_mode == 'crossover':
            chroms_sorted = chroms_sorted[::-1]
            genomes_sorted = genomes_sorted[::-1]
            pop = crossover_and_mutate_cnn(genomes=genomes_sorted,
                         num_parents=num_parents,
                         num_mutations=num_mutations,
                         num_classes=num_classes,
                         nc=nc,
                         max_epochs=max_epochs)
            if len(pop[0]) <= num_chroms: #if a model is aborted during training, this handles that error
                new_pop = make_genetic_cnns(num_chroms=num_chroms-len(pop[0]),
                                            num_classes=num_classes,
                                            nc=nc,
                                            max_epochs=max_epochs)
                pop[0].extend(new_pop[0])
                pop[1].extend(new_pop[1])
            chroms = (pop[0],pop[1])
            parents_and_offspring = pop[2]
        if max_score >= target_score:
            print('Solved! - model with >={} {} score found!'.format(target_score,metric))
            break
    if evolution_mode == 'elitist':
        return models_and_results,max_fitness
    if evolution_mode == 'crossover':
        return models_and_results,max_fitness,parents_and_offspring


class GeneticGenerator(nn.Module):
    def __init__(self,genome):
        super(GeneticGenerator,self).__init__()
        self.genome = genome
        self.num_conv_layers = len(self.genome['G_filters'])
        
        self.conv_layers = [nn.ConvTranspose2d(in_channels=self.genome['noise_dim'],
                                          out_channels=self.genome['G_filters'][0],
                                          kernel_size=self.genome['G_kernels'][0])]
        for layer in range(self.num_conv_layers-1):
            self.conv_layers.append(nn.ConvTranspose2d(in_channels=self.genome['G_filters'][layer],
                                              out_channels=self.genome['G_filters'][layer+1],
                                              kernel_size=self.genome['G_kernels'][layer]))
            if self.genome['G_activations'][layer] == 'prelu':
                self.conv_layers.append(nn.PReLU(num_parameters=self.genome['G_filters'][layer+1]))
            if self.genome['G_activations'][layer] == 'relu':
                self.conv_layers.append(nn.ReLU())
            if self.genome['G_activations'][layer] == 'selu':
                self.conv_layers.append(nn.SELU())
        height = self.genome['img_size'][0]
        width = self.genome['img_size'][1]
        self.conv_layers.append(nn.ConvTranspose2d(in_channels=self.genome['G_filters'][-1],
                                                   out_channels=self.genome['num_channels'],
                                                   kernel_size=1))
        if self.genome['adaptive_type'] == 'max':
            self.conv_layers.append(nn.AdaptiveMaxPool2d(output_size=(height,width)))
        if self.genome['adaptive_type'] == 'avg':
            self.conv_layers.append(nn.AdaptiveAvgPool2d(output_size=(height,width)))
        self.Generator = nn.Sequential(*self.conv_layers)
    
    def forward(self,x):
        return self.Generator.forward(x)

class GeneticDiscriminator(nn.Module):
    def __init__(self,genome):
        super(GeneticDiscriminator,self).__init__()
        self.genome = genome
        self.num_conv_layers = len(self.genome['D_filters'])
        
        self.conv_layers = [nn.Conv2d(in_channels=self.genome['num_channels'],
                                          out_channels=self.genome['D_filters'][0],
                                          kernel_size=self.genome['D_kernels'][0])]
        for layer in range(self.num_conv_layers-1):
            self.conv_layers.append(nn.Conv2d(in_channels=self.genome['D_filters'][layer],
                                              out_channels=self.genome['D_filters'][layer+1],
                                              kernel_size=self.genome['D_kernels'][layer]))
            if self.genome['D_activations'][layer] == 'prelu':
                self.conv_layers.append(nn.PReLU(num_parameters=self.genome['D_filters'][layer+1]))
            if self.genome['D_activations'][layer] == 'relu':
                self.conv_layers.append(nn.ReLU())
            if self.genome['D_activations'][layer] == 'selu':
                self.conv_layers.append(nn.SELU())
        height = self.genome['adaptive_size']
        width = self.genome['adaptive_size']
        if self.genome['adaptive_type'] == 'max':
            self.conv_layers.append(nn.AdaptiveMaxPool2d(output_size=(height,width)))
        if self.genome['adaptive_type'] == 'avg':
            self.conv_layers.append(nn.AdaptiveAvgPool2d(output_size=(height,width)))
        
        self.clf_features = self.genome['D_filters'][-1]*height*width
        self.clf_layer = nn.Linear(in_features=self.clf_features,
                                   out_features=2)
        
        self.Discriminator = nn.Sequential(*self.conv_layers)
        
        
    def forward(self,x):
        self.conv_map = self.Discriminator.forward(x)
        self.conv_map = self.conv_map.reshape(-1,self.clf_features)
        return torch.sigmoid(self.clf_layer.forward(self.conv_map))
        

def make_genetic_dcgans(num_chroms,nc=3,noise_dim=3,
                        symmetric=True,img_size=(40,40),
                        max_epochs=100):
    
    chroms = []
    genomes = []
    filter_choices = [2,4,6,8,12,16,24,32,64,128]
    kernel_choices = [1,3,5]
    activation_choices = ['relu','prelu','selu']
    opt_choices = ['sgd','adam','rmsprop']
    lr_choices = [5e-5,1e-4,2e-4,5e-4,1e-3,2e-3,5e-3]
    momentum_choices = [0.4,0.5,0.75,0.8,0.9]
    epoch_choices = [2,3,5,10,20,max_epochs]
    mean_choices = [0,0.1]
    std_choices = [1e-8,1e-5,1e-4,1e-3,0.01,0.1,0.5]
    adaptive_out_choices = ['max','avg']
    adaptive_out_sizes = [4,5,6,7,8]
    num_convs = [2,3,4,5,6,7]
    for chrom in range(num_chroms):
        num_conv_layers = np.random.choice(a=num_convs)
        noise_size = int(np.sum(img_size)/4)
        if symmetric == False:
            genome = {'D_filters': list(np.random.choice(a=filter_choices,size=num_conv_layers)),
                      'D_kernels': list(np.random.choice(a=kernel_choices,size=num_conv_layers,p=[0.1,0.7,0.2])),
                      'D_activations': list(np.random.choice(a=activation_choices,size=num_conv_layers)),
                      'D_opt': np.random.choice(a=opt_choices),
                      'D_lr': np.random.choice(a=lr_choices),
                      'D_mom': np.random.choice(a=momentum_choices),
                      'G_filters': list(np.random.choice(a=filter_choices,size=num_conv_layers)),
                      'G_kernels': list(np.random.choice(a=kernel_choices,size=num_conv_layers,p=[0.1,0.7,0.2])),
                      'G_activations': list(np.random.choice(a=activation_choices,size=num_conv_layers)),
                      'G_opt': np.random.choice(a=opt_choices),
                      'G_lr': np.random.choice(a=lr_choices),
                      'G_mom': np.random.choice(a=momentum_choices),
                      'epochs': np.random.choice(a=epoch_choices),
                      'img_size': img_size,
                      'noise_dim': noise_dim,
                      'noise_size': noise_size,
                      'adaptive_type': np.random.choice(a=adaptive_out_choices),
                      'adaptive_size': np.random.choice(a=adaptive_out_sizes),
                      'mean': np.random.choice(a=mean_choices),
                      'std': np.random.choice(a=std_choices),
                      'num_channels':nc}
            genomes.append(genome)
            
        if symmetric == True:
            filters = list(np.random.choice(a=filter_choices,size=num_conv_layers))
            kernels = list(np.random.choice(a=kernel_choices,size=num_conv_layers,p=[0.1,0.7,0.2]))
            activations = list(np.random.choice(a=activation_choices,size=num_conv_layers))
            genome = {'D_filters': filters,
                      'D_kernels': kernels,
                      'D_activations': activations,
                      'D_opt': np.random.choice(a=opt_choices),
                      'D_lr': np.random.choice(a=lr_choices),
                      'D_mom': np.random.choice(a=momentum_choices),
                      'G_filters': filters,
                      'G_kernels': kernels,
                      'G_activations': activations,
                      'G_opt': np.random.choice(a=opt_choices),
                      'G_lr': np.random.choice(a=lr_choices),
                      'G_mom': np.random.choice(a=momentum_choices),
                      'epochs': np.random.choice(a=epoch_choices),
                      'img_size': img_size,
                      'noise_dim': noise_dim,
                      'noise_size': (noise_size,noise_size),
                      'adaptive_type': np.random.choice(a=adaptive_out_choices),
                      'adaptive_size': np.random.choice(a=adaptive_out_sizes),
                      'mean': np.random.choice(a=mean_choices),
                      'std': np.random.choice(a=std_choices),
                      'num_channels':nc}
            genomes.append(genome)
    for chrom in range(num_chroms):
        gan = {'G': GeneticGenerator(genomes[chrom]),
               'D': GeneticDiscriminator(genomes[chrom])}
        chroms.append(gan)
    return chroms,genomes



def make_genetic_conv_lstm(num_chroms=10,
                           input_features=1,out_features=1):
    chroms = []
    genomes = []
    filter_choices = [2,4,6,8,12,16,24,32,64,128]
    kernel_choices = [2,3,4,5]
    activation_choices = ['relu','prelu','selu']
    hidden_sizes = [25,50,100,250,500,1000,2000]
    opt_choices = ['sgd','adam','rmsprop']
    lr_choices = [1e-4,2e-4,5e-4,1e-3,2e-3,1e-3,5e-3,1e-2]
    momentum_choices = [0.4,0.5,0.75,0.8,0.9]
    epoch_choices = [2,3,5,10,15,20,50]
    adaptive_out_choices = ['max','avg']
    adaptive_out_sizes = [4,5,6,7,8]
    norm_choices = ['batch','instance','none']
    conv_layers = [2,3,4,5,6]
    for g in range(num_chroms):
        num_conv_layers = np.random.choice(a=conv_layers)
        adaptive_pool = np.random.choice(a=adaptive_out_sizes)
        genome = {'filters': list(np.random.choice(a=filter_choices,size=num_conv_layers)),
                  'kernels': list(np.random.choice(a=kernel_choices,size=num_conv_layers)),
                  'norms': list(np.random.choice(a=norm_choices,size=num_conv_layers,
                                                 p=[0.375,0.125,0.5])),
                  'activations': list(np.random.choice(a=activation_choices,size=num_conv_layers)),
                  'adaptive_type': list(np.random.choice(a=adaptive_out_choices)),
                  'adaptive_sizes': list((adaptive_pool,adaptive_pool)),
                  'lstm_in': list(input_features),
                  'lstm_hidden': list(np.random.coice(a=hidden_sizes)),
                  'lstm_out': out_features,
                  'optimizer': np.random.choice(a=opt_choices),
                  'lr': np.random.choice(a=lr_choices),
                  'momentum': np.random.choice(a=momentum_choices),
                  'epochs': np.random.choice(a=epoch_choices)}
        genomes.append(genome)
    return genomes 


class Genetic_ConvLSTM(nn.Module):
    def __init__(self,genome,nc=3):
        super(Genetic_ConvLSTM,self).__init__()
        self.genome = genome 
        self.num_conv_layers = len(self.genome['filters'])
        self.conv_layers = [nn.Conv2d(in_channels=nc,out_channels=self.genome['filters'][0],
                                      kernel_size=self.genome['kernels'][0]),
                            nn.ReLU()]
        
        for layer in range(self.num_conv_layers-1):
            self.conv_layers.append(nn.Conv2d(in_channels=self.genome['filters'][layer],
                                              out_channels=self.genome['filters'][layer+1],
                                              kernel_size=self.genome['kernels'][layer]))
            if self.genome['norms'][layer] == 'batch':
                self.conv_layers.append(nn.BatchNorm2d(self.genome['filters'][layer+1]))
            if self.genome['norms'][layer] == 'instance':
                self.conv_layers.append(nn.InstanceNorm2d(self.genome['filters'][layer+1]))
            if self.genome['activations'][layer] == 'prelu':
                self.conv_layers.append(nn.PReLU(num_parameters=self.genome['filters'][layer+1]))
            if self.genome['activations'][layer] == 'relu':
                self.conv_layers.append(nn.ReLU())
            if self.genome['activations'][layer] == 'selu':
                self.conv_layers.append(nn.SELU())
            if self.genome['dropouts'][layer] != 'none':
                self.conv_layers.append(nn.Dropout(p=float(self.genome['dropouts'][layer])))
                
        self.width = self.genome['adaptive_size'][0]
        self.length = self.genome['adaptive_size'][1]
        if self.genome['adaptive_type'] == 'max':
            self.conv_layers.append(nn.AdaptiveMaxPool2d(output_size=(self.width,self.length)))
        if self.genome['adaptive_type'] == 'avg':
            self.conv_layers.append(nn.AdaptiveAvgPool2d(output_size=(self.width,self.length)))
            
        self.conv_block = nn.Sequential(*self.conv_layers)
        self.transfer_features = self.genome['filters'][-1]*self.width*self.length
        self.transfer = nn.Linear(nn.Linear(in_features=self.transfer_features,
                                                  out_features=self.genome['lstm_in']))
        self.lstm = nn.LSTM(self.genome['lstm_in'],self.genome['lstm_hidden'])
        self.linear = nn.Linear(self.genome['lstm_hidden'],self.genome['lstm_out'])
        self.hidden_cell = (torch.zeros(1,1,self.genome['lstm_hidden']),
                            torch.zeros(1,1,self.genome['lstm_hidden']))
    
    def forward(self,x):
        self.conv_map = self.conv_block(x)
        self.conv_map = self.conv_map.view(-1,self.transfer_features)
        self.transfer_map = self.transfer.forward(self.conv_map)
        lstm_out, self.hidden_cell = self.lstm(self.transfer_map.view(len(self.transfer_map) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(self.transfer_map), -1))
        return predictions[-1]
        
        
def evolve_dcgan(path,
                 num_generations=3,num_chroms=6,
                 transform='default',
                 batch_size=100,img_size=(40,40),
                 cmap=None,
                 normalize_plot=False,
                 noise_dim=3,
                 loss_function='bce',
                 elite_frac=0.5,
                 loss_thres=50,
                 plot_samples=True,
                 max_epochs=100):
    
    if transform == 'default':
        transf = transforms.Compose([transforms.Resize(img_size),transforms.ToTensor()])
       
    if transform != 'default':
        transf=transf
        
    if path == 'mnist':
        nc = 1
        train_set = torchvision.datasets.MNIST(root='.',train=True,transform=transf,download=True)
        train_loader = utils.data.DataLoader(dataset=train_set,
                                             batch_size=batch_size,shuffle=True)
    if path == 'fashion_mnist':
        nc = 1
        train_set = torchvision.datasets.FashionMNIST(root='.',train=True,transform=transf,download=True)
        train_loader = utils.data.DataLoader(dataset=train_set,
                                             batch_size=batch_size,shuffle=True)
    if path == 'cifar10':
        nc = 3
        train_set = torchvision.datasets.CIFAR10(root='.',train=True,transform=transf,download=True)
        train_loader = utils.data.DataLoader(dataset=train_set,
                                             batch_size=batch_size,shuffle=True)
    if path == 'cifar100':
        nc = 3
        train_set = torchvision.datasets.CIFAR100(root='.',train=True,transform=transf,download=True)
        train_loader = utils.data.DataLoader(dataset=train_set,
                                             batch_size=batch_size,shuffle=True)
    
    if path == 'imagenet':
        nc = 3
        train_set = torchvision.datasets.ImageNet(root='.',train=True,transform=transf,download=True)
        train_loader = utils.data.DataLoader(dataset=train_set,
                                             batch_size=batch_size,shuffle=True)
        
    if path not in ['imagenet','cifar100','cifar10','mnist','fashion_mnist']:
        nc = 3
        train_loader=utils.data.DataLoader(dataset=torchvision.datasets.ImageFolder(root=path, 
                                     transform=transf),batch_size=batch_size,shuffle=True)
    
    chroms = make_genetic_dcgans(num_chroms=num_chroms,
                               nc=nc,
                               noise_dim=noise_dim,
                               max_epochs=max_epochs)
    max_fitness = []
    for gen in range(num_generations):
        models_and_results = []
        fitness = []
        for chrom in range(num_chroms):
            D = chroms[0][chrom]['D']
            G = chroms[0][chrom]['G']
            g_opt_choice = chroms[1][chrom]['G_opt']
            d_opt_choice = chroms[1][chrom]['D_opt']
            if g_opt_choice == 'sgd':
                g_opt = torch.optim.SGD(G.parameters(),
                                        lr=chroms[1][chrom]['G_lr'],
                                        momentum=chroms[1][chrom]['G_mom'])
            if g_opt_choice == 'adam':
                g_opt = torch.optim.Adam(G.parameters(),
                                        lr=chroms[1][chrom]['G_lr'])
            if g_opt_choice == 'rmsprop':
                g_opt = torch.optim.RMSprop(G.parameters(),
                                        lr=chroms[1][chrom]['G_lr'],
                                        momentum=chroms[1][chrom]['G_mom'])
            if d_opt_choice == 'sgd':
                d_opt = torch.optim.SGD(D.parameters(),
                                        lr=chroms[1][chrom]['D_lr'],
                                        momentum=chroms[1][chrom]['D_mom'])
            if d_opt_choice == 'adam':
                d_opt = torch.optim.Adam(D.parameters(),
                                        lr=chroms[1][chrom]['D_lr'])
            if d_opt_choice == 'rmsprop':
                d_opt = torch.optim.RMSprop(D.parameters(),
                                        lr=chroms[1][chrom]['D_lr'],
                                        momentum=chroms[1][chrom]['D_mom'])
            epochs = chroms[1][chrom]['epochs']
            if loss_function == 'mse':
                criterion = nn.MSELoss()
            if loss_function == 'bce':
                criterion = nn.BCELoss()
            if loss_function == 'kl_div':
                criterion = nn.KLDivLoss()
            if loss_function == 'mae':
                criterion == nn.L1Loss()
            for epoch in range(epochs):
                D_loss_list = []
                G_loss_list = []
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
                    noise = truncated_noise(torch.randn(batch_size,
                                                        noise_dim,
                                                        chroms[1][chrom]['noise_size'][0],
                                                        chroms[1][chrom]['noise_size'][1]),
                                            mean=chroms[1][chrom]['mean'],
                                            std=chroms[1][chrom]['std'])
                    noise_var = Variable(noise)
                    G_result = G(noise_var)
                    
                    prob_vec = D(G_result)
                    D_fake_result = torch.max(prob_vec,dim=1)[0]
                    D_fake_loss = criterion(D_fake_result, label_fake_var.view(1,-1))
                    D_train_loss = D_real_loss + D_fake_loss
                    D.zero_grad()
                    D_train_loss.backward()
                    d_opt.step()
                    #noise = torch.randn((batch_size, 100)).view(-1, 100, 1, 1)
                            #noise_var = Variable(noise)
                    G_result = G(noise_var)
                    prob_vec = D(G_result)
                    D_fake_result = torch.max(prob_vec,dim=1)[0]
                    G_train_loss = criterion(D_fake_result, label_real_var.view(1,-1))#+(loss_coeffs[chrom]*criterion_l1(G_result,input_var))
                    D.zero_grad()
                    G.zero_grad()
                    G_train_loss.backward()
                    g_opt.step()
                    D_loss_list.append(D_train_loss.item())
                    G_loss_list.append(G_train_loss.item())
                    print('Generation: [{}/{}] | Chromosome: [{}/{}]'.format(gen+1,num_generations,chrom+1,num_chroms))
                    print('epoch: [{}/{}] | batch: [{}/{}]\n'.format(epoch+1,epochs,i+1,len(train_loader)))
                    print('G_Loss: {}  |  D_Loss: {}\n'.format(G_train_loss.item(),D_train_loss.item()))
                    if plot_samples == True:
                        plot_batch(tensor=G_result,title='Generator',cmap=cmap)
                plot_batch(tensor=G_result,title='samples',
                           path='neuroevolution_outputs/gen_{}/chrom_{}/epoch_{}'.format(gen+1,chrom+1,epoch+1),
                           save=True)        
                if max(G_loss_list) + max(D_loss_list) > loss_thres:
                    print('DANG IT BOBBY!!! The GAN loss is too high...aborting training')
                    score = 0
                    fitness.append(score)
                    results = {'aborted'}
                    break
                #samples.append(im)            
            
            score = dcgan_fitness(genome=chroms[1][chrom],
                                D_loss=max(D_loss_list),
                                G_loss=max(G_loss_list))
            
            results = {'genome': chroms[1][chrom],
                       'G_loss': G_loss_list, 
                       'D_loss': D_loss_list,
                       'G_weights': G.state_dict(),
                       'D_weights': D.state_dict(),
                       'fitness': score,
                       'samples': G_result}
            fitness.append(score)
            models_and_results.append(results)
        max_fitness.append(max(fitness))
        num_elites = int(np.ceil(elite_frac*num_chroms)) #THIS LINE IS CORRECT, CHECK ALL EVOLUTIONARY SCRIPTS!!!!
        fitness_sorted = np.argsort(fitness)
        chroms_sorted = [i for _,i in sorted(zip(fitness_sorted,chroms[0]))]
        genomes_sorted = [i for _,i in sorted(zip(fitness_sorted,chroms[1]))]
        chroms_sorted = chroms_sorted[::-1]
        genomes_sorted = genomes_sorted[::-1]
        
        chroms = chroms_sorted[:num_elites]
        genomes = genomes_sorted[:num_elites]
        num_new = num_chroms-num_elites
        new_pop = make_genetic_dcgans(num_chroms=num_new,
                                      nc=nc,
                                      noise_dim=noise_dim,
                                      max_epochs=max_epochs)

        
        new_chroms = new_pop[0] #gets the new chromosomes
        new_genomes = new_pop[1] #gets the new genomes 
        chroms.extend(new_chroms)
        genomes.extend(new_genomes)
        pop = (chroms,genomes) 
        chroms = pop
    return models_and_results,max_fitness
                

def genetic_cnn_info(results,metric='fitness'):
    '''
    gets results from genetic_train_cnn - will always be the last generation 
    the algorithm completed
    
    metric: string - metric of interest to examine
                     1) Fitness
                     2) Accuracy
                     3) conf_mat (Confusion Matrix)
                     4) precision
                     5) recall
                     6) F1 
                     7) Loss (train loss)
    '''
    num_chroms = len(results[0])
    for chrom in range(num_chroms):
        if metric == 'fitness':
            print('Chrom {} Fitness: {:4f}'.format(chrom+1,results[0][chrom]['fitness']))
        if metric == 'accuracy':
            print('Chrom {}\n======'.format(chrom+1))
            print('Train\nCorrect: {}'.format(results[0][chrom]['results']['train']['correct']))
            print('{}%'.format(results[0][chrom]['results']['train']['accuracy']))
            print('Test\nCorrect: {}'.format(results[0][chrom]['results']['test']['correct']))
            print('{}%'.format(results[0][chrom]['results']['test']['accuracy']))
        if metric == 'conf_mat':
            print('Chrom {}\n======'.format(chrom+1))
            plt.title('Confusion Matrix Chrom (Train): {}/{}'.format(chrom+1,num_chroms))
            plt.imshow(results[0][chrom]['results']['train']['confusion_matrix'])
            plt.show()
            plt.close()
            print('Chrom {}\n======'.format(chrom+1))
            plt.title('Confusion Matrix Chrom (Train): {}/{}'.format(chrom+1,num_chroms))
            plt.imshow(results[0][chrom]['results']['train']['confusion_matrix'])
            plt.show()
            plt.close()
            print('TRAIN: {}'.format(results[0][chrom]['results']['train']['confusion_matrix']))
            print('TEST: {}'.format(results[0][chrom]['results']['test']['confusion_matrix']))
        if metric == 'precision':
            print('Chrom {}\n======'.format(chrom+1))
            num_classes = len(results[0][chrom]['results']['train']['{}'.format(metric)])
            for c in range(num_classes):
                print('TRAIN\n=========')
                print('Class {} Precision: {}\n'.format(c+1,
                      format(results[0][chrom]['results']['train']['{}'.format(metric)][c])))
                print('TEST\n==========')
                print('Class {} Precision: {}\n'.format(c+1,
                      format(results[0][chrom]['results']['test']['{}'.format(metric)][c])))
        if metric == 'recall':
            print('Chrom {}\n======'.format(chrom+1))
            num_classes = len(results[0][chrom]['results']['train']['{}'.format(metric)])
            for c in range(num_classes):
                print('TRAIN\n=========')
                print('Class {} {}: {}\n'.format(c+1,metric,
                      format(results[0][chrom]['results']['train']['{}'.format(metric)][c])))
                print('TEST\n==========')
                print('Class {} {}: {}\n'.format(c+1,metric,
                      format(results[0][chrom]['results']['test']['{}'.format(metric)][c])))
        
        if metric == 'F1':
            print('Chrom {}\n======'.format(chrom+1))
            num_classes = len(results[0][chrom]['results']['train']['{}'.format(metric)])
            for c in range(num_classes):
                
                print('TRAIN\n=========')
                print('Class {} {}: {}\n'.format(c+1,metric,
                      format(results[0][chrom]['results']['train']['{}'.format(metric)][c])))
            for c in range(num_classes):
                
                print('TEST\n==========')
                print('Class {} {}: {}\n'.format(c+1,metric,
                      format(results[0][chrom]['results']['test']['{}'.format(metric)][c])))
        if metric == 'loss':
            epochs = results[0][chrom]['genome']['epochs']
            lin = np.linspace(0,epochs,len(results[0][chrom]['results']['train']['loss']))
            plt.title('Chrom {} Loss'.format(chrom+1))
            plt.plot(lin,results[0][chrom]['results']['train']['loss'],c='blue')
            plt.xlabel('Epochs')
            plt.xticks(ticks=np.arange(0,epochs+1,step=1))
            plt.show()


def get_best_model(results):
    '''
    gets best performing model, along with weights and entire genome
    results: tuple of dictionaries - see genetic_train_cnn code 
    '''
    fitness_scores = []
    for chrom in range(len(results[0])):
        fitness_scores.append(results[0][chrom]['fitness'])
    best_model_idx = np.argmax(fitness_scores)
    return {'genome': results[0][best_model_idx]['genome'],
            'model': results[0][best_model_idx]['results']['model'],
            'weights': results[0][best_model_idx]['results']['weights']}

def dump_genetic_cnn(results,path='./neuroevolution_cnn_data.txt',
                     evolution_mode='elitist'):
    '''
    function that dumps results from evolving the CNN classifier to a text file
    results : tuple of dictionaries - comes from the return value of the genetic_cnn_train
              the dictionaries themselves often contain more nested dictionaries -
              the code below shows the structure
    path: string - path to store the text file
    '''
   
    num_chroms = len(results[0])
    for chrom in range(num_chroms):
        f = open(path,"a")
        f.write('=======================\n')
        f.write('CHROMOSOME {}/{}\n'.format(chrom+1,num_chroms))
        f.write('FITNESS: {:4f}\n'.format(results[0][chrom]['fitness']))
        f.write('filters: {}\n'.format(results[0][chrom]['genome']['filters']))
        f.write('kernels: {}\n'.format(results[0][chrom]['genome']['kernels']))
        f.write('dropouts: {}\n'.format(results[0][chrom]['genome']['dropouts']))
        f.write('norms: {}\n'.format(results[0][chrom]['genome']['norms']))
        f.write('adaptive type: {}\n'.format(results[0][chrom]['genome']['adaptive_type']))
        f.write('adaptive size: {}\n'.format(results[0][chrom]['genome']['adaptive_size']))
        f.write('hidden sizes: {}\n'.format(results[0][chrom]['genome']['hidden_sizes']))
        f.write('activations: {}\n'.format(results[0][chrom]['genome']['activations']))
        f.write('optimizer: {}\n'.format(results[0][chrom]['genome']['optimizer']))
        f.write('learning rate: {}\n'.format(results[0][chrom]['genome']['lr']))
        f.write('momentum: {}\n'.format(results[0][chrom]['genome']['momentum']))
        f.write('epochs: {}\n'.format(results[0][chrom]['genome']['epochs']))
        f.write('TRAIN METRICS ======\n')
        f.write('Correct: {}\n'.format(results[0][chrom]['results']['train']['correct']))
        f.write('Accuracy: {}%\n'.format(results[0][chrom]['results']['train']['accuracy']))
        f.write('F1: {:4f}\n'.format(np.mean(results[0][chrom]['results']['train']['F1'])))
        f.write('precision: {:4f}\n'.format(np.mean(results[0][chrom]['results']['train']['precision'])))
        f.write('recall: {:4f}\n'.format(np.mean(results[0][chrom]['results']['train']['recall'])))
        f.write('TEST METRICS ======\n')
        f.write('correct: {}\n'.format(results[0][chrom]['results']['test']['correct']))
        f.write('accuracy: {}%\n'.format(results[0][chrom]['results']['test']['accuracy']))
        f.write('F1: {:4f}\n'.format(np.mean(results[0][chrom]['results']['test']['F1'])))
        f.write('precision: {:4f}\n'.format(np.mean(results[0][chrom]['results']['test']['precision'])))
        f.write('recall: {:4f}\n'.format(np.mean(results[0][chrom]['results']['test']['recall'])))
        f.write('=======================\n')
    f.write('Evolutionary Mode: {}\n'.format(evolution_mode))
    if evolution_mode == 'crossover':
        num_parents = len(results[2]['parents'])
        num_offspring = num_parents - 1 #only first N of these come from parents
        for p in range(num_parents):
            f.write('parent: {}/{}\n'.format(p+1,num_parents))
            f.write('filters: {}\n'.format(results[2]['parents'][p]['filters']))
            f.write('kernels: {}\n'.format(results[2]['parents'][p]['kernels']))
            f.write('dropouts: {}\n'.format(results[2]['parents'][p]['dropouts']))
            f.write('norms: {}\n'.format(results[2]['parents'][p]['norms']))
            f.write('adaptive type: {}\n'.format(results[2]['parents'][p]['adaptive_type']))
            f.write('adaptive size: {}\n'.format(results[2]['parents'][p]['adaptive_size']))
            f.write('hidden sizes: {}\n'.format(results[2]['parents'][p]['hidden_sizes']))
            f.write('activations: {}\n'.format(results[2]['parents'][p]['activations']))
            f.write('optimizer: {}\n'.format(results[2]['parents'][p]['optimizer']))
            f.write('learning rate: {}\n'.format(results[2]['parents'][p]['lr']))
            f.write('momentum: {}\n'.format(results[2]['parents'][p]['momentum']))
            f.write('epochs: {}\n'.format(results[2]['parents'][p]['epochs']))
            f.write('===================\n')
        f.write('=================\n')
        for c in range(num_offspring):
            f.write('offspring: {}/{}\n'.format(c+1,num_offspring))
            f.write('filters: {}\n'.format(results[2]['offspring'][c]['filters']))
            f.write('kernels: {}\n'.format(results[2]['offspring'][c]['kernels']))
            f.write('dropouts: {}\n'.format(results[2]['offspring'][c]['dropouts']))
            f.write('norms: {}\n'.format(results[2]['offspring'][c]['norms']))
            f.write('adaptive type: {}\n'.format(results[2]['offspring'][c]['adaptive_type']))
            f.write('adaptive size: {}\n'.format(results[2]['offspring'][c]['adaptive_size']))
            f.write('hidden sizes: {}\n'.format(results[2]['offspring'][c]['hidden_sizes']))
            f.write('activations: {}\n'.format(results[2]['offspring'][c]['activations']))
            f.write('optimizer: {}\n'.format(results[2]['offspring'][c]['optimizer']))
            f.write('learning rate: {}\n'.format(results[2]['offspring'][c]['lr']))
            f.write('momentum: {}\n'.format(results[2]['offspring'][c]['momentum']))
            f.write('epochs: {}\n'.format(results[2]['offspring'][c]['epochs']))
            f.write('==================\n')
        f.close()
    if evolution_mode == 'elitist':
        f.close()
    return 'complete'

def plot_fitness(results):
    num_gens = len(results[1])
    if num_gens <= 15:
        plt.title('Fitness')
        plt.xlabel('Generations')
        plt.plot(np.arange(0,num_gens),results[1])
        plt.xticks(ticks=np.arange(0,num_gens+1))
        plt.show()
    if num_gens >= 15:
        plt.title('Fitness')
        plt.xlabel('Generations')
        plt.plot(np.arange(0,num_gens),results[1])
        plt.show()

        
def dump_gan_results(results,path='./neuroevolution_gan_data.txt'):
    num_chroms = len(results[0])
    for chrom in range(num_chroms):
        f = open(path,"a")
        f.write('=============\n')
        f.write('CHROMOSOME {}/{}\n'.format(chrom+1,num_chroms))
        f.write('FITNESS: {:4f}\n'.format(results[0][chrom]['fitness']))
        f.write('img size: {}\n'.format(results[0][chrom]['genome']['img_size']))
        f.write('GENERATOR:\n')
        f.write('filters: {}\n'.format(results[0][chrom]['genome']['G_filters']))
        f.write('kernels: {}\n'.format(results[0][chrom]['genome']['G_kernels']))
        f.write('activations: {}\n'.format(results[0][chrom]['genome']['G_activations']))
        f.write('adaptive type: {}\n'.format(results[0][chrom]['genome']['adaptive_type']))
        f.write('adaptive size: {}\n'.format(results[0][chrom]['genome']['adaptive_size']))
        f.write('optimizer: {}\n'.format(results[0][chrom]['genome']['G_opt']))
        f.write('learning rate: {}\n'.format(results[0][chrom]['genome']['G_lr']))
        f.write('momentum: {}\n'.format(results[0][chrom]['genome']['G_mom']))
        f.write('Noise shape: [{},{},{}]\n'.format(results[0][chrom]['genome']['noise_dim'],
                results[0][chrom]['genome']['noise_size'][0],
                results[0][chrom]['genome']['noise_size'][1]))
        f.write('mean: {} standard deviation: {}\n'.format(results[0][chrom]['genome']['mean'],
                results[0][chrom]['genome']['std']))
        f.write('DISCRIMINATOR:\n')
        f.write('filters: {}\n'.format(results[0][chrom]['genome']['D_filters']))
        f.write('kernels: {}\n'.format(results[0][chrom]['genome']['D_kernels']))
        f.write('activations: {}\n'.format(results[0][chrom]['genome']['D_activations']))
        f.write('optimizer: {}\n'.format(results[0][chrom]['genome']['D_opt']))
        f.write('learning rate: {}\n'.format(results[0][chrom]['genome']['D_lr']))
        f.write('momentum: {}\n'.format(results[0][chrom]['genome']['D_mom']))
    f.close()
    return 'complete'



    
            
                    
                    
                    