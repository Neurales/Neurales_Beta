# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 16:38:03 2020

@author: XZ-WAVE
"""

import numpy as np
import sklearn as sk
from sklearn import metrics
from sklearn import svm 
from sklearn import feature_selection 

import heapq 
from Neurales_2020_main import *
import torch 
import torch.nn as nn 


def make_nn_pipeline_chroms(num_chroms,mode='reg',in_size=1,out_size=1):
    activation_choices = ['relu','tanh','prelu','selu']
    hidden_sizes = [5,10,25,50,100,250,500,1000,2500,5000]
    depth_choices = [2,3,4,5,6,7,8,9]
    dropout_choices = [0.1,0.2,0.25,0.3,0.5,'none']
    opt_choices = ['sgd','adam','rmsprop']
    lr_choices = [1e-4,2e-4,5e-4,1e-3,2e-3,5e-3,1e-2]
    momentum_choices = [0.4,0.5,0.6,0.75,0.8,0.9]
    feature_scale_choices = ['standard','min_max','none']
    
    genomes = []
    chroms = []
    for g in range(num_chroms):
        num_layers = np.random.choice(a=depth_choices)
        feature_scale = np.random.choice(a=feature_scale_choices)
        if feature_scale == 'standard':
            scale = sk.preprocessing.StandardScaler()
        if feature_scale == 'min_max':
            scale = sk.preprocessing.MinMaxScaler()
        if feature_scale == 'none':
            scale = sk.preprocessing.FunctionTransformer()
        
        genome = {'hidden_sizes': list(np.random.choice(a=hidden_sizes,size=num_layers)),
                  'dropouts': list(np.random.choice(a=dropout_choices,size=num_layers,
                                                 p=[0.1,0.1,0.1,0.1,0.1,0.5])),
                    'activations': list(np.random.choice(a=activation_choices,size=num_layers)),
               'optimizer': np.random.choice(a=opt_choices),
               'lr': np.random.choice(a=lr_choices),
               'momentum': np.random.choice(a=momentum_choices),
               'feature_scale': scale }
        genomes.append(genome)
    for chrom in range(num_chroms):
        model = NeuralReg(genome=genomes[chrom],in_size=in_size,out_size=out_size)
        chroms.append(model)
    return chroms,genomes
        

class NeuralReg(nn.Module):
    def __init__(self,genome,in_size=1,out_size=1):
        super(NeuralReg,self).__init__()
        self.genome = genome
        self.layers = [nn.Linear(in_features=in_size,
                                   out_features=self.genome['hidden_sizes'][0]),nn.ReLU()]
        for layer in range(len(self.genome['hidden_sizes'])-1):
            self.layers.append(nn.Linear(in_features=self.genome['hidden_sizes'][layer],
                                         out_features=self.genome['hidden_sizes'][layer+1]))
            
            if self.genome['dropouts'][layer] != 'none':
                p = float(self.genome['dropouts'][layer])
                self.layers.append(nn.Dropout(p=p))
            if self.genome['activations'][layer] == 'relu':
                self.layers.append(nn.ReLU())
            if self.genome['activations'][layer] == 'tanh':
                self.layers.append(nn.Tanh())
            if self.genome['activations'][layer] == 'selu':
                self.layers.append(nn.SELU())
            if self.genome['activations'][layer] == 'prelu':
                self.layers.append(nn.PReLU())
        self.layers.append(nn.Linear(in_features=self.genome['hidden_sizes'][-1],
                                     out_features=out_size))
        
        self.network = nn.Sequential(*self.layers)
    
    def forward(self,x):
        output = self.network(x)
        return output
        

def pipeline_fitness(X,K,y_true,y_pred,metric='f1'):
    feature_ratio = K/X.shape[1]
    if metric == 'f1':
        score_fitness = sk.metrics.f1_score(y_true=y_true,y_pred=y_pred,average=None)
    if metric == 'precision':
        score_fitness = sk.metrics.precision_score(y_true=y_true,y_pred=y_pred,average=None)
    if metric == 'recall':
        score_fitness = sk.metrics.recall_score(y_true=y_true,y_pred=y_pred,average=None)
    if metric == 'accuracy':
        score_fitness = sk.metrics.accuracy_score(y_true=y_true,y_pred=y_pred)
    arg = (0.5*feature_ratio)+score_fitness
    fitness = np.tanh(arg)
    return np.mean(fitness)


def classifier_fitness(model,y_true,y_pred,train_per,metric='f1'):
    t_score = 1/train_per
    if metric == 'f1':
        score_fitness = sk.metrics.f1_score(y_true=y_true,y_pred=y_pred,average=None)
    if metric == 'precision':
        score_fitness = sk.metrics.precision_score(y_true=y_true,y_pred=y_pred,average=None)
    if metric == 'recall':
        score_fitness = sk.metrics.recall_score(y_true=y_true,y_pred=y_pred,average=None)
    if metric == 'accuracy':
        score_fitness = sk.metrics.accuracy_score(y_true=y_true,y_pred=y_pred)
    arg = (0.75*t_score)+score_fitness
    fitness = np.tanh(arg)
    return np.mean(fitness)


def pipeline_fitness_regression(y_true,y_pred,neural_net=False,genome=None):
    if neural_net == True:
        loss_score = np.mean(1/abs(y_true-y_pred))
        param_score = 10/np.sum(genome['hidden_sizes'])
        score = loss_score + param_score
        return score
    score = np.mean(1/abs(y_true-y_pred))
    
    return score


def make_gradient_booster_chroms(num_chroms=10):
    depth_choices = np.arange(2,15)
    num_learner_choices = np.arange(5,200,step=5)
    lr_choices = [1e-4,2e-4,5e-4,1e-3,2e-3,5e-3,1e-2,2e-2]
    train_per_choices = [0.05,0.15,0.25,0.5,0.6,0.7,0.75,0.8,0.85]
    chroms = []
    for chrom in range(num_chroms):
        chroms.append({'lr': np.random.choice(a=lr_choices),
                       'depth': np.random.choice(a=depth_choices),
                       'estimators': np.random.choice(a=num_learner_choices),
                       'train_per': np.random.choice(a=train_per_choices)})
    return chroms 


def make_gradient_booster_pipeline_chroms(X,num_chroms=10,mode='reg'):
    depth_choices = np.arange(2,15)
    num_learner_choices = np.arange(5,200,step=5)
    lr_choices = [1e-4,2e-4,5e-4,1e-3,2e-3,5e-3,1e-2,2e-2]
    K_choices = [0.1,0.25,0.5,0.75,1]
    if mode == 'clf':
        score_fn_choices = [sk.feature_selection.f_classif,
                        sk.feature_selection.mutual_info_classif]
    if mode == 'reg':
        score_fn_choices = [sk.feature_selection.mutual_info_regression,
                            sk.feature_selection.f_regression]
    
    feature_scale_choices = ['standard','min_max','none']
    chroms = []
    for chrom in range(num_chroms):
        K = np.random.choice(a=K_choices)*X.shape[1]
        feature_scale = np.random.choice(a=feature_scale_choices)
        if feature_scale == 'standard':
            scale = sk.preprocessing.StandardScaler()
        if feature_scale == 'min_max':
            scale = sk.preprocessing.MinMaxScaler()
        if feature_scale == 'none':
            scale = sk.preprocessing.FunctionTransformer()
        chroms.append({'depth': np.random.choice(a=depth_choices),
                       'estimators': np.random.choice(a=num_learner_choices),
                       'lr': np.random.choice(a=lr_choices),
                       'K': int(K),
                       'score_function': np.random.choice(a=score_fn_choices),
                       'feature_scale': scale})
    return chroms 


def make_svm_pipeline_chroms(X,y,num_chroms=10,mode='clf'):

    gamma_choices = [1e-4,2e-4,5e-4,1e-3,2e-3,5e-3,1e-2,2e-2,5e-2,1e-1,5e-1]
    C_choices = [1e-3,5e-3,1e-2,5e-2,1e-1,2e-1,5e-1,1,2]
    if mode == 'clf':
        score_fn_choices = [sk.feature_selection.f_classif,
                        sk.feature_selection.mutual_info_classif]
    if mode == 'reg':
        score_fn_choices = [sk.feature_selection.mutual_info_regression,
                            sk.feature_selection.f_regression]
    
    feature_scale_choices = ['standard','min_max','none']
    K_choices = [0.1,0.25,0.5,0.75,1]
    kernel_choices = ['linear','rbf','poly','sigmoid']
    degree_choices = [2,3,4,5,6,7]
    chroms = []
    for i in range(num_chroms):
        K = np.random.choice(a=K_choices)*X.shape[1]
        feature_scale = np.random.choice(a=feature_scale_choices)
        kernel_choice = np.random.choice(a=kernel_choices)
        C = np.random.choice(a=C_choices)
        if kernel_choice == 'rbf':
            gamma = np.random.choice(a=gamma_choices)
            deg = 3
        if kernel_choice == 'linear' or kernel_choice == 'sigmoid':
            gamma = 'scale'
            deg = 3
        if kernel_choice == 'poly':
            deg = np.random.choice(a=degree_choices)
            gamma = 'scale'
        if feature_scale == 'standard':
            scale = sk.preprocessing.StandardScaler()
        if feature_scale == 'min_max':
            scale = sk.preprocessing.MinMaxScaler()
        if feature_scale == 'none':
            scale = sk.preprocessing.FunctionTransformer()
        chroms.append({'kernel': kernel_choice,
                       'C': np.random.choice(a=C_choices),
                       'gamma': gamma,
                       'degree': deg,
                       'feature_scale': scale,
                       'score_function': np.random.choice(a=score_fn_choices),
                       'K': max(2,int(K))})
    return chroms 


def evolve_gradient_booster_clf(X,y,num_generations=10,num_chroms=20,metric='f1',
                            elite_frac=0.2,target_acc=0.95):
    chroms = make_gradient_booster_chroms(num_chroms=num_chroms)
    for gen in range(num_generations):
        elites = []
        fitness = []
        models = []
        for chrom in range(num_chroms):

              X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                             train_per=chroms[chrom]['train_per'])
              
              model = make_gradient_booster_clf(lr=chroms[chrom]['lr'],
                                                estimators=chroms[chrom]['estimators'],
                                                depth=chroms[chrom]['depth'])
              model.fit(X_train,y_train)
              y_test_pred = model.predict(X_test)
              score = classifier_fitness(model,y_true=y_test,y_pred=y_test_pred,
                                         train_per=chroms[chrom]['train_per'])
              correct = np.sum(y_test_pred == y_test)
              total = len(y_test)
              acc = 100*np.around(correct/total,decimals=3)
              acc = np.format_float_positional(acc,precision=3)
              print('Generation: [{}/{}] | Chromosome: [{}/{}]'.format(gen+1,num_generations,chrom+1,num_chroms))
              print('{} correct/total: {}/{}\n'.format('test',correct,total))
              print('{} accuracy: {} %\n'.format('test',acc))
              print("Gradient Booster Fitness: {}".format(score))
              fitness.append(score)
              models.append(model)

def evolve_gradient_booster_pipeline_clf(X,y,num_generations=10,num_chroms=20,
                                         metric='f1',elite_frac=0.2,target_acc=0.98,
                                         train_per=0.75):
    chroms = make_gradient_booster_pipeline_chroms(X=X,num_chroms=num_chroms)
    max_fitness = []
    max_acc = 0
    for gen in range(num_generations):
        fitness = []
        models = []
        for chrom in range(num_chroms):
            K=chroms[chrom]['K']
            y = sk.preprocessing.LabelEncoder().fit_transform(y) #this line is needed because sometimes errors will occur otherwise
            X_new = sk.feature_selection.SelectKBest(chroms[chrom]['score_function'],k=int(K)).fit_transform(X,y)
            X_new = chroms[chrom]['feature_scale'].fit_transform(X)
            X_train,X_test,y_train,y_test = train_test_split(X_new,y,
                                                             train_per=train_per)
            
            model = make_gradient_booster_clf(lr=chroms[chrom]['lr'],
                                              estimators=chroms[chrom]['estimators'],
                                              depth=chroms[chrom]['depth'])
           
            y_train = sk.preprocessing.LabelEncoder().fit_transform(y_train)
            y_test = sk.preprocessing.LabelEncoder().fit_transform(y_test)
            model.fit(X_train,y_train)
               
            y_train_pred = predict(model,X_train)
            y_test_pred = predict(model,X_test)
            score = pipeline_fitness(X=X,K=K,y_true=y_test,
                                     y_pred=y_test_pred,
                                     metric=metric)
            train_correct = np.sum(y_train_pred == y_train)
            correct = np.sum(y_test_pred == y_test)
            total = len(y_test)
            acc = 100*(correct/total)
            acc = float(acc)
            print('Generation: [{}/{}] | Chromosome: [{}/{}]'.format(gen+1,num_generations,chrom+1,num_chroms))
            print('{} correct/total: {}/{}\n'.format('train',train_correct,len(y_train)))
            print('{} accuracy: {:4f}%\n'.format('train',100*(train_correct/len(y_train))))
            print('{} correct/total: {}/{}\n'.format('test',correct,total))
            print('{} accuracy: {:4f}%\n'.format('test',acc))
            print("Gradient Booster Fitness: {:4f}".format(score))
            fitness.append(score)
            models.append((model,{'fitness': score,'correct':'{}/{}'.format(correct,total),
                                  'F1': sk.metrics.f1_score(y_true=y_test,y_pred=y_test_pred,average=None),
                                  'precision': sk.metrics.precision_score(y_true=y_test,y_pred=y_test_pred,average=None),
                                  'recall': sk.metrics.recall_score(y_true=y_test,y_pred=y_test_pred,average=None)}))
            if acc > max_acc:
               max_acc = acc
        max_fitness.append(max(fitness))
        print("Generation {} max accuracy: {}".format(gen+1,max_acc))
        num_elites = int(np.ceil(elite_frac*num_chroms))
        fitness_sorted = np.argsort(fitness)
        chroms_sorted = [i for _,i in sorted(zip(fitness_sorted,chroms))]
        chroms_sorted = chroms_sorted[::-1]
        chroms = chroms_sorted[0:num_elites]
        new_chroms = make_gradient_booster_pipeline_chroms(X=X,num_chroms=num_chroms-len(chroms))
        chroms.extend(new_chroms)
        if max_acc >= 100*target_acc:
            print('Solved! - model with >={}% metric accuracy found!'.format(100*target_acc))
            break
        
    plt.title('Fitness')
    plt.plot(np.arange(0,len(max_fitness)),max_fitness)
    plt.xlabel("Generations")
    plt.show()
    return {'models': models,
            'fitness': fitness,
            'genomes': chroms}

      
def evolve_svm_pipeline_clf(X,y,num_generations=10,num_chroms=20,metric='f1',
                        elite_frac=0.2,target_acc=0.99,train_per=0.75):
    
    chroms = make_svm_pipeline_chroms(X,y,num_chroms=num_chroms)
    max_acc = 0.0
    max_fitness = []
    for gen in range(num_generations):
        elites = []
        fitness = []
        models = []
        
        for chrom in range(num_chroms):
            K=chroms[chrom]['K']
            X_new = sk.feature_selection.SelectKBest(chroms[chrom]['score_function'],k=K).fit_transform(X,y)
            X_new = chroms[chrom]['feature_scale'].fit_transform(X)
            X_train,X_test,y_train,y_test = train_test_split(X_new,y,
                                                             train_per=train_per)
            model = sk.svm.SVC(kernel=chroms[chrom]['kernel'],
                               C=chroms[chrom]['C'],
                               gamma=chroms[chrom]['gamma'],
                               degree=chroms[chrom]['degree'])
            
            model.fit(X_train,y_train)
            y_train_pred = predict(model,X_train)
            y_test_pred = predict(model,X_test)
            score = pipeline_fitness(X=X,K=K,y_true=y_test,
                                     y_pred=y_test_pred,
                                     metric=metric)
            correct = np.sum(y_test_pred == y_test)
            total = len(y_test)
            acc = 100*np.around(correct/total,decimals=3)
            acc = np.format_float_positional(acc,precision=3)
            acc = float(acc)
            print('Generation: [{}/{}] | Chromosome: [{}/{}]'.format(gen+1,num_generations,chrom+1,num_chroms))
            print('{} correct/total: {}/{}\n'.format('test',correct,total))
            print('{} accuracy: {} %\n'.format('test',acc))
            print("SVM Fitness: {}".format(score))
            fitness.append(score)
            models.append(model)
            if acc > max_acc:
               max_acc = acc
        max_fitness.append(max(fitness))
        print("Generation {} max accuracy: {}".format(gen+1,max_acc))
        num_elites = int(np.ceil(elite_frac*num_chroms))
        fitness_sorted = np.argsort(fitness)
        chroms_sorted = [i for _,i in sorted(zip(fitness_sorted,chroms))]
        chroms_sorted = chroms_sorted[::-1]
        chroms = chroms_sorted[0:num_elites]
        new_chroms = make_svm_pipeline_chroms(X,y,num_chroms=num_chroms-len(chroms))
        chroms.extend(new_chroms)
        if max_acc >= 100*target_acc:
            print('Solved! - model with >={}% test accuracy found!'.format(100*target_acc))
            break
        
    plt.title('Fitness')
    plt.plot(np.arange(0,len(max_fitness)),max_fitness)
    plt.xlabel("Generations")
    plt.show()
    return {'models': models,'fitness': fitness}
                
def evolve_svm_pipeline_reg(X,y,num_generations=10,num_chroms=20,
                        elite_frac=0.2,tol=1e-2,
                        train_per=0.75):
    
    chroms = make_svm_pipeline_chroms(X,y,num_chroms=num_chroms,mode='reg')
    min_loss = 1000
    max_fitness = []
    for gen in range(num_generations):
        elites = []
        fitness = []
        models = []

        for chrom in range(num_chroms):
            K=chroms[chrom]['K']
            X_new = sk.feature_selection.SelectKBest(chroms[chrom]['score_function'],k=K).fit_transform(X,y)
            X_new = chroms[chrom]['feature_scale'].fit_transform(X)
            X_train,X_test,y_train,y_test = train_test_split(X_new,y,
                                                             train_per=train_per)
            model = sk.svm.SVR(kernel=chroms[chrom]['kernel'],
                               C=chroms[chrom]['C'],
                               gamma=chroms[chrom]['gamma'],
                               degree=chroms[chrom]['degree'])
            model.fit(X_train,y_train)
            y_test_pred = predict(model,X_test)
            loss = sk_mse_loss(y_true=y_test,y_pred=y_test_pred)
            score = pipeline_fitness_regression(y_true=y_test,y_pred=y_test_pred)
            print('Generation: [{}/{}] | Chromosome: [{}/{}]'.format(gen+1,num_generations,chrom+1,num_chroms))
            print('loss: {:4f}'.format(loss))
            print('fitness: {:4f}'.format(score))
            fitness.append(score)
            models.append(model)
            if loss < min_loss:
               min_loss = loss
        max_fitness.append(max(fitness))
        print("Generation {} min loss: {}".format(gen+1,min_loss))
        num_elites = int(np.ceil(elite_frac*num_chroms))
        fitness_sorted = np.argsort(fitness)
        chroms_sorted = [i for _,i in sorted(zip(fitness_sorted,chroms))]
        chroms_sorted = chroms_sorted[::-1]
        chroms = chroms_sorted[0:num_elites]
        new_chroms = make_svm_pipeline_chroms(X,y,num_chroms=num_chroms-len(chroms),mode='reg')
        chroms.extend(new_chroms)
        if min_loss <= tol:
            print('Solved! - model with under {} loss found!'.format(loss))
            break
        
    plt.title('Fitness')
    plt.plot(np.arange(0,len(max_fitness)),max_fitness)
    plt.xlabel("Generations")
    plt.show()
    return {'models': models,
            'fitness': fitness}
            
    
def evolve_gradient_booster_pipeline_reg(X,y,num_generations=10,num_chroms=20,
                        elite_frac=0.2,tol=1e-2,
                        train_per=0.75):
    
    chroms = make_gradient_booster_pipeline_chroms(X,y,
                                                   num_chroms=num_chroms,
                                                   mode='reg')
    min_loss = 1000
    max_fitness = []
    for gen in range(num_generations):
        elites = []
        fitness = []
        models = []

        for chrom in range(num_chroms):
            K=chroms[chrom]['K']
            X_new = sk.feature_selection.SelectKBest(chroms[chrom]['score_function'],k=K).fit_transform(X,y)
            X_new = chroms[chrom]['feature_scale'].fit_transform(X)
            X_train,X_test,y_train,y_test = train_test_split(X_new,y,
                                                             train_per=train_per)
            model = make_gradient_booster_reg(lr=chroms[chrom]['lr'],
                                              estimators=chroms[chrom]['estimators'],
                                              depth=chroms[chrom]['depth'])
            model.fit(X_train,y_train)
            y_train_pred = predict(model,X_train)
            y_test_pred = predict(model,X_test)
            loss = sk_mse_loss(y_true=y_test,y_pred=y_test_pred)
            score = pipeline_fitness_regression(y_true=y_test,y_pred=y_test_pred)
            print('Generation: [{}/{}] | Chromosome: [{}/{}]'.format(gen+1,num_generations,chrom+1,num_chroms))
            print('loss: {:4f}'.format(loss))
            print('fitness: {:4f}'.format(score))
            fitness.append(score)
            if loss < min_loss:
               min_loss = loss
        max_fitness.append(max(fitness))
        print("Generation {} min loss: {}".format(gen+1,min_loss))
        num_elites = int(np.ceil(elite_frac*num_chroms))
        fitness_sorted = np.argsort(fitness)
        chroms_sorted = [i for _,i in sorted(zip(fitness_sorted,chroms))]
        chroms_sorted = chroms_sorted[::-1]
        chroms = chroms_sorted[0:num_elites]
        new_chroms = make_gradient_booster_pipeline_chroms(X,y,num_chroms=num_chroms-len(chroms),
                                                           mode='reg')
        chroms.extend(new_chroms)
        if min_loss <= tol:
            print('Solved! - model with under {} loss found!'.format(loss))
            break
        
    plt.title('Fitness')
    plt.plot(np.arange(0,len(max_fitness)),max_fitness)
    plt.xlabel("Generations")
    plt.show()
    return {'models': models}


def evolve_nn_regression_pipeline(X,y,
                                  num_generations=10,
                                  num_chroms=20,
                                  epochs=10,
                                  elite_frac=0.2,
                                  tol=1e-3,
                                  train_per=0.75):
    
    if len(y.shape) == 1:
        out_size = 1
    if len(y.shape) != 1:
        out_size = y.shape[1]
    chroms, genomes = make_nn_pipeline_chroms(num_chroms=num_chroms,
                                              in_size=X.shape[1],
                            out_size=out_size)
   
    min_loss = 1000
    max_fitness = []
    criterion = nn.MSELoss()
    for gen in range(num_generations):
        elites = []
        fitness = []
        models = []
        for chrom in range(num_chroms):
            X_new = genomes[chrom]['feature_scale'].fit_transform(X)
            model = chroms[chrom]
            train_losses = []
            opt = genomes[chrom]['optimizer']
            if opt == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(),
                                      lr=genomes[chrom]['lr'],
                                      momentum=genomes[chrom]['momentum'])
            if opt == 'adam':
                optimizer = torch.optim.Adam(model.parameters(),
                                       lr=genomes[chrom]['lr'])
            if opt == 'rmsprop':
                optimizer = torch.optim.RMSprop(model.parameters(),
                                      lr=genomes[chrom]['lr'],
                                      momentum=genomes[chrom]['momentum'])
                
            X_train,X_test,y_train,y_test = train_test_split(X_new,y,
                                                             train_per=train_per)
                
            for epoch in range(epochs):
                
                y_pred = model.forward(torch.Tensor(X_train))
                optimizer.zero_grad()
                train_loss = criterion(y_pred,torch.Tensor(y_train))
                train_losses.append(train_loss.item())
                train_loss.backward()
                optimizer.step()
                print('Generation: [{}/{}] | Chromosome: [{}/{}]'.format(gen+1,num_generations,chrom+1,num_chroms))
                print('Epoch: [{}/{}]'.format(epoch+1,epochs))
                print('loss: {:3f}'.format(train_loss.item()))
                if train_loss.item() >= 100000:
                    print('DANG IT BOBBY, the loss is too high, aborting training...')
                    score = 0
                    break
                
            model.eval()
            y_test_pred = model(torch.Tensor(X_test))
            test_loss = criterion(y_test_pred,torch.Tensor(y_test))
            score = pipeline_fitness_regression(y_true=y_test,
                                                y_pred=y_test_pred.detach().numpy(),
                                                neural_net=True,
                                                genome=genomes[chrom])
            print('Test Loss: {:3f}'.format(test_loss.item()))
            print('fitness: {:4f}'.format(score))
            
            fitness.append(score)
            models.append(model)
            if test_loss.item() < tol:
                min_loss = test_loss.item()
            if min_loss <= tol:
                print('Solved! - model with under {} loss found!'.format(min_loss))
                break
        max_fitness.append(max(fitness))
        print("Generation {} min loss: {}".format(gen+1,min_loss))
        num_elites = int(np.ceil(elite_frac*num_chroms))
        fitness_sorted = np.argsort(fitness)
        chroms_sorted = [i for _,i in sorted(zip(fitness_sorted,chroms))]
        genomes_sorted = [i for _,i in sorted(zip(fitness_sorted,genomes))]
        chroms_sorted = chroms_sorted[::-1]
        genomes_sorted = genomes_sorted[::-1]
        chroms = chroms_sorted[0:num_elites]
        
        genomes = genomes_sorted[0:num_elites]
        new_chroms, new_genomes = make_nn_pipeline_chroms(num_chroms=num_chroms-len(chroms),
                                                           mode='reg',in_size=X.shape[1],
                                                           out_size=out_size)
        chroms.extend(new_chroms)
        genomes.extend(new_genomes)
        pop = (chroms,genomes)
        chroms, genomes = pop
    
    plt.title('Fitness')
    plt.plot(np.arange(0,len(max_fitness)),max_fitness)
    plt.xlabel("Generations")
    plt.show()
    return {'models': models}
                
                
                
      