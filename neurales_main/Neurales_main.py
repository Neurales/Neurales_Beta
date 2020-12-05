# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 19:47:02 2019

@author: XZ-WAVE
"""

from abc import ABCMeta, abstractmethod
import json 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sklearn as sk
import collections
from sklearn import naive_bayes
from sklearn import neural_network
from sklearn import linear_model
from sklearn import feature_selection 
from sklearn import ensemble
from sklearn import metrics
from sklearn import neighbors
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import cluster
from sklearn import svm
from sklearn import tree
from sklearn import pipeline
from sklearn import manifold
from sklearn import decomposition
from neurales_utils import *


class RegressionPipeline_csvA():
    __metaclass__= ABCMeta
    def __init__(self,filepath,feat_eng_steps,train_per=0.8,data_drop='skip'):
        super(RegressionPipeline_csvA,self).__init__()
        self.steps=feat_eng_steps
        self.pipeline=sk.pipeline.make_pipeline(*self.steps)
        self.filepath=filepath
        self.data_drop=data_drop
        self.df=pd.read_csv(self.filepath)
        self.df=self.df.dropna()
        if self.data_drop=='feature':
            print(self.df.keys())
            self.selection=input('enter in a column (i.e. variable/feature) to drop')
            self.df=self.df.drop(columns=self.selection.strip(),axis=0)
        if self.data_drop=='sample':
            self.selection=input('enter in a row (i.e. variable/feature) to drop')
            self.df=self.df.drop([int(self.selection)],axis=0)
        targets=[]
        for i in range(len(self.df.keys())):
            targets.append(self.df.keys()[i])
        print(targets)
        self.target=input('which would you like to predict/forecast?')
        self.target=self.target.strip()
        if self.target not in targets:
            print("this isn't in the list of choices")
        if self.target in targets: 
            user_choice=targets.index(self.target)+1
        
        self.target_idx=self.df.columns.get_loc(self.target)
        
        self.X=self.df.drop(self.target,1)# N samples x K features/variables
        self.y=self.df[self.target]
        self.X=np.array(self.X)
        self.y=np.array(self.y)
        self.dataset={'data': self.X, 
                      'target': self.y, 
                      'labels': targets, 
                      'index': self.target_idx}
        self.train_per=train_per
        self.test_per=1-self.train_per
        self.X_train, self.X_test, self.y_train, self.y_test=sk.model_selection.train_test_split(self.X,
                                                                                                 self.y,
                                                                                                test_size=self.test_per)   
    
    def feature_analysis(self):
        self.feature_model=sk.ensemble.ExtraTreesRegressor(n_estimators=20,random_state=0)
        self.feature_model.fit(self.X,self.y)
        self.rank=self.feature_model.feature_importances_
        self.feature_names=self.df.drop([self.target],axis=1)
        self.series=pd.Series(data=self.rank,index=self.feature_names.keys())
        if len(self.series)<=10:
            self.feature_ranks=self.series.sort_values()
            self.feature_ranks.plot(kind='bar',title='top features')
        
        if len(self.series)>10:
            self.feature_ranks=self.series.sort_values()
            self.top10=self.feature_ranks[0:10]
            self.top10.plot(kind='bar',title='top 10 features')
        return out2json(self.series)
            
    @abstractmethod
    def create_model(self):
        pass
    
    def train(self):
        self.clf.fit(self.X_train,self.y_train)
        self.pred=self.clf.predict(self.X_train)
        error=np.linalg.norm(self.pred-self.y_train)/len(self.pred)
        self.train_loss=np.around(error,decimals=4)
        print('train loss: {}'.format(self.train_loss))
        return out2json(self.train_loss)
    
    def validate(self):
        self.eval=self.clf.predict(self.X_test)
        error=np.linalg.norm(self.eval-self.y_test)/len(self.eval)
        self.val_loss=np.around(error,decimals=4)
        print('validation loss: {}'.format(self.val_loss))
        return out2json(self.val_loss)
    
    def cross_validate(self,k_fold=5): #Restrict user from entering 1 on the front end 
        self.k_fold=k_fold
        self.cross_val_scores=sk.model_selection.cross_val_score(self.clf,self.X,self.y,cv=self.k_fold)
        return out2json(self.cross_val_scores)
    
    def anomaly_detection(self,new_data):
        self.new_data=new_data
        self.lof=sk.neighbors.LocalOutlierFactor(n_neighbors=10)
        self.lof_scores=self.lof.fit_predict(self.new_data)
        counter=collections.Counter(self.lof_scores)
        print('anomalies: {}/{}'.format(counter[-1],counter[1]))
        return out2json(self.lof_scores)
       
    
    def predict(self,new_data):
        self.new_data=new_data
        self.prediction=1*np.around(self.clf.predict(self.new_data),decimals=3)
        for sample in range(len(self.new_data)):
            print('{}: prediction {}'.format(sample+1,self.prediction[sample]))
        return out2json(self.prediction)
    
    def tsne(self,mode='val',dim=2): #adding class lables to plot legend to be done on the front end GUI
        self.mode=mode
        self.tsne_dim=dim
        if self.mode=='train':
                self.X_embedded=sk.manifold.TSNE(n_components=self.tsne_dim).fit_transform(self.X_train)
           
                self.tsne_score=sk.manifold.trustworthiness(X=self.X_train,
                                                   X_embedded=self.X_embedded)
        if self.mode=='val':
            self.X_embedded=sk.manifold.TSNE(n_components=self.tsne_dim).fit_transform(self.X_test)
        
            self.tsne_score=sk.manifold.trustworthiness(X=self.X_test,
                                                   X_embedded=self.X_embedded)
        self.sc=1*np.around(self.tsne_score,decimals=3)
        if self.tsne_dim==2: 
            plt.title('TSNE Plot')
            plt.xlabel('Embedding Trustworthiness Score: {}'.format(self.sc))
            plt.scatter(self.X_embedded[:,0],self.X_embedded[:,1],c='blue')
            plt.show()
        return out2json(self.X_embedded)
        
       
            
    def isomap(self,iso_neighbors=5,mode='val',dim=2): #adding class lables to plot legend to be done on the front end GUI
        self.iso_neighbors=iso_neighbors
        self.mode=mode
        self.isomap_dim=dim
        if self.mode=='train':
                self.X_embedded=sk.manifold.Isomap(n_neighbors=self.iso_neighbors,
                                                   n_components=self.isomap_dim).fit_transform(self.X_train)
                self.iso_score=sk.manifold.trustworthiness(X=self.X_train,
                                                   X_embedded=self.X_embedded,
                                                   n_neighbors=self.iso_neighbors)
                
        if self.mode=='val':
            self.X_embedded=sk.manifold.Isomap(n_neighbors=self.iso_neighbors,
                                               n_components=self.isomap_dim).fit_transform(self.X_test)
        
            self.iso_score=sk.manifold.trustworthiness(X=self.X_test,
                                                   X_embedded=self.X_embedded,
                                                   n_neighbors=self.iso_neighbors)
        
        self.sc=1*np.around(self.iso_score,decimals=3)
        if self.isomap_dim==2: 
            plt.title('Isomap Plot')
            plt.xlabel('Embedding Trustworthiness Score: {}'.format(self.sc))
            plt.scatter(self.X_embedded[:,0],self.X_embedded[:,1],c='red')
            plt.show()
        return out2json(self.sc)
    
    def LLE(self,lle_neighbors=5,mode='val',dim=2): #adding class lables to plot legend to be done on the front end GUI
        self.lle_neighbors=lle_neighbors
        self.mode=mode
        self.lle_dim=dim
        if self.mode=='train':
                self.X_embedded=sk.manifold.LocallyLinearEmbedding(n_neighbors=self.lle_neighbors,
                                                   n_components=self.lle_dim).fit_transform(self.X_train)
                self.lle_score=sk.manifold.trustworthiness(X=self.X_train,
                                                   X_embedded=self.X_embedded,
                                                   n_neighbors=self.lle_neighbors)
                
        if self.mode=='val':
            self.X_embedded=sk.manifold.LocallyLinearEmbedding(n_neighbors=self.lle_neighbors,
                                               n_components=self.lle_dim).fit_transform(self.X_test)
        
            self.lle_score=sk.manifold.trustworthiness(X=self.X_test,
                                                   X_embedded=self.X_embedded,
                                                   n_neighbors=self.lle_neighbors)
        
        self.sc=1*np.around(self.lle_score,decimals=3)
        if self.lle_dim==2: 
            plt.title('Locally Linear Embedding Plot')
            plt.xlabel('Embedding Trustworthiness Score: {}'.format(self.sc))
            plt.scatter(self.X_embedded[:,0],self.X_embedded[:,1],c='green')
            plt.show()
        return out2json(self.sc)
    
    
    def PCA(self,n_components=10,mode='val',dim=2):
        self.mode=mode
        self.pcs=n_components
        self.pca=sk.decomposition.PCA(n_components=self.pcs)
        if self.mode=='train':
            self.pca_model=self.pca.fit(self.X_train)
            self.pca_scores=self.pca_model.score_samples(self.X_train)
            self.pca_transform=self.pca.fit_transform(self.X_train)
            
        if self.mode=='val':
            self.pca_model=self.pca.fit(self.X_test)
            self.pca_scores=self.pca_model.score_samples(self.X_test)
            self.pca_transform=self.pca.fit_transform(self.X_test)
        
        
        self.evr=1*np.around(self.pca_model.explained_variance_ratio_,decimals=4)
        if self.pcs<=10:
            plt.scatter(self.pca_transform[:,0],self.pca_transform[:,1],c='magenta')
            plt.title('PCA plot')
            plt.show()
            
            plt.plot(np.arange(1,len(np.cumsum(self.evr))+1),np.cumsum(self.evr))
            plt.xticks(np.arange(1,len(np.cumsum(self.evr))+1))
            plt.show()
        
        if self.pcs>10:
            plt.scatter(self.pca_transform[:,0],self.pca_transform[:,1],c='magenta')
            plt.title('PCA plot')
            plt.show()
            
            plt.plot(np.arange(1,len(np.cumsum(self.evr))+1),np.cumsum(self.evr),c='purple')
            plt.title('EVR plot')
            plt.xlabel('number of components')
            plt.ylabel('explained variance ratio')
            plt.show()
            
        return out2json(self.evr)
    
    
    
    def dump2csv(self):
        pass
    

class NeuralNetRegressor_csvA(RegressionPipeline_csvA):
    
    def create_model(self,num_layers=2,hidden_size=50,lr=1e-4):
        
        self.num_layers=num_layers
        self.hidden_size=hidden_size
        self.lr=lr
        self.model=sk.neural_network.MLPRegressor(alpha=self.lr,hidden_layer_sizes=(np.repeat(self.hidden_size,self.num_layers)))
        self.steps.append(self.model)
        self.clf=sk.pipeline.make_pipeline(*self.steps)
        return model2json(self.clf)
    
         
class RegTree_csvA(RegressionPipeline_csvA):
    
    def create_model(self,depth=5):
        self.depth=depth
        self.model=sk.tree.DecisionTreeRegressor(max_depth=self.depth)
        self.steps.append(self.model)
        self.clf=sk.pipeline.make_pipeline(*self.steps)
        return model2json(self.clf)
    
    
class SVR_Linear_csvA(RegressionPipeline_csvA):
    
    def create_model(self,C=1):
        self.C=C
        self.model=sk.svm.LinearSVR(C=self.C)
        self.steps.append(self.model)
        self.clf=sk.pipeline.make_pipeline(*self.steps)
        return model2json(self.clf)


class SVR_Poly_csvA(RegressionPipeline_csvA):
    
    def create_model(self,degree=3,C=1,gamma=0.2):
        self.C=C
        self.degree=degree
        self.gamma=gamma
        self.model=sk.svm.SVR(kernel='poly',degree=self.degree,C=self.C,gamma=self.gamma)
        self.steps.append(self.model)
        self.clf=sk.pipeline.make_pipeline(*self.steps)
        return model2json(self.clf)
    
    
class SVR_RBF_csvA(RegressionPipeline_csvA):
    
    def create_model(self,C=1,gamma=0.2):
        self.C=C
        self.gamma=gamma
        self.model=sk.svm.SVR(C=self.C,gamma=self.gamma)
        self.steps.append(self.model)
        self.clf=sk.pipeline.make_pipeline(*self.steps)
        return model2json(self.clf)


class SVR_Sigmoid_csvA(RegressionPipeline_csvA):
    
    def create_model(self,C=1,gamma=0.2):
        self.C=C
        self.model=sk.svm.SVR(kernel='sigmoid',C=self.C)
        self.steps.append(self.model)
        self.clf=sk.pipeline.make_pipeline(*self.steps)
        return model2json(self.clf)
    

class RandomForestReg_csvA(RegressionPipeline_csvA):
    
    def create_model(self,num_trees=20,depth=7):
        self.num_trees=num_trees
        self.depth=7
        self.model=sk.ensemble.RandomForestRegressor(n_estimators=num_trees,max_depth=7)
        self.steps.append(self.model)
        self.clf=sk.pipeline.make_pipeline(*self.steps)
        return model2json(self.clf)
    
class ClassificationPipeline_csvA():
    __metaclass__= ABCMeta
    def __init__(self,filepath,feat_eng_steps,train_per=0.8,data_drop='skip'):
        super(ClassificationPipeline_csvA,self).__init__()
        self.steps=feat_eng_steps
        self.pipeline=sk.pipeline.make_pipeline(*self.steps)
        self.filepath=filepath
        self.data_drop=data_drop
        self.df=pd.read_csv(self.filepath)
        cols_to_remove=[]
        self.df = self.df[[col for col in self.df.columns if col not in cols_to_remove]]
        self.df=self.df.dropna()
        if self.data_drop=='feature':
            print(self.df.keys())
            self.selection=input('enter in a column (i.e. variable/feature) to drop')
            self.df=self.df.drop(columns=self.selection.strip(),axis=1)
        if self.data_drop=='sample':
            self.selection=input('enter in a row (i.e. data sample) to drop')
            self.df=self.df.drop([int(self.selection)],axis=0)
        self.targets=[]
        for i in range(len(self.df.keys())):
            self.targets.append(self.df.keys()[i])
        print(self.targets)
        self.target=input('which would you like to predict/forecast?')
        self.target=self.target.strip()
        if self.target not in self.targets:
            print("this isn't in the list of choices")
        if self.target in self.targets: 
            user_choice=self.targets.index(self.target)+1
       
        self.target_idx=self.df.columns.get_loc(self.target)
        
        self.X=self.df.drop(self.target,1)# N samples x K features/variables
        
        self.y=self.df[self.target]
        self.X=np.array(self.X)
        self.y=np.array(self.y)
        self.labels=np.unique(self.targets)
        labels=[]
        for i in range(len(labels)): 
            labels.append(labels[i])
        self.dataset={'data': self.X, 
                      'target': self.y, 
                      'labels': self.labels, 
                      'index': self.target_idx}
        self.train_per=train_per
        self.test_per=1-self.train_per
        self.X_train, self.X_test, self.y_train, self.y_test=sk.model_selection.train_test_split(self.X,
                                                                                                 self.y,
                                                                                               test_size=self.test_per)
    def class_breakdown(self):
        self.class_labels=pd.Series(data=self.y)
        plt.title('class breakdown')
        plt.hist(self.df[self.target],label=pd.unique(self.df[self.target]))
        plt.show()
        return out2json(self.class_labels)
         
    
    def feature_analysis(self):
        self.feature_model=sk.ensemble.ExtraTreesClassifier(n_estimators=20,random_state=0)
        self.feature_model.fit(self.X,self.y)
        self.rank=self.feature_model.feature_importances_
        self.feature_names=self.df.drop([self.target],axis=1)
        self.series=pd.Series(data=self.rank,index=self.feature_names.keys())
        if len(self.series)<=10:
            self.feature_ranks=self.series.sort_values()
            self.series.plot(kind='bar',title='top features')
        
        if len(self.series)>10:
            self.feature_ranks=self.series.sort_values()
            self.top10=self.feature_ranks[0:10]
            self.top10.plot(kind='bar',title='top 10 features')
        return out2json(self.series)
    
    @abstractmethod
    def create_model(self):
        pass
    
    def train(self):
        self.clf.fit(self.X_train,self.y_train)
        self.pred=self.clf.predict(self.X_train)
        error=sum(np.equal(self.pred,self.y_train))
        acc=np.around(error,decimals=4)
        print('correct predictions: {}/{}'.format(acc,len(self.y_train)))
        self.train_acc=(100*acc)/len(self.y_train)
        print('train accuracy: {} %'.format(np.around(self.train_acc,decimals=4)))
        return out2json(self.train_acc)
    
    def validate(self):
        self.eval=self.clf.predict(self.X_test)
        error=sum(np.equal(self.eval,self.y_test))
        acc=np.around(error,decimals=4)
        print('correct predictions: {}/{}'.format(acc,len(self.y_test)))
        self.val_acc=(100*acc)/len(self.y_test)
        print('validation accuracy: {}'.format(np.around(self.val_acc,decimals=4)))
        return out2json(self.val_acc)
    
    def cross_validate(self,k_fold=5):
        self.k_fold=k_fold
        self.cross_val_scores=sk.model_selection.cross_val_score(self.clf,self.X,self.y,cv=self.k_fold)
        for i in range(len(self.cross_val_scores)):
            print('fold {} | accuracy: {} %'.format(i+1,100*np.around(self.cross_val_scores[i],decimals=4)))
        return out2json(self.cross_val_scores)
    
    def predict(self,new_data):
        self.new_data=new_data
        self.prediction=self.clf.predict(self.new_data)
        for sample in range(len(self.new_data)):
            print('{}: prediction: {}'.format(sample+1,self.prediction[sample]))
       
        return out2json(self.prediction)
    
    def pred_prob(self,new_data):
        self.new_data=new_data
        self.prob=self.clf.predict_proba(self.new_data)
        for sample in range(len(self.prob)):
            print('{}: class probabilities: {}'.format(sample+1,1*np.around(self.prob[sample],decimals=3)))
        return out2json(self.prob)
    
    def anomaly_detection(self,new_data):
        self.new_data=new_data
        self.lof=sk.neighbors.LocalOutlierFactor(n_neighbors=10)
        self.lof_scores=self.lof.fit_predict(self.new_data)
        counter=collections.Counter(self.lof_scores)
        print('anomalies: {}/{}'.format(counter[-1],counter[1]))
        return out2json(self.lof_scores)
    
    def precision_recall_scores(self,mode):
        self.mode=mode
        if self.mode=='train':
            self.precision=sk.metrics.precision_score(self.y_train,self.pred,average=None)
            self.recall=sk.metrics.recall_score(self.y_train,self.pred,average=None)
            self.f1_score=sk.metrics.f1_score(self.y_train,self.pred,average=None)
        if self.mode=='val':
            self.precision=sk.metrics.precision_score(self.y_test,self.eval,average=None)
            self.recall=sk.metrics.recall_score(self.y_test,self.eval,average=None)
            self.f1_score=sk.metrics.f1_score(self.y_test,self.eval,average=None)
        
        self.metrics={'precision': out2json(self.precision), 
                      'recall': out2json(self.recall),
                      'F1 score': out2json(self.f1_score)}
        print('precision: {}'.format(1*np.around(self.precision,decimals=3)))
        print('recall: {}'.format(1*np.around(self.recall,decimals=3)))
        print('F1: {}'.format(1*np.around(self.f1_score,decimals=3)))
        return json.dumps(self.metrics)
    
    def mAP(self):
        self.mAP=sk.metrics.average_precision_score(self.y_test,self.eval)
        return out2json(self.mAP)
    
    def ConfusionMat(self,mode):
        if mode=='train':
            self.conf_mat=sk.metrics.confusion_matrix(self.y_train,self.pred)
        if mode=='val':
            self.conf_mat=sk.metrics.confusion_matrix(self.y_test,self.eval)
        return self.conf_mat
    
    def adverse_impact_analysis(self):
        plt.title('Adverse Impact Breakdown')
        plt.pie(self.f1_score,labels=np.arange(0,len(self.f1_score)))
        plt.show()
        return out2json(self.f1_score)
    
    def tsne(self,mode='val',dim=2): #adding class lables to plot legend to be done on the front end GUI
        self.mode=mode
        self.tsne_dim=dim
        if self.mode=='train':
                self.X_embedded=sk.manifold.TSNE(n_components=self.tsne_dim).fit_transform(self.X_train)
                self.tsne_score=sk.manifold.trustworthiness(X=self.X_train,
                                                   X_embedded=self.X_embedded)
        if self.mode=='val':
            self.X_embedded=sk.manifold.TSNE(n_components=self.tsne_dim).fit_transform(self.X_test)
            self.tsne_score=sk.manifold.trustworthiness(X=self.X_test,
                                                   X_embedded=self.X_embedded)
        
       
        self.sc=1*np.around(self.tsne_score,decimals=3)
        if self.tsne_dim==2: 
            plt.title('TSNE Plot')
            plt.xlabel('Embedding Trustworthiness Score: {}'.format(self.sc))
            plt.scatter(self.X_embedded[:,0],self.X_embedded[:,1],c='blue')
            plt.show()
        
       
            
    def isomap(self,iso_neighbors=5,mode='val',dim=2): #adding class lables to plot legend to be done on the front end GUI
        self.iso_neighbors=iso_neighbors
        self.mode=mode
        self.isomap_dim=dim
        if self.mode=='train':
                self.X_embedded=sk.manifold.Isomap(n_neighbors=self.iso_neighbors,
                                                   n_components=self.isomap_dim).fit_transform(self.X_train)
           
                self.iso_score=sk.manifold.trustworthiness(X=self.X_train,
                                                   X_embedded=self.X_embedded,
                                                   n_neighbors=self.iso_neighbors)
        if self.mode=='val':
            self.X_embedded=sk.manifold.Isomap(n_neighbors=self.iso_neighbors,
                                               n_components=self.isomap_dim).fit_transform(self.X_test)
        
            self.iso_score=sk.manifold.trustworthiness(X=self.X_test,
                                                   X_embedded=self.X_embedded,
                                                   n_neighbors=self.iso_neighbors)
        self.sc=1*np.around(self.iso_score,decimals=3)
        if self.isomap_dim==2: 
            plt.title('Isomap Plot')
            plt.xlabel('Embedding Trustworthiness Score: {}'.format(self.sc))
            plt.scatter(self.X_embedded[:,0],self.X_embedded[:,1],c='red')
            plt.show()
    
    def LLE(self,lle_neighbors=5,mode='val',dim=2): #adding class lables to plot legend to be done on the front end GUI
        self.lle_neighbors=lle_neighbors
        self.mode=mode
        self.lle_dim=dim
        if self.mode=='train':
                self.X_embedded=sk.manifold.LocallyLinearEmbedding(n_neighbors=self.lle_neighbors,
                                                   n_components=self.lle_dim).fit_transform(self.X_train)
                self.lle_score=sk.manifold.trustworthiness(X=self.X_train,
                                                   X_embedded=self.X_embedded,
                                                   n_neighbors=self.lle_neighbors)
                
        if self.mode=='val':
            self.X_embedded=sk.manifold.LocallyLinearEmbedding(n_neighbors=self.lle_neighbors,
                                               n_components=self.lle_dim).fit_transform(self.X_test)
            self.lle_score=sk.manifold.trustworthiness(X=self.X_test,
                                                   X_embedded=self.X_embedded,
                                                   n_neighbors=self.lle_neighbors)
       
        self.sc=1*np.around(self.lle_score,decimals=3)
        
        if self.lle_dim==2: 
           
            plt.title('Locally Linear Embedding Plot')
            plt.xlabel('Embedding Trustworthiness Score: {}'.format(self.sc))
            plt.scatter(self.X_embedded[:,0],self.X_embedded[:,1],c='green')
            plt.show()
    
    def PCA(self,n_components=10,mode='val',dim=2):
        self.mode=mode
        self.pcs=n_components
        self.pca=sk.decomposition.PCA(n_components=self.pcs)
        if self.mode=='train':
            self.pca_model=self.pca.fit(self.X_train)
            self.pca_scores=self.pca_model.score_samples(self.X_train)
            self.pca_transform=self.pca.fit_transform(self.X_train)
            
        if self.mode=='val':
            self.pca_model=self.pca.fit(self.X_test)
            self.pca_scores=self.pca_model.score_samples(self.X_test)
            self.pca_transform=self.pca.fit_transform(self.X_test)
        
        self.evr=1*np.around(self.pca_model.explained_variance_ratio_,decimals=4)
        if self.pcs<=10:
            plt.scatter(self.pca_transform[:,0],self.pca_transform[:,1],c='magenta')
            plt.title('PCA plot')
            plt.show()
            
            plt.plot(np.arange(1,len(np.cumsum(self.evr))+1),np.cumsum(self.evr))
            plt.xticks(np.arange(1,len(np.cumsum(self.evr))+1))
            plt.xlabel('number of components')
            plt.ylabel('explained variance ratio')
            plt.show()
        
        if self.pcs>10:
            plt.scatter(self.pca_transform[:,0],self.pca_transform[:,1])
            plt.title('PCA plot')
            plt.show()
            
            plt.plot(np.arange(1,len(np.cumsum(self.evr))+1),np.cumsum(self.evr),c='purple')
            plt.title('EVR')
            plt.xlabel('number of components')
            plt.ylabel('explained variance ratio')
            plt.show()
            
        return out2json(self.evr)
        
    def dump2csv(self):
        pass
    
    
class TreeClassifier_csvA(ClassificationPipeline_csvA):
    
    def create_model(self,depth=5):
        self.depth=depth
        self.model=sk.tree.DecisionTreeClassifier(max_depth=self.depth)
        self.steps.append(self.model)
        self.clf=sk.pipeline.make_pipeline(*self.steps)
        return model2json(self.clf)
    
    
class RandomForestClassif_csvA(ClassificationPipeline_csvA):
    
    def create_model(self,num_trees=50,depth=5):
        self.depth=depth
        self.num_trees=num_trees
        self.model=sk.ensemble.RandomForestClassifier(n_estimators=self.num_trees,max_depth=self.depth)
        self.steps.append(self.model)
        self.clf=sk.pipeline.make_pipeline(*self.steps)
        return model2json(self.clf)
  
    
class SVC_Linear_csvA(ClassificationPipeline_csvA):
    
    def create_model(self,C=1):
        self.C=C
        self.model=sk.svm.SVC(kernel='linear',C=self.C,probability=True)
        self.steps.append(self.model)
        self.clf=sk.pipeline.make_pipeline(*self.steps)
        return model2json(self.clf)
    
        
    
class SVC_Sigmoid_csvA(ClassificationPipeline_csvA):
    
    def create_model(self,C=1):
        self.C=C
        self.model=sk.svm.SVC(kernel='sigmoid',C=self.C)
        self.steps.append(self.model)
        self.clf=sk.pipeline.make_pipeline(*self.steps)
        return model2json(self.clf)
    
class SVC_RBF_csvA(ClassificationPipeline_csvA):
    
    def create_model(self,C=1,gamma=0.2):
        self.C=C
        self.gamma=gamma
        self.model=sk.svm.SVC(kernel='rbf',C=self.C,gamma=self.gamma)
        self.steps.append(self.model)
        self.clf=sk.pipeline.make_pipeline(*self.steps)
        return model2json(self.clf)
    
    
class SVC_Poly_csvA(ClassificationPipeline_csvA):
    
    def create_model(self,degree=3,C=1,gamma=0.2):
        self.C=C
        self.gamma=gamma
        self.degree=degree
        self.model=sk.svm.SVC(kernel='poly',degree=self.degree,C=self.C,gamma=self.gamma)
        self.steps.append(self.model)
        self.clf=sk.pipeline.make_pipeline(*self.steps)
        return model2json(self.clf)
    
              
class NeuralNetClassif_csvA(ClassificationPipeline_csvA):
    
    def create_model(self,num_layers=2,hidden_size=50,lr=1e-4):
        self.num_layers=num_layers
        self.hidden_size=hidden_size
        self.lr=lr
        self.model=sk.neural_network.MLPClassifier(alpha=self.lr,hidden_layer_sizes=(np.repeat(self.hidden_size,self.num_layers)))
        self.steps.append(self.model)
        self.clf=sk.pipeline.make_pipeline(*self.steps)
        return model2json(self.clf)


class LogisticReg_csvA(ClassificationPipeline_csvA):
    
    def create_model(self,C=1):
        self.C=C
        self.model=sk.linear_model.LogisticRegression(C=self.C)
        self.steps.append(self.model)
        self.clf=sk.pipeline.make_pipeline(*self.steps)
        return model2json(self.clf)
    

class NaiveBayesClf_csvA(ClassificationPipeline_csvA):
    
    def create_model(self,choice='gaussian'):
        if choice=='gaussian':
            self.model=sk.naive_bayes.GaussianNB()
        if choice=='multinomial':
            self.model=sk.naive_bayes.MultinomialNB()
        if choice=='complement':
            self.model=sk.naive_bayes.ComplementNB()
        if choice=='bernoulli':
            self.model=sk.naive_bayes.BernoulliNB()
        if choice=='categorical':
            self.model=sk.naive_bayes.CategoricalNB()
        self.steps.append(self.model)
        self.clf=sk.pipeline.make_pipeline(*self.steps)
        return model2json(self.clf)
    
class Neurales_memory(ClassificationPipeline_csvA):
    def __init__(self,model):
        super(Neurales_memory,self).__init__()
        self.model=model
        if type(self.model)=='sklearn.tree._classes.DecisionTreeClassifier':
            return 0

class ArkiTEKT_clf(ClassificationPipeline_csvA):
    
    def create_model(self):
        self.num_trees=50
        self.depth=7
        #self.model=sk.ensemble.RandomForestClassifier(n_estimators=self.num_trees,max_depth=self.depth)
        self.steps=[sk.preprocessing.StandardScaler(),sk.decomposition.PCA(n_components=int(len(self.X)/4))]
        self.model=self.steps.append(sk.svm.SVC(C=1.0,kernel='linear'))
        self.clf=sk.pipeline.make_pipeline(*self.model)
        '''
        self.clf=sk.pipeline.make_pipeline([sk.preprocessing.StandardScaler(),
                                                 sk.decomposition.PCA(n_components=int(len(self.X)/4)),
                                                 self.model])
        '''
    
    
    
    

