# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 01:01:52 2020

@author: XZ-WAVE
"""

import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold
from sklearn import tree
from sklearn import decomposition
from sklearn import ensemble
from sklearn import svm
from sklearn import metrics
from sklearn import naive_bayes
from sklearn import linear_model
from sklearn import model_selection 
from sklearn import datasets
from sklearn import neural_network
from sklearn import cluster
from sklearn import neighbors
from sklearn import pipeline
import pandas as pd
from PIL import Image

import os
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# =============== utils =========================
colors = ['r','g','b','c','m','y','k','orange','purple','brown','salmon','darkorchid']

make_color_list = lambda colors,num_colors: list(np.random.choice(colors,num_colors,replace=False))

get_feature_names_csv = lambda csv_file: list(pd.read_csv(csv_file).keys()) #gets feature names from csv directly

get_feature_names_df = lambda df: list(df.keys()) #gets feature names from resulting data frame

label_binarizer = lambda target: sk.preprocessing.LabelBinarizer().fit_transform(target)

multilabel_binarizer = lambda target: sk.preprocessing.MultiLabelBinarizer().fit_transform(target)

label_encoder = lambda target: sk.preprocessing.LabelEncoder().fit_transform(target)

drop_column = lambda df,name: df.drop(columns=name) #drops a feature/column

drop_row = lambda df,idx: df.drop([idx]) #drops a single row

load_csv = lambda csv_file: pd.read_csv(csv_file) 

load_excel = lambda excel_file: pd.read_excel(excel_file) #reads in excel file

load_json = lambda file: pd.read_json(file) #reads in json file, alpha version, work for iris json, different arguments can be passed to parse json files differently

df_types = lambda csv_file: load_csv(csv_file).dtypes #gets data types for all the columns. 



def get_strings_idx(csv_file):
    '''
    this function reads in a csv file and gets the indices of the columns that contain strings and non-numeric formats
    '''
    types = df_types(csv_file)
    str_idxs = []
    for t in range(len(types)):
        if types[t] == object:
            str_idxs.append(t)
        if types[t] == str:
            str_idxs.append(t)
    return str_idxs

    
def remove_cols_with_strings(csv_file): 
    '''
    this function removes columns with strings and non-numeric data types as it is loaded in. 
    '''
    df = load_csv(csv_file)
    df = df[df.T[df.dtypes!=np.object].index]
    return df

    

def drop_multiple_cols(df):
    '''
    =============================================================
    df: Pandas DataFrame object 
    data frame is loaded, then the user is given a list of all the features 
    loaded in the data frame. They then select how many of those they want to drop, 
    and then actually select the ones to drop
    =============================================================
    '''
    cols = []
    print(df.keys())
    num_cols = int(input('enter in the number of feature columns you wish to drop:\t'))
    for name in range(num_cols):
        feature = str(input('enter in a feature column to drop: ')).strip() #have to be careful with keys that DO have spaces
        cols.append(feature)
    new_df = df.drop(cols,axis=1)
    return new_df

def data_target_split(filepath):
    '''
    ===============================
    filepath: string - path to the csv file of interest
    ===============================
    '''
    data = load_csv(filepath)
    data = drop_multiple_cols(data)
    
    print(data.keys())
    target = str(input('select a variable you want to predict: ')).strip()
    y = data[target]
    X = data.drop(columns=[target])
    return X,y

def data_target_split_idx(filepath):
    data = load_csv(filepath)
    data = drop_multiple_cols(data)
    print(data.keys())
    

def join_dfs(df1,df2): #two dataframes
    '''
    df1, df2: Pandas DataFrame objects
    function combines any number of columns from both DataFrames into a new dataset
    in the case where the datasets differ in the number of features and number of data points (common cases),
    the resulting DataFrame fills in zeros for columns and rows with null values. 
    '''
    cols_1, cols_2 = [],[]
    print(df1.keys())
    num_cols = int(input('enter in the number of feature columns you wish to drop:\t'))
    for name in range(num_cols):
        feature = str(input('enter in a feature column to drop: ')).strip() #have to be careful with keys that DO have spaces
        cols_1.append(feature)
    new_df1 = df1.drop(cols_1,axis=1)
    print(df2.keys())
    num_cols = int(input('enter in the number of feature columns you wish to drop:\t'))
    for name in range(num_cols):
        feature = str(input('enter in a feature column to drop: ')).strip() #have to be careful with keys that DO have spaces
        cols_2.append(feature)
    new_df2 = df2.drop(cols_1,axis=1)
    new_df = pd.concat([new_df1,new_df2],sort=False)
    new_df = new_df.fillna(0)
    return new_df


def df_numeric_subset(df,col_name,mode='less'):
    '''
    df: Pandas DataFrame object
    col_name: string - name of column to perform analysis on (note that the API will allow users to simply click/check the column of interest)
    mode: str - self-explanatory when going through the function
    function produces a subset of the original DataFrame based on the criteria in mode
    '''
    if mode == 'less':
        val = float(input('enter in lowest value:\t'))
        new_df = df.loc[df[col_name] > val]
    if mode == 'equal':
        val = float(input('enter in value:\t'))
        new_df=df.loc[df[col_name] == val]
    if mode == 'greater':
        val = float(input('enter in highest value:\t'))
        new_df = df.loc[df[col_name] < val]
    if mode == 'multiple':
        A = float(input('enter in the lowest value:\t'))
        B = float(input('enter in the highest value:\t'))
        new_df = df[col_name] >= A & df[col_name] <= B
    return new_df

def df_count_occurences(df,col_name,mode='less'):
    '''
    similar to df_numeric_subset, but counts occurences
    '''
    if mode == 'less':
        val = float(input('enter in lowest value:\t'))
        new_df = df.loc[df[col_name] > val]
        num = new_df[col_name].value_counts()
    if mode == 'equal':
        val = float(input('enter in lowest value:\t'))
        new_df=df.loc[df[col_name] == val]
        num = new_df[col_name].value_counts()
    if mode == 'greater':
        val = float(input('enter in lowest value:\t'))
        new_df = df.loc[df[col_name] < val]
        num = new_df[col_name].value_counts()
    if mode == 'multiple':
        A = float(input('enter in the lowest value:\t'))
        B = float(input('enter in the highest value:\t'))
        new_df = df[col_name] >= A & df[col_name] <= B
        num = new_df[col_name].value_counts()
    return num

def df_encode_strings(df):
    '''
    a function to encode strings
    the goal of this function is to be able to handle cases where there are say 4 city names
    like ['Tokyo', 'London', 'Paris', 'Berlin'] and encode them as [0,1,2,3]
    so a column that looks like ['Tokyo', 'Berlin', 'Berlin', 'London'] will be changed to
    [0,3,3,1]
    '''
    print(df.keys())
    col_name = str(input('enter in a column to encode')).strip()
    new_col = df[col_name].dropna().astype('category').cat.codes
#    import pdb
#    pdb.set_trace()
    return pd.concat([df,new_col])
    
    
# ============= Feature selection/engineering =================== #

feature_importance = lambda X,y: sk.ensemble.ExtraTreesClassifier(n_estimators=80).fit(X,y)

feature_importance_reg = lambda X,y: sk.ensemble.ExtraTreesRegressor(n_estimators=80).fit(X,y)

pca_transform = lambda X,n: sk.decomposition.PCA(n_components=n).fit_transform(X) #PCA transform data using n components

sparse_pca_transform = lambda X,n,alpha: sk.decomposition.SparsePCA(n_components=n,alpha=alpha).fit_transform(X)

tsne_transform = lambda X,dim: sk.manifold.TSNE(n_components=dim).fit_transform(X) #TSNE transform data, dim=2 or 3

isomap_transform = lambda X,dim: sk.manifold.Isomap(n_components=dim).fit_transform(X) #Isomap transform data,dim=2 or 3

lle_transform = lambda X,dim: sk.manifold.LocallyLinearEmbedding(n_components=dim).fit_transform(X) #LLE transform data,dim=2 or 3

make_k_means_cluster = lambda X,clusters: sk.cluster.KMeans(n_clusters=clusters).fit(X) #fit k-means cluster

k_means_transform = lambda X,clusters: sk.cluster.KMeans(n_clusters=clusters).fit_transform(X) #k_means transform on data X

select_k_best_clf = lambda X,y,K: sk.feature_selection.SelectKBest(k=K).fit_transform(X,y) #select K best and apply transformation

select_k_best_reg = lambda X,y,K: sk.feature_selection.SelectKBest(score_fun='f_regression',k=K).fit_transform(X,y) #select K best and apply transformation

standard_scale = lambda X: sk.preprocessing.StandardScaler().fit_transform(X) #standard scaler

null_transform = lambda X: sk.preprocessing.FunctionTransformer().fit_transform(X) #do-nothing transform, here for API consistency

normalize_l2 = lambda X: sk.preprocessing.Normalizer(norm='l2').fit_transform(X) #L2 normalization of data

normalize_l1 = lambda X: sk.preprocessing.Normalizer(norm='l1').fit_transform(X) #L1 normalization of data

normalize_max = lambda X: sk.preprocessing.Normalizer(norm='max').fit_transform(X) #L_infinity normalization of data

encode_labels = lambda y: sk.preprocessing.LabelEncoder().fit_transform(y) #encoding labels numerically

decode_labels = lambda labels: labels.inverse_transform(labels) #decoding original numeric labels

min_max_scale = lambda X,a,b: sk.preprocessing.MinMaxScaler(feature_range=(a,b)).fit_transform(X) #mix max scaling


def pca_reconstruction(X,n_comps):
    '''
    X: DataFrame or numpy array of shape num_samples x num_features (might also work with scipy sparse matrices - need to check)
    n_comps: int: number of principle components, note that this is between 2 and num_features
    reconstructs X using a given number of PCs
    '''
    pca = sk.decomposition.PCA(n_components=n_comps)
    pca.fit(X)
    X_rec = pca.inverse_transform(pca.transform(X))
   
    return X_rec

    
def train_test_split(X,y,train_per=0.8): 
    '''
    X: np array or pandas dataframe
    y: np.array or pandas dataframe
    this will work for both classification and regression
    '''
    X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X,y,test_size=1-train_per)
    return X_train,X_test,y_train,y_test

def plot_feature_importances(X,y,orient='horizontal',mode='clf',feature_labels=None):
        '''
        X: DataFrame or numpy array of shape (num_samples x num_features)
        y: Target to predict
        orient: string (horizontal or vertical)
        mode: string - classification (clf) or regression (reg)
        feature_labels - string to pass in the names of the features for plot labeling
        
        ranks the importance of each feature in X in predicting the outcome y
        '''
        if mode == 'clf':
            rank=feature_importance(X,y).feature_importances_
        if mode == 'reg':
            rank = feature_importance_reg(X,y).feature_importances_
        if feature_labels == None:
           feature_names=X.keys()
        if feature_labels != None:
            feature_names = feature_labels
        series=pd.Series(data=rank,index=feature_names)
        if len(series)<=10:
            choices = make_color_list(colors,len(series))
            plt.title('Feature Importances')
            plt.bar(x=np.arange(1,len(series)+1,1),height=rank,color=choices,tick_label=feature_names)
            plt.xticks(np.arange(1,len(series)+1,1))
            plt.show()
            
        if len(series)>10:
            choices = make_color_list(colors,10)
            feature_ranks=series.sort_values()
            top10=feature_ranks[0:10]
            if orient == 'horizontal':
                top10.plot(kind='bar',title='top features',color=choices)
            if orient == 'vertical':
                top10.plot(kind='bar',title='top 10 features',color=choices)


def manifold_plot(X,dim=2,mode='tsne',color='magenta',plot_trustworthiness='on'): #for unsupervised tasks or regression
    '''
    X: numpy array of shape num_samples x num_features (untested for DataFrames)
    dim: int (2 or 3) 
    mode: string - TSNE, LLE, Isomap, or PCA 
    plot_trustworthiness: trustworthiness is a metric that quantifies the quality of the learned embedding 
    
    here we represent our data X as a 2 or 3 dimensional manifold using one of the four techniques in mode and we then
    plot the result
    '''
    if mode == 'tsne':
        if dim == 2:
            tsne_embedding = tsne_transform(X,2)
            tsne_score = 1*np.around(tsne_t_score(X,2),decimals=3)
            if plot_trustworthiness == 'on':
                plt.title('TSNE Plot\n Trustworthiness: {}'.format(tsne_score))
            if plot_trustworthiness == 'off':
                plt.title('TSNE Plot')
            plt.scatter(tsne_embedding[:,0],tsne_embedding[:,1],color=color)
            plt.show()
        if dim == 3:
            tsne_embedding = tsne_transform(X,3)
            tsne_score = 1*np.around(tsne_t_score(X,2),decimals=3)
            
            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot(111,projection='3d')
            ax.scatter(tsne_embedding[:,0],tsne_embedding[:,1],tsne_embedding[:,2],color=color)
            if plot_trustworthiness == 'on':
                plt.title('TSNE Plot\n Trustworthiness: {}'.format(tsne_score))
            if plot_trustworthiness == 'off':
                plt.title('TSNE Plot')
            plt.show()
    
    if mode == 'lle':
        if dim == 2:
            lle_embedding = lle_transform(X,2)
            lle_score = 1*np.around(lle_t_score(X,2),decimals=3)
            if plot_trustworthiness == 'on':
                plt.title('LLE Plot\n Trustworthiness: {}'.format(lle_score))
            if plot_trustworthiness == 'off':
                plt.title('LLE Plot')
            plt.scatter(lle_embedding[:,0],lle_embedding[:,1],color=color)
            plt.show()
        if dim == 3:
            lle_embedding = lle_transform(X,3)
            lle_score = 1*np.around(lle_t_score(X,2),decimals=3)
            
            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot(111,projection='3d')
            ax.scatter(lle_embedding[:,0],lle_embedding[:,1],lle_embedding[:,2],color=color)
            if plot_trustworthiness == 'on':
                plt.title('LLE Plot\n Trustworthiness: {}'.format(lle_score))
            if plot_trustworthiness == 'off':
                plt.title('LLE Plot')
            plt.show()
            
    if mode == 'isomap':
        if dim == 2:
            isomap_embedding = isomap_transform(X,2)
            isomap_score = 1*np.around(isomap_t_score(X,2),decimals=3)
            if plot_trustworthiness == 'on':
                plt.title('Isomap Plot\n Trustworthiness: {}'.format(isomap_score))
            if plot_trustworthiness == 'off':
                plt.title('Isomap Plot')
            plt.scatter(isomap_embedding[:,0],isomap_embedding[:,1],color=color)
            plt.show()
            
        if dim == 3:
            isomap_embedding = isomap_transform(X,3)
            isomap_score = 1*np.around(isomap_t_score(X,2),decimals=3)
            
            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot(111,projection='3d')
            ax.scatter(isomap_embedding[:,0],isomap_embedding[:,1],isomap_embedding[:,2],color=color)
            if plot_trustworthiness == 'on':
                plt.title('Isomap Plot\n Trustworthiness: {}'.format(isomap_score))
            if plot_trustworthiness == 'off':
                plt.title('Isomap Plot')
            plt.show()
            
    if mode == 'pca':
        n = int(input('enter in the number of principal components:\t'))
        pca_embedding = pca_transform(X,n)[:,0:dim]
        pca_score = 1*np.around(t_score(X,pca_embedding),decimals=4)
        if dim == 2:
            if plot_trustworthiness == 'on':
                plt.title('PCA plot - {}/{} components\n Trustworthiness: {}'.format(n,X.shape[1],pca_score))
            if plot_trustworthiness == 'off':
                plt.title('PCA plot - {}/{} components'.format(n,X.shape[1]))
            plt.scatter(pca_embedding[:,0],pca_embedding[:,1],color=color)
            plt.show()
            
        if dim == 3:
            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot(111,projection='3d')
            ax.scatter(pca_embedding[:,0],pca_embedding[:,1],pca_embedding[:,2],color=color)
            if plot_trustworthiness == 'on':
                plt.title('PCA plot - {}/{} components\n Trustworthiness: {}'.format(n,X.shape[1],pca_score))
            if plot_trustworthiness == 'off':
                plt.title('PCA plot - {}/{} components'.format(n,X.shape[1]))
            plt.show()
#================================= MODELS ==============================================
#note arguments into the lambda functions are hyperparameters for the model. This API is convenient

#Classification =============================

#-----SVMs--------------------------
       
make_svc_linear = lambda C: sk.svm.SVC(C=C,kernel='linear',probability=True) #linear SVM for classification
make_svc_rbf = lambda C,gamma: sk.svm.SVC(C=C,kernel='rbf',gamma=gamma,probability=True) #RBF SVM for classification
make_svc_poly = lambda C,d,gamma: sk.svm.SVC(C=C,kernel='poly',degree=d,gamma=gamma,probability=True) #Polynomial SVM for classification
make_svc_sig = lambda C: sk.svm.SVC(C=C,kernel='sigmoid',probability=True) #Sigmoid SVM for classification

#---------Tree-based models----------------------------
make_decision_tree_clf = lambda depth: sk.tree.DecisionTreeClassifier(max_depth=depth) #Decision Tree for classification
make_random_forest_clf = lambda depth,trees: sk.ensemble.RandomForestClassifier(n_estimators=trees,max_depth=depth) #Random forest for classification


def make_gradient_booster_clf(lr,estimators,depth):
    model = sk.ensemble.GradientBoostingClassifier(learning_rate=lr,n_estimators=estimators,max_depth=depth) #Gradient Booster for classification
    return model

#-----Naive-Bayes models-----------------------------
    
make_gaussian_NB = lambda smoothing=1e-9: sk.naive_bayes.GaussianNB(var_smoothing=smoothing) #Gaussian NB model
make_multinomial_NB = lambda smoothing=1.0: sk.naive_bayes.MultinomialNB(alpha=smoothing) #Multinomial NB model
make_complement_NB = lambda smoothing=1.0: sk.naive_bayes.ComplementNB(alpha=smoothing) #Complement NB model
make_bernoulli_NB = lambda smoothing=1.0: sk.naive_bayes.BernoulliNB(alpha=smoothing) #Bernoulli NB model
make_categorical_NB = lambda smoothing=1.0: sk.naive_bayes.CategoricalNB(alpha=smoothing) #Categorical NB model 

#------Logistic Regression ----------
make_logistic_regression = lambda C: sk.linear_model.LogisticRegression(C=C) #logistic regression can be used for classification

#Regression ===========================
#-------SVMs--------------------------------------------
make_svr_linear = lambda C: sk.svm.SVR(C=C,kernel='linear') #SVM with linear kernel for regression
make_svr_rbf = lambda C,gamma: sk.svm.SVR(C=C,kernel='rbf',gamma=gamma) #SVM with RBF kernel foe regression 
make_svr_poly = lambda C,d,gamma: sk.svm.SVR(C=C,degree=d,gamma=gamma,kernel='poly') #SVM with polynomial kernel for regression 
make_svr_sig = lambda C: sk.svm.SVR(C=C,kernel='sigmoid') #SVM with sigmoid kernel for regressiion 

#--------Tree-based models--------------------------
make_decision_tree_reg = lambda depth: sk.tree.DecisionTreeRegressor(max_depth=depth) #Decision Tree for Regression
make_random_forest_reg = lambda depth,trees: sk.ensemble.RandomForestRegressor(n_estimators=trees,max_depth=depth) #Random Forest for Regression 

def make_gradient_booster_reg(lr,estimators,depth): #Gradient Booster for Regression 
    model = sk.ensemble.GradientBoostingRegressor(learning_rate=lr,n_estimators=estimators,max_depth=depth)
    return model

#------Neural Networks--------
'''these two will be deprecated in favor of the Pytorch ones,
but for the alpha, these will work ok
'''
make_vanilla_NN_clf = lambda layers,lr,mom,activation: sk.neural_network.MLPClassifier(hidden_layer_sizes=layers,
                                                                                   activation=activation,
                                                                                   learning_rate_init=lr,
                                                                                   momentum=mom)
make_vanilla_NN_reg = lambda layers,lr,mom,activation: sk.neural_network.MLPRegressor(hidden_layer_sizes=layers,
                                                                                   activation=activation,
                                                                                   learning_rate_init=lr,
                                                                                   momentum=mom)

                                                                    
#=============Model actions============
train = lambda model,X,y: model.fit(X,y) #trains a model (pass in a model object from above)
predict = lambda model,X: model.predict(X) #predicts new data (predicts y from novel data X)


def cross_validate(model,X,y,folds):
    scores=sk.model_selection.cross_val_score(estimator=model,X=X,y=y,cv=folds)
    for score in range(len(scores)):
        print('K = {} score: {:4f}'.format(score+1,scores[score]))
    return scores
def clf_accuracy(model,X,y,mode='train'):
    '''
    model: trained model object (can be untrained, but this is useless)
    X: DataFrame or numpy array of shape num_samples x num_features
    y: numpy array of predictor variable (shape either (num_samples,) or (num_samples x 1) - need to check , probably the first')
    mode: 'train' or 'validate'
    prints accuracy of a classifier
    '''
    correct = np.sum(predict(model,X) == y)
    total = len(y)
    acc = 100*np.around(correct/total,decimals=3)
    acc = np.format_float_positional(acc,precision=3)
    print('{} correct/total: {}/{}\n'.format(mode,correct,total))
    print('{} accuracy: {} %\n'.format(mode,acc))
    return model 

# ============== Metrics and metric Analysis for Classification & Regression ========================
    
predict_probability = lambda model,X: model.predict_proba(X) #predict probabilty for each class (sklearn softmax implementation?)

precision_score = lambda model,X,y: sk.metrics.precision_score(y_true=y,y_pred=model.predict(X),average=None) #Precision score

recall_score = lambda model,X,y: sk.metrics.recall_score(y_true=y,y_pred=model.predict(X),average=None) #Recall score

f1_score = lambda model,X,y: sk.metrics.f1_score(y_true=y,y_pred=model.predict(X),average=None) #F1 score

conf_mat = lambda model,X,y: sk.metrics.confusion_matrix(y_true=y,y_pred=model.predict(X)) #Confusion Matrix

sk_mse_loss = lambda y_true,y_pred: sk.metrics.mean_squared_error(y_true,y_pred) #MSE Loss

sk_l1_loss = lambda y_true,y_pred: sk.metrics.mean_absolute_error(y_true,y_pred) #L1 Loss

mae = lambda y_true,y_pred: np.mean(abs(y_pred,y_true)) #Mean absolute error

rmse_loss = lambda y_true,y_pred: np.sqrt(sk_mse_loss(y_true,y_pred)) #RMSE loss

log_loss = lambda y_true,y_pred: sk.metrics.log_loss(y_true,y_pred) #cross-entropy/log_loss for sklearn 

fbeta_score = lambda y_true,y_pred: sk.metrics.fbeta_score(y_true,y_pred) #F-beta score

hamming_loss = lambda y_true,y_pred: sk.metrics.hamming_loss(y_true,y_pred) #Hamming score (not yet tested)

def roc_auc_score(model,X,y):
    '''
    computes the ROC_AUC score for multi-class classification, note the model is NOT trained in this case
    '''
    mod = model.fit(X,y)
    probs = predict_probability(mod,X)
    return sk.metrics.roc_auc_score(y_true=y,y_score=probs,multi_class='ovr')


def roc_auc_plot(model,X,y):
    '''
    here we want to plot the ROC-AUC curve for multi-class classification problems,
    make the plots look slick. 
    '''
    pass

def mAP_plot(model,X,y):
    '''
    here we want to plot mean average precision for multi-class classification problems,
    make the plots look slick. 
    '''
    pass

def trained_roc_auc(model,X,y):
    '''
    computing roc_auc for an already trained model
    '''
    return sk.metrics.roc_auc_score(y_true=y,y_score=predict_probability(model,X),multi_class='ovr')

#======== Manifold Analysis  ============

t_score = lambda X,X_: sk.manifold.trustworthiness(X=X,X_embedded=X_)

tsne_t_score = lambda X,dim: t_score(X,tsne_transform(X,dim))

lle_t_score = lambda X,dim: t_score(X,lle_transform(X,dim))

isomap_t_score = lambda X,dim: t_score(X,isomap_transform(X,dim))

pca_t_score = lambda X,n: t_score(X,pca_transform(X,n))

pca_ev = lambda X,n: sk.decomposition.PCA(n_components=n).fit(X).explained_variance_ #PCA explained variance

pca_evr = lambda X,n: sk.decomposition.PCA(n_components=n).fit(X).explained_variance_ratio_ #PCA explained variance ratio


def plot_pca_evr(X,n,color='red'):
    '''
    X: Pandas DataFrame or numpy array of shape num_samples x num_features
    n: number of Principle components for PCA 
    function plots explained variance ratio as a function of the number of PCs on a dataset, helps aid in dimensionality reduction 
    '''
    plt.title('PCA Explained Variance Ratio')
    plt.plot(np.arange(1,len(pca_evr(X,n))+1),np.cumsum(pca_evr(X,n)),c=color)
    plt.xticks(ticks=np.arange(1,n+1,1))
    plt.xlabel('components')
    plt.show()
    

# ============== Anomaly detection & Adverse Impact Analysis ==================
def anomaly_detection(X,N=20):
    detector = sk.neighbors.LocalOutlierFactor(n_neighbors=N)
    samples = detector.fit_predict(X)
    anomalies = []
    for s in range(len(samples)):
        if samples[s] == -1:
            print('Sample {} - anomaly'.format(s+1))
            anomalies.append(s)
        if samples[s] == 1:
            print('Sample {}'.format(s+1))
    print('number of anomalies: {}'.format(len(anomalies)))
    return anomalies 

def adverse_impact_analysis(trained_model,X,y_true,labels):
    '''
    Plots a pie chart visualizing how skewed the model is towards predicting classes relative to other classes
    
    trained_model: object - some sklearn model that has been trained (i.e. called the "fit" method on X,y)
    X: 2D np.array - an array of (num_samples x num_features) - consistent with sklearn API 
    y_true: 1D np.array - an array of target labels (num_samples,) - consistent with sklearn API 
    labels: list of strings - names corresponding to the class labels (i.e. 0 - dog, 1 - cat ...)
    '''
    plt.title('Adverse Impact Analysis')
    y_pred = trained_model.predict(X)
    f1_score = sk.metrics.f1_score(y_true=y_true,y_pred=y_pred,average=None)
    plt.pie(f1_score,labels=labels)
    plt.show()


#============Pipleines / workflows==================
    
#===pipeline steps=================================
    
'''
These are the steps that can be fed into create_pipeline. Consult the code below (it has been commented out), and the 
how-to notebook for further details. For our own sanity, let's limit users to 3 unless it's easy to add N steps. 
Calling the function calls the appopriate object/method from sklearn and creates the step
'''
    
pca_step = lambda n: sk.decomposition.PCA(n_components=n) 

tsne_step = lambda dim: sk.manifold.TSNE(n_components=dim)

isomap_step = lambda dim: sk.manifold.Isomap(n_components=dim)

lle_step = lambda dim: sk.manifold.LocallyLinearEmbedding(n_components=dim)

k_means_step = lambda clusters: sk.cluster.KMeans(n_clusters=clusters)

select_k_best_clf_step = lambda K: sk.feature_selection.SelectKBest(k=K)

select_k_best_reg_step = lambda K: sk.feature_selection.SelectKBest(score_fun='f_regression',k=K)

standard_scale_step = sk.preprocessing.StandardScaler()

null_transform = sk.preprocessing.FunctionTransformer() #do-nothing transform

normalize_step = lambda n: [sk.preprocessing.Normalizer(norm='{}'.format(n))]

create_pipeline = lambda steps: sk.pipeline.make_pipeline(*steps)

'''
ArkiTEKT is the Neurales default model that is chosen. 
The goal is to have an automatic selection of a machine learning model that performs well on the given data.
One way to achieve this is to log the datasets users use, construct a "dataset of datasets" which includes

number of features, number of samples, class breakdown, model used, and validation performance
based on this data, we train a model to then predict the most appropriate model on novel datasets,

and then we fine-tune hyperparameters either with genetic algorithms or a better approach. 

Gradient Boosters tend to perform the best on these 2-D datasets (not computer vision or NLP), and as such, are
the default model for ArkiTEKT 1.0, eventually, we will replace this with something more sophisticated
'''

ArkiTEKT_reg = lambda num_learners=100: create_pipeline([standard_scale_step,make_gradient_booster_reg(lr=0.005, 
                                                                                        estimators=num_learners,
                                                                                        depth=9)])
ArkiTEKT_clf = lambda num_learners=100: create_pipeline([standard_scale_step,pca_step(3),
                                      make_gradient_booster_clf(lr=0.005,estimators=num_learners,depth=9)]) #default pipeline
    
class ArkiTEKT():
    def __init__(self,memory_dir):
        super(ArkiTEKT,self).__init__()
        self.memory_dir = memory_dir #path to store the memory of the class
        self.memory = [] #starts with blank memory 
    
    def make_memory(self,results):
        pass
        
    def add_memory(self,results):
        '''
        results: dict - a dictionary of results obtained by calling the above functions
        '''
        self.memory.append(results)
    def build_pipeline(self,X,y):
        pass
        

def genetic_gradient_booster(X,y,num_generations=10,num_chromosomes=6):
    trees = list(np.arange(25,150))
    lr = [1e-4,2e-4,5e-4,1e-3,2e-3,5e-3,1e-2]
    depth = list(np.arange(1,12))

#--------workflows-------------------
''' 
These workflows are certainly not exhaustive, but the are common ones that many will use over and over
In general, workflows will be higher (3rd and higher) order functions built from the lambda functions above
'''
def train_and_validate_clf(X,y,train_per=0.8,predictor=ArkiTEKT_clf(),
                           compute_metrics=False,
                           labels=None): #train and validate workflow
    '''
    X: data for classification task (np.array, pandas dataframe should work fine too)
    y: target variable for prediction (np.array, pandas dataframe should work fine too)
    train_per: float between 0.01 and 0.99 for training
    predictor = model or pipeline, default the ArkiTEKT for classification
    note, X needs to be defined before calling this function with the default predictor
    '''
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_per)
    #...training
    trained_model = train(predictor,X_train,y_train)
    train_acc = clf_accuracy(trained_model,X_train,y_train)
    #...validating
    val_acc = clf_accuracy(trained_model,X_test,y_test,mode='validation')
    if compute_metrics == True:
        precision_train = precision_score(trained_model,X_train,y_train)
        precision_val = precision_score(trained_model,X_test,y_test)
        recall_train = recall_score(trained_model,X_train,y_train)
        recall_val = recall_score(trained_model,X_test,y_test)
        f1_train = f1_score(trained_model,X_train,y_train)
        f1_val = f1_score(trained_model,X_test,y_test)
    if labels != None:
        print('TRAIN SCORES:\n')
        for cl in range(len(labels)):
            print('Class:\t {} Precision: {:4f} | Recall: {:4f} | F1: {:4f}'.format(labels[cl],precision_train[cl],
                  recall_train[cl],f1_train[cl]))
        print('\n')
        print('VAL SCORES:\n')
        for cl in range(len(labels)):
             print('Class:\t {} Precision: {:4f} | Recall: {:4f} | F1: {:4f}'.format(labels[cl],precision_val[cl],
                  recall_val[cl],f1_val[cl]))
        train_metrics = {'precision': precision_train, 'recall': recall_train, 'f1': f1_train}
        val_metrics = {'precision': precision_val, 'recall': recall_val, 'f1': f1_val}
        results = {'model': trained_model,'train':train_metrics,'test':val_metrics}
        return results
    return trained_model

def train_and_validate_clfs(X,y,predictors,predictor_names,train_per=0.8,
                            compute_metrics=False,
                            labels=None):
    '''
    trains and validates MULTIPLE classifiers and returns the results
    '''
    trained_models=[]
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_per)
    for model in range(len(predictors)):
        print('====={}=====\n'.format(predictor_names[model]))
        trained_models.append(train_and_validate_clf(X=X,y=y,
                                                     train_per=train_per,
                                                     predictor=predictors[model],
                                                     compute_metrics=compute_metrics,
                                                     labels=labels))
    return trained_models
    
                                
def train_and_validate_reg(X,y,train_per,predictor=ArkiTEKT_reg()):
    '''
    X: data for regression task (np.array, pandas dataframe should work fine too)
    y: target variable for prediction (np.array, pandas dataframe should work fine too)
    train_per: float between 0.01 and 0.99 for training
    predictor = model or pipeline, default the ArkiTEKT for regression
    note,X needs to be defined before calling this function with the default predictor 
    '''
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_per)
    trained_model = train(predictor,X_train,y_train)
    train_loss = sk_mse_loss(y_true=y_train,y_pred=predict(trained_model,X_train))
    val_loss = sk_mse_loss(y_true=y_test,y_pred=predict(trained_model,X_test))
    print('Train Loss: {}'.format(1*np.around(train_loss,decimals=3)))
    print('Validation Loss: {}'.format(1*np.around(val_loss,decimals=3)))
    return trained_model 
    
def fast_prf1_scores(y_true,y_pred,labels):
    '''
    outputs precision,recall, and F1 scores based on model predictions, true data, and labels provided
    '''
    precision = precision_score(y_true,y_pred)
    recall = recall_score(y_true,y_pred)
    f1 = f1_score(y_true,y_pred)
    return {'precision: {:4f}': precision, 'recall: {:4f}': recall, 'F1: {:4f}': f1}


def adverse_impact_analysis_clfs(X,y_true,labels,predictors,predictor_names):
    '''
    adverse impact analysis for MULTIPLE classifiers
    '''
    for model in range(len(predictors)):
        print('====={}=====\n'.format(predictor_names[model]))
        adverse_impact_analysis(trained_model=predictors[model],X=X,y_true=y_true,labels=labels)
        
def PRF1_metrics(X,y,labels,trained_model,train_per=0.8):
    '''
    function to get precision,recall, and F1 scores on a dataset using a classifier
    function arguments are the same as in for train_and_validate_clf
    decimals indicate how many decimals to round the metric scores to at the end (int only)
    '''
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_per)
    precision_train = precision_score(trained_model,X_train,y_train)
    recall_train = recall_score(trained_model,X_train,y_train)
    f1_train = f1_score(trained_model,X_train,y_train)
    
    precision_val = precision_score(trained_model,X_test,y_test)
    recall_val = recall_score(trained_model,X_test,y_test)
    f1_val = f1_score(trained_model,X_test,y_test)
    
    print('TRAIN SCORES:\n')
    for cl in range(len(labels)):
        print('Class:\t {} Precision: {:4f} | Recall: {:4f} | F1: {:4f}'.format(labels[cl],precision_train[cl],
              recall_train[cl],f1_train[cl]))
    print('\n')
    print('VAL SCORES:\n')
    for cl in range(len(labels)):
         print('Class:\t {} Precision: {:4f} | Recall: {:4f} | F1: {:4f}'.format(labels[cl],precision_val[cl],
              recall_val[cl],f1_val[cl]))
    train_metrics = {'precision': precision_train, 'recall': recall_train, 'f1': f1_train}
    val_metrics = {'precision': precision_val, 'recall': recall_val, 'f1': f1_val}
    return {'train metrics': train_metrics, 'val metrics': val_metrics}


def PRF1_metrics_clfs(X,y,labels,predictors,predictor_names,train_per=0.75):
    '''
    this will calculate precision,recall, and F1 scores for multiple predictors. Note that we need to train the models
    again if they were trained already, the fast_prf1_scores method 
    '''
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_per)
    for model in range(len(predictors)):
        print('====={}=====\n'.format(predictor_names[model]))
        print(PRF1_metrics(X,y,labels,trained_model=predictors[model],train_per=train_per))
        
    
def manifold_first_(X,y,train_per=0.8,predictor=make_svc_linear(C=1),mode='tsne',dim=2): #don't use yet
    '''
    this is going to be a more modular pipeline where the user can transform data via
    tsne,lle, or isomap before using the other pipeline steps, as the pipeline method is not as robust
    as desired
    '''
    if mode == 'tsne':
        model = train_and_validate_clf(tsne_transform(X,dim),predictor=predictor)
    pass
#we can combine them all together to create a workflow
    
def workflow(X,y,workflow_steps=['plot_pca_evr','ArkiTEKT_reg']):
    '''
    This is a very high level API which will allow the user to schedule and implement
    multiple full data science experiments in the sequence they choose and log results
    when possible. Ability to perform multiple data science experiments without the need
    to code. 
    
    X: 2D np.array of dim (num_samples x num_features) - dataset of interest
    y: 1D np.array of dim (num_samples,) - targets/labels of interest
    workflow_steps: list of strings - strings encoding the workflow steps, which call the appropriate functions. 
    
    steps: plot_pca_evr - computes pca on X and plots the explained variance ratio as a function of the number of PCA components
           ArkiTEKT_reg - performs regression using the ArkiTEKT model
           ArkiTEKT_clf - performs classification using the ArkiTEKT model 
           tsne_plot - performs TSNE on X and plots the results
           lle_plot - performs LLE on X and plots the results
           isomap_plot - performs Isomap on X and plots the results
           train_and_validate_clf - asks a user for a model, then calls train_and_validate_clf (see function for further documentation)
    '''
    for step in range(len(workflow_steps)):
        
        if workflow_steps[step] == 'plot_pca_evr':
            n = int(input('enter the number of principal components for PCA'))
            plot_pca_evr(X,n)
            
        if workflow_steps[step] == 'tsne_plot':
            dim = int(input('tsne dimension (2 or 3)'))
            manifold_plot(X,dim=dim,mode='tsne')
        
        if workflow_steps[step] == 'lle_plot':
            dim = int(input('lle dimension (2 or 3)'))
            manifold_plot(X,dim=dim,mode='lle')
        
        if workflow_steps[step] == 'isomap_plot':
            dim = int(input('isomap dimension (2 or 3)'))
            manifold_plot(X,dim=dim,mode='isomap')
        
        if workflow_steps[step] == 'ArkiTEKT_clf':
            train_per = float(input('Enter in training percentage: '))
            model = train_and_validate_clf(X,y,train_per=train_per)
            
        if workflow_steps[step] == 'ArkiTEKT_reg':
            train_per = float(input('Enter in a training percentage'))
            model = train_and_validate_reg(X,y,train_per=train_per)
        
        if workflow_steps[step] == 'svm_classifier':
            train_per = float(input('Enter in training percentage: '))
            model = train_and_validate_clf(X,y,train_per=train_per)
        
        
            

def csv_pca_reconstruction(filepath,n_components):
    data = load_csv(filepath)
    data = drop_multiple_cols(data)
    pass
    

#=========EXAMPLES===========
#1) load in data, get X,y (in this case, the iris dataset)
X, y = sk.datasets.load_iris(True)
labels= ['Iris-Setosa', 'Iris-Versicolor', 'Iris-Virginica']
'''
#users wants to train and validate their data
#user selects to standard scale data, then use pca with 2 components, and finally, 
#selects linear svm as the model with C=0.85
#users uses a train percentage of 65
example_1= train_and_validate_clf(X,y,train_per=0.65,
                                  predictor=create_pipeline([standard_scale_step,pca_step(2),
                                                                            make_svc_linear(C=0.85)]))
    
#example 2:
#users selects this time to feature engineer with 2-D tsne and then use a random forest with 20 trees with depth=7
#user selects a train/test split of 50%
#sklearn make_pipeline method is limited - the API in example 1 works only for models which have BOTH a
#fit AND transform method (tsne has fit and fit_transform but not transform). In these cases, use the API below
    
example_2 = train_and_validate_clf(X=tsne_transform(X,dim=3),y=y,train_per=0.5,
                                   predictor=make_random_forest_clf(depth=7,trees=20))
#here we want to calculate precision,recall and f1 scores on the data, this time using the following pipeline:
#standard scale and Gaussian naive bayes, 
example_3 = PRF1_metrics(X,y,labels=labels,predictor=create_pipeline([standard_scale_step,
                                                                      make_gaussian_NB()]))
    
workflow_example = workflow(X,y,workflow_steps=['plot_pca_evr','ArkiTEKT_clf'])
#when prompted for number of principal components, select 2 and the workflow will be executed
#note that just by entering the data and the target, the app behind the scenes trains a model 
#with no input from the user whatsoever

#regression example=====
X,y = sk.datasets.load_boston(True)
example_4 = train_and_validate_reg(X,y) #simply getting the train and validation loss scores for this dataset
'''
