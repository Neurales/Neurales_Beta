# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 10:58:37 2020

@author: XZ-WAVE
"""

import argparse
from Neurales_2020_main import *
from neurales_vision_utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='running custom models')
    parser.add_argument('--path',type=str,help='enter in a the full path for the dataset')
    parser.add_argument('--mode',type=str,default='clf',help='clf (classification) reg (regression)')
    parser.add_argument('--train_per',type=float,default=0.75,help='training percentage')
    parser.add_argument('--cross_val',type=bool,default=False,
                        help='whether or not to perform cross validation on a dataset')
    parser.add_argument('--class_hist',type=bool,default=True,help='plot class histogram or not')
    parser.add_argument('--model',type=str,default='RF',
    help='RF (Random Forest) NB (Naive Bayes) SVM (Support Vector Machine) GB (Gradient Booster) DT (Decision Tree)')
    parser.add_argument('--feature_scale',type=bool,default=True,help='use feature scaling or not')
    parser.add_argument('--feature_analysis',type=bool,default=False,help='do feature analysis')
    parser.add_argument('--manifold_plot',type=bool,default=False,
                        help='plots manifold breakdown of data')
    parser.add_argument('--plot_evr',type=bool,default=False,
                        help='plots explained variance ratio as a function of PCA components')
    parser.add_argument('--anomaly_detection',type=bool,default=False)
    parser.add_argument('--metrics',type=bool,default=True,help='compute precision, recall and F1?')
    parser.add_argument('--adverse_impact',type=bool,default=False,
                        help='conduct Adverse Impact Analysis on classification problems')
    parser.add_argument('--explore',type=bool,default=True,
                        help='debug/explore variables and models further')
    args = parser.parse_args()
    
    
    if args.mode ==  'clf':
        X,y, feature_names, labels = data_target_parse(filepath=args.path,mode=args.mode)
        labels = list(labels)
    if args.mode == 'reg':
        X,y, features = data_target_parse(filepath=args.path,mode=args.mode)
    if args.class_hist == True:
        if args.mode == 'clf':
            class_breakdown_(y)
        if args.mode == 'reg':
            pass
    
    if args.feature_scale == True:
        typ = str(input('use min/max scaling (M) or standard scaler (S)?')).strip()
        if typ == 'M':
            a = float(input('enter lowest value for scaled features: '))
            b = float(input('enter highest value for scaled features: '))
            X = min_max_scale(X=X,a=a,b=b)
        if typ == 'S':
            X = standard_scale(X)
            
    if args.feature_analysis == True:
        if args.mode == 'clf':
            plot_feature_importances(X,y,mode=args.mode,feature_labels=list(feature_names))
        if args.mode == 'reg':
            plot_feature_importances(X,y,mode=args.mode)
        
    if args.manifold_plot == True:
        manifold_type = str(input('enter in one of the following manifold types: tsne | lle | isomap | pca: ')).strip()
        dim = int(input('manifold dimension (2 or 3): '))
        if X.shape[0] >= 250:
            print('Manifold plotting is usually a compute intensive process')
            print('The first 250 of {} samples will be plotted'.format(X.shape[0]))
            go = input('Do you wish to plot the entire dataset instead: Y/N?').strip()
            if go == 'Y' or 'y':
                if args.mode == 'reg':
                    manifold_plot(X=X,
                                  dim=dim,
                                  mode=manifold_type,
                                  color='magenta',
                                  plot_trustworthiness='on')
                if args.mode == 'clf':
                    if manifold_type == 'tsne':
                        name = 'TSNE'
                        X_embedded = tsne_transform(X,dim=dim)
                    if manifold_type == 'lle':
                        name = 'LLE'
                        X_embedded == lle_transform(X,dim=dim)
                    if manifold_type == 'isomap':
                        name = 'Isomap'
                        X_embedded = isomap_transform(X,dim=dim)
                    if manifold_type == 'pca':
                        name = 'PCA'
                        n = int(input('enter in the number of components for PCA: '))
                        X_embedded = pca_transform(X,n=n)[:,0:dim]
                    plot_class_scatter(X=X,
                                       X_embedded=X_embedded,
                                       y=y,
                                       labels=labels,
                                       colors=colors,
                                       title='{} Class Clusters'.format(name),
                                       plot_trustworthiness='on')
                        
            if go != 'Y' or 'y':
                if args.mode == 'reg':
                    manifold_plot(X=X[:250],
                                  dim=dim,
                                  mode=manifold_type,
                                  color='magenta',
                                  plot_trustworthiness='on')
                if args.mode == 'clf':
                    if manifold_type == 'tsne':
                        name = 'TSNE'
                        X_embedded = tsne_transform(X[:250],dim=dim)
                    if manifold_type == 'lle':
                        name = 'LLE'
                        X_embedded == lle_transform(X[:250],dim=dim)
                    if manifold_type == 'isomap':
                        name = 'Isomap'
                        X_embedded = isomap_transform(X[:250],dim=dim)
                    if manifold_type == 'pca':
                        n = int(input('enter in the number of components for PCA: '))
                        X_embedded = pca_transform(X[:250],n=n)[:,0:dim]
                    plot_class_scatter(X=X[:250],
                                       X_embedded=X_embedded,
                                       y=y[:250],
                                       labels=labels,
                                       colors=colors,
                                       title='{} Class Clusters'.format(name),
                                       plot_trustworthiness='on')
               
        if X.shape[0] < 750:
            if args.mode == 'reg':
                
                manifold_plot(X=X,
                              dim=dim,
                              mode=manifold_type,
                              color='magenta')
            if args.mode == 'clf':
                if manifold_type == 'tsne':
                    name = 'TSNE'
                    X_embedded = tsne_transform(X,dim=dim)
                if manifold_type == 'lle':
                    name = 'LLE'
                    X_embedded = lle_transform(X,dim=dim)
                if manifold_type == 'isomap':
                    name = 'Isomap'
                    X_embedded = isomap_transform(X,dim=dim)
                if manifold_type == 'pca':
                    n = int(input('enter in the number of components for PCA: '))
                    X_embedded = pca_transform(X,n=n)[:,0:dim]
                plot_class_scatter(X=X,
                                   X_embedded=X_embedded,
                                   y=y,
                                   labels=labels,
                                   colors=colors,
                                   title='{} Class Clusters'.format(name),
                                   plot_trustworthiness='off')
            
                    
                
    if args.plot_evr == True:
        n = int(input('enter the number of principle components: '))
        plot_pca_evr(X,n=n,color='red')
    
    if args.model == 'DT':
        depth = int(input('enter in the max depth of the tree (recommended between 3 and 12)'))
        if args.mode == 'clf':
            model = make_decision_tree_clf(depth=depth)
        if args.mode == 'reg':
            model = make_decision_tree_reg(depth=depth)
            
    if args.model == 'RF':
        depth = int(input('enter in the max depth of the trees: '))
        num_trees = int(input('enter in the number of trees: '))
        
        if args.mode == 'clf':
            model = make_random_forest_clf(depth=depth,trees=num_trees)
        if args.mode == 'reg':
            model = make_random_forest_reg(depth=depth,trees=num_trees)
            
    if args.model == 'GB':
        depth = int(input('enter in the max depth of the trees (recommend between 3 and 12): '))
        num_trees = int(input('enter in the number of trees (recommend between 20 and 200): '))
        lr = float(input('enter in learning rate (recommended 0.001 or 0.01): '))
        if args.mode == 'clf':
            model = make_gradient_booster_clf(lr=lr,estimators=num_trees,depth=depth)
        if args.mode == 'reg':
            model = make_gradient_booster_reg(lr=lr,estimators=num_trees,depth=depth)
            
    if args.model == 'NB':
        print('enter in a Naive Bayes type: ')
        print('G (Gaussian) M (Multinomial) C (Complement) B (Bernoulli) Ca (Categorical)')
        nb_type = str(input('enter in a letter from the above choices')).strip()
        if args.mode == 'clf':
            if nb_type == 'G':
                model = make_gaussian_NB()
            if nb_type == 'M':
                model = make_multinomial_NB()
            if nb_type == 'C':
                model = make_complement_NB()
            if nb_type == 'B':
                model = make_bernoulli_NB()
            if nb_type == 'Ca':
                model = make_categorical_NB()
                
        if args.mode == 'reg':
            print('Neurales does not yet support NB-based regression!')
            import time
            time.sleep(3)
            exit 
    if args.model == 'SVM':
        if args.mode == 'clf':
            kernel = str(input('enter in a kernel: L (linear) | R (rbf) | P (poly) | S (sigmoid)')).strip()
            if kernel == 'L':
                C = float(input('enter in a C value (recommended between 0.1 and 1)'))
                model = make_svc_linear(C=C)
            if kernel == 'R':
                C = float(input('enter in a C value (recommended between 0.1 and 1)'))
                gamma = float(input('enter in a gamma value (recommended between 0.01 and 0.1)'))
                model = make_svc_rbf(C=C,gamma=gamma)
            if kernel == 'P':
                C = float(input('enter in a C value (recommended between 0.1 and 1)'))
                gamma = float(input('enter in a gamma value (recommended between 0.01 and 0.1)'))
                d = int(input('enter in a degree for polynomial kernel (recommended between 2 and 5)'))
                model = make_svc_poly(C=C,d=d,gamma=gamma)
            if kernel == 'S':
                C = float(input('enter in a C value (recommended between 0.1 and 1)'))
                model = make_svc_sig(C=C)
        if args.mode == 'reg':
            kernel = str(input('enter in a kernel: L (linear) | R (rbf) | P (poly) | S (sigmoid)')).strip()
            if kernel == 'L':
                C = float(input('enter in a C value (recommended between 0.1 and 1)'))
                model = make_svr_linear(C=C)
            if kernel == 'R':
                C = float(input('enter in a C value (recommended between 0.1 and 1'))
                gamma = float(input('enter in a gamma value (recommended between 0.01 and 0.1)'))
                model = make_svr_rbf(C=C,gamma=gamma)
            if kernel == 'P':
                C = float(input('enter in a C value (recommended between 0.1 and 1'))
                gamma = float(input('enter in a gamma value (recommended between 0.01 and 0.1)'))
                d = int(input('enter in a degree for polynomial kernel (recommended between 2 and 5)'))
                model = make_svr_poly(C=C,d=d,gamma=gamma)
            if kernel == 'S':
                C = float(input('enter in a C value (recommended between 0.1 and 1)'))
                model = make_svr_sig(C=C)

    if args.mode == 'clf':
        if args.metrics == True:
            trained_model = train_and_validate_clf(predictor=model,
                               X=X,
                               y=y,
                               train_per=args.train_per,
                               compute_metrics=args.metrics,
                               labels=labels)
            if args.cross_val == True:
                 folds = int(input('Enter a value for K (number of folds): '))
                 cross_validate(model=trained_model,
                                X=X,
                                y=y,
                                folds=folds)
        if args.metrics == False:
            trained_model = train_and_validate_clf(predictor=model,
                               X=X,
                               y=y,
                               train_per=args.train_per)
            if args.cross_val == True:
                folds = int(input('Enter a value for K (number of folds): '))
                cross_validate(model=trained_model,
                                X=X,
                                y=y,
                                folds=folds)
    if args.mode == 'reg':
        trained_model = train_and_validate_reg(predictor=model,
                               X=X,
                               y=y,
                               train_per=args.train_per) 
   
        
    if args.adverse_impact == True:
        adverse_impact_analysis(trained_model=trained_model['model'],
                                X=X,
                                y_true=y,
                                labels=labels)
    if args.anomaly_detection == True:
        anomalies = anomaly_detection(X)
        
    if args.explore == True:
        import pdb
        pdb.set_trace()
            

            
            
                
    
    
    