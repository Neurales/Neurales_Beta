# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 16:23:24 2020

@author: XZ-WAVE
"""

import argparse
from genetic_models import *
from Neurales_2020_main import *
from neurales_vision_utils import *

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='genetic models')
    parser.add_argument('--path',type=str,help='path to data')
    parser.add_argument('--train_per',type=float,default=0.75,help='train percentage')
    parser.add_argument('--models',type=str,default='GB',
                          help='GB (Gradient Boosters) or SVM (Support Vector Machines)'),
    parser.add_argument('--mode',type=str,default='clf',help='clf (classification) reg (regression)')
    parser.add_argument('--generations',type=int,default=20,help='number of generations')
    parser.add_argument('--chromosomes',type=int,default=10,help='number of chromosomes/species')
    parser.add_argument('--elite_frac',type=float,default=0.2,help='fraction of specis considered "elite"')
    parser.add_argument('--target_score',type=float,default=0.99,
                          help='target score for metric (between 0 and 1)')
    parser.add_argument('--metric',type=str,default='f1',
                          help='metric choices to prioritize: f1, precision, recall, accuracy')
  
    parser.add_argument('--explore',type=bool,default=True,
                        help='explore/debug after evolution finishes')
    
    args = parser.parse_args()
    
    if args.mode ==  'clf':
        X,y, feature_names, labels = data_target_parse(filepath=args.path,mode=args.mode)
        labels = list(labels)
    if args.mode == 'reg':
        X,y, features = data_target_parse(filepath=args.path,mode=args.mode)
    
    if args.mode == 'clf':
        
        if args.models == 'SVM':
            models = evolve_svm_pipeline_clf(X=X,
                                         y=y,
                                         num_generations=args.generations,
                                         num_chroms=args.chromsomes,
                                         metric=args.metric,
                                         elite_frac=args.elite_frac,
                                         target_acc=args.target_score,
                                         train_per=args.train_per)
        if args.models == 'GB':
            models = evolve_gradient_booster_pipeline_clf(X=X,
                                         y=y,
                                         num_generations=args.generations,
                                         num_chroms=args.chromosomes,
                                         metric=args.metric,
                                         elite_frac=args.elite_frac,
                                         target_acc=args.target_score,
                                         train_per=args.train_per)
    if args.explore == True:
        import pdb
        pdb.set_trace()
    
        
    