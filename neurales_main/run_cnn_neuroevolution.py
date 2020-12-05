# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 00:05:39 2020

@author: XZ-WAVE
"""

import argparse
from neuroevolution import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='argparse for neuroevolution')
    parser.add_argument('--train_path',type=str,default='mnist',help='training path data')
    parser.add_argument('--val_path',type=str,default=None,help='validation data path')
    parser.add_argument('--generations',type=int,default=50,help='number of generations')
    parser.add_argument('--chromosomes',type=int,default=20,help='number of chromosomes/species')
    parser.add_argument('--elite_frac',type=float,default=0.2,help='fraction of species deemed "elite"')
    parser.add_argument('--evolution_mode',type=str,default='elitist',help='crossover or elitist')
    parser.add_argument('--num_mutations',type=int,default=2,
                        help='number of mutations when using crossover-mutation evolution method')
    parser.add_argument('--num_parents',type=int,default=4,
                        help='number of parents when using crossover mutation')
    parser.add_argument('--channels',type=int,default=3,help='number of channels for the the images')
    parser.add_argument('--img_size',type=int,default=40,help='tuple for image size (H X W)')
    parser.add_argument('--batch_size',type=int,default=100,help='batch size for dataset')
    parser.add_argument('--max_epochs',type=int,default=50,
                        help='maximum number of epochs to train each model')
    parser.add_argument('--target_score',type=float,default=0.95,help='desired target score for metric')
    parser.add_argument('--metric',type=str,default='F1',help='metric to choose from: F1,precision,recall,accuracy')
    parser.add_argument('--max_loss_thres',type=float,default=1e6,help='maximum loss allowed in model epoch before aborting')
    parser.add_argument('--explore',type=bool,default=False,help='debug/explore when script finishes')
    args = parser.parse_args()
    
    results=genetic_train_cnn(train_path=args.train_path,
                              val_path=args.val_path,
                              num_generations=args.generations,
                              num_chroms=args.chromosomes,
                              elite_frac=args.elite_frac,
                              transform='default',
                              img_size=(args.img_size,args.img_size),
                              batch_size=args.batch_size,
                              target_score=args.target_score,
                              metric=args.metric,
                              max_loss_thresh=args.max_loss_thres,
                              evolution_mode=args.evolution_mode,
                              num_parents=args.num_parents,
                              num_mutations=args.num_mutations,
                              max_epochs=args.max_epochs)
    dump_genetic_cnn(results,evolution_mode=args.evolution_mode)
       
    if args.explore == True:
        import pdb
        pdb.set_trace()
        