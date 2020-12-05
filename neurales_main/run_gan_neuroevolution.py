# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 13:41:20 2020

@author: XZ-WAVE
"""

import argparse
from neuroevolution import *

if __name__ == '__main__':
    argparse = argparse.ArgumentParser(description='GAN neuroevolution script')
    argparse.add_argument('--path',type=str,default='mnist',help='path to train data')
    argparse.add_argument('--generations',type=int,default=10,help='number of generations')
    argparse.add_argument('--chromosomes',type=int,default=10,help='number of chromosomes/species')
    argparse.add_argument('--batch_size',type=int,default=100,help='path to train data')
    argparse.add_argument('--img_size',type=int,default=40,help='number of generations')
    argparse.add_argument('--cmap',type=str,default=None,help='matplotlib color map')
    argparse.add_argument('--normalize',type=bool,default=True,help='normalizes images for plots')
    argparse.add_argument('--noise_dim',type=int,default=3,help='noise dimension/channels to feed into generator')
    argparse.add_argument('--loss',type=str,default='bce',help='loss functions: bce, mse, mae, kl_div loss')
    argparse.add_argument('--elite_frac',type=float,default=0.2,help='fraction of species deemed "elite"')
    argparse.add_argument('--loss_thres',type=float,default=50,help='max loss threshold for GAN before aborting')
    argparse.add_argument('--max_epochs',type=float,default=100,help='maximum number of epochs to evovle GAN')
    args = argparse.parse_args()
    
    
    
    results=evolve_dcgan(path=args.path,
                 num_generations=args.generations,
                 num_chroms=args.chromosomes,
                 transform='default',
                 batch_size=args.batch_size,
                 img_size=(args.img_size,args.img_size),
                 cmap=args.cmap,
                 normalize_plot=args.normalize,
                 noise_dim=args.noise_dim,
                 loss_function=args.loss,
                 elite_frac=args.elite_frac,
                 loss_thres=args.loss_thres,
                 plot_samples=False,
                 max_epochs=args.max_epochs)
    
    dump_gan_results(results)