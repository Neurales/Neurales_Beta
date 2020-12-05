# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 14:03:46 2020

@author: XZ-WAVE
"""

import argparse
from neurales_obj_detect import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neurales Object Detection')
    
    parser.add_argument('--img_path',type=str,default='./test_img.jpg',help='img path')
    parser.add_argument('--model',type=str,default='fast_rcnn',help='fast_rcnn or mask_rcnn')
    parser.add_argument('--thres',type=float,default=0.9,
                        help='detect objects at or above given confidence threshold')
    parser.add_argument('--text_size',type=float,default=1.5,
                        help='fontsize to display text above bounding box(es)')
    parser.add_argument('--plot_conf',type=bool,default=False,
                        help='plot confidence threshold above bounding box(es)')
    parser.add_argument('--box_color',type=str,default='green',
                        help='red green or blue box color (custom colors in main file)')
    parser.add_argument('--text_color',type=str,default='red',help='text coolor: red green or blue')
    parser.add_argument('--explore',type=bool,default=False,help='explore/debug after script finishes')
    
    args = parser.parse_args()
    
    detect_obj(img_path=args.img_path,
               model=args.model,
               threshold=args.thres,
               rect_th=4,
               text_size=args.text_size,
               text_th=2,
               plot_obj_conf=args.plot_conf,
               decimals=3,
               box_color=args.box_color,
               text_color=args.text_color,
               custom_box_color=None,
               custom_text_color=None)
    if args.explore == True:
        import pdb
        pdb.set_trace()