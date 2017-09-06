# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 17:18:49 2017

@author: dena
"""
# python confidenceOfCropped.py /path/to/images/*.jpg

import scipy
import scipy.io
import sys
import numpy as np
import cv2
import os

if __name__=='__main__':
	#croppedWord_dir = sys.argv[1:]
	os.makedirs('confidence')
	for croppedWord_img in sys.argv[1:]:
	    # Read each image
	    img=cv2.imread(croppedWord_img)   
		#read the size of each image     
	    cv_size = lambda img: tuple(img.shape[1::-1])
	    width, height = cv_size(img)
	#write a csv file per each image
	    img_name = (croppedWord_img.split('/')[-1]).split('.')[0]
	    conf = open('confidence/'+ img_name + '.csv', "w")
	#write the confidence and coordinate of each one same as the conf_prop
	    conf.write('0.00000,0.00000,%f,%f,0.99988,0.99988'%(width,height))
	    conf.close()