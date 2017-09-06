# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 11:32:17 2017

@author: dena
"""
#python readCroppedImages.py /path/to/vggtr_confidence/*.csv

import scipy
import scipy.io
import sys
import numpy as np
from matplotlib import pyplot as plt
import cv2
from collections import defaultdict
from commands import getoutput as go
import glob, os
import re

if __name__=='__main__':
    idx = 0
    res = open('result.txt', "w")
    for DictNet_res in sys.argv[1:]:
        confDic  = open (DictNet_res).read()
        numQWords =  confDic.count('\n')
        idx +=1
        print idx
        confDic = confDic.split('\n')
        img_name = (DictNet_res.split('/')[-1]).split('.')[0]
        for nqw in range(0,1):
            #print nqw
            point = confDic[nqw].split(',')
            transcription = point[5]                                           
            res.write('%s.png, "%s"\r\n' % (img_name,transcription))
            res.flush()
                
    res.close()
        
    print idx
