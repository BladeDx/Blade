# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 13:31:19 2021

@author: takase
"""

import numpy as np
import sys,cv2


def iou_score(output,target):

    #print(output)
    intersection = 0
    union = 0
    
    
    
    
    
    height,width = output.shape[:2]
    
    for i in range(height):
        for j in range(width):
            
            if (output[i][j] == 255 and target[i][j] == 255):
                intersection = intersection + 1
                    
    
    for i in range(height):
        for j in range(width):
            
            if (output[i][j] == 255 or target[i][j] == 255):
                union = union + 1
                

    
    #print(intersection)
    #print(union)
    
    return intersection / union
        

    