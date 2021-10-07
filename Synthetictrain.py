# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 11:56:18 2021

@author: takase
"""

from matplotlib import pylab as plt
import numpy as np
import cv2


def synthetic(save_locate, org_img, pre_img):
    
    
    for p in range(len(pre_img)): 
     rgb_pre_img = np.zeros((512,512,3))
     #print(org_img.shape)
     #print(pre_img.shape)
     #print(rgb_pre_img.shape)
     for i in range(512):
        for j in range(512):
            
            """if pre_img[p][i][j] == 255:
                for k in range(2):
                 rgb_pre_img[i][j][k] += 50
            else
                for k in range(3):
                 rgb_pre_img[i][j][k] = org_img[p][i][j][k]"""
            
            #ブレード色変え
            """for k in range(3):
                rgb_pre_img[i][j][k] = org_img[p][i][j][k]         
                if pre_img[p][i][j] == 255:
                    #for k in range(2):
                        rgb_pre_img[i][j][k+1] += 40"""
            
            
            #背景色変え
            for k in range(3):
                rgb_pre_img[i][j][k] = org_img[p][i][j][k] 
            if pre_img[p][i][j] == 0:
                rgb_pre_img[i][j][0] = 0
                rgb_pre_img[i][j][1] = 0
                rgb_pre_img[i][j][2] = (org_img[p][i][j][0]*0.299) + (org_img[p][i][j][1]*0.587) + (org_img[p][i][j][2]*0.114)        
                
                #blend = cv2.addWeighted(org_img[p],0.8,rgb_pre_img[p],0.2)
     cv2.imwrite(save_locate + "/synthetic/" + str(p) + ".bmp", rgb_pre_img)
    
    #そのままウインドウに表示
    """rgb_pre_img = cv2.imread(save_locate + "/synthetic/" + str(p) + ".bmp") 
    cv2.imshow("color",rgb_pre_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""
    #return rgb_pre_img
    #print("rgb_pre_img.shape = " + str(rgb_pre_img.shape))

    """for i in range(5):
        #blend.append(cv2.addWeighted(org_img[i],0.8,rgb_pre_img[i],0.2))
        cv2.imwrite(save_locate + "/synthetic/" + str(0) + ".bmp", rgb_pre_img)"""

    
