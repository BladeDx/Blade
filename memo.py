# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 13:25:40 2021

@author: Owner
"""

def gen_memo(save_locate, epochs, batch_size, loss_name, img_data_locate, img_height, img_width, img_numY,img_numU,img_numC ,elapsed_time ,iou_score):
    #data_list = [[]for j in 20]
    with open(save_locate + "/memo.dat", "w",newline="") as f:
        f.write("CNNの学習条件""\n")
        f.write("epochs : " + str(epochs) +"\n")
        f.write("batch_size : " + str(batch_size) +"\n")
        f.write("loss_name : " + str(loss_name) +"\n")
        f.write("img_data_locate : " + img_data_locate + "\n")
        f.write("img_height : " + str(img_height) +"\n")
        f.write("img_width : " + str(img_width) + "\n")
        f.write("img_numMount : " + str(img_numY) + "\n")
        f.write("img_numSea : " + str(img_numU) + "\n")    
        f.write("img_numCloud : " + str(img_numC) + "\n") 
        f.write("iou_score : " + str(iou_score) + "\n")
        f.write("elapsed_time : " + str(elapsed_time) + "\n")
        
        
       