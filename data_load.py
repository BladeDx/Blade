# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 20:40:17 2021

@author: Owner
"""

import numpy as np
from PIL import Image
import cv2,gen_folder


def data_load(im_data_locate, img_numY,img_numU,img_numC):
    
    #変数宣言
    org_img = []
    msk_img = []
    moji = ['mount','sea','cloud']
    height = 512
    width = 512
    center = (int(width/2), int(height/2))
    #白黒反転
    tmp_img = []
    tmp_img2 = []
    #画像サイズ変える
    Yorg_size = []
    Uorg_size = []
    Corg_size = []
    Ymsk_size = []
    Umsk_size = []
    Cmsk_size = []
    Ymsk_size2 = []
    Umsk_size2 = []
    Cmsk_size2 = []


    
    #i.bmpの原画像とマスク画像読み込み
    #山
    k = 0
    for i in range(img_numY):
     for j in range(0,360,180):#j°回転
      Yorg_size = cv2.imread (im_data_locate + '/' + moji[k] + '/org' + '/' + str(i) + '.bmp')
      Yorg_size = cv2.resize(Yorg_size,dsize=(512,512))
      trans = cv2.getRotationMatrix2D(center, j , 1)
      org_img.append(cv2.warpAffine(Yorg_size, trans, (width,height)))
      Ymsk_size = cv2.imread (im_data_locate + "/" + moji[k] + "/msk" + "/" + str(i) + ".bmp",0)
      Ymsk_size = cv2.resize(Ymsk_size,dsize=(512,512))
      Ymsk_size2 = 255 - Ymsk_size   #白黒反転
      msk_img.append(cv2.warpAffine(Ymsk_size2, trans, (width,height)))
      print(moji[k] + str(i) + ".bmp " + str(j) + "°回転 読み込み")
     
    #海
    k = 1 
    for i in range(img_numU):
     for j in range(0,360,180):#j°回転
      """Uorg_size = cv2.imread (im_data_locate + "/" + moji[k] + "/org" + "/" + str(i) + ".bmp")
      org_img.append( np.array(cv2.resize(Uorg_size,dsize=(512,512))))
      Umsk_size = cv2.imread (im_data_locate + "/" + moji[k] + "/msk" + "/" + str(i) + ".bmp",0)
      Umsk_size = cv2.resize(Umsk_size,dsize=(512,512))
      Umsk_size2 = 255 - Umsk_size   #白黒反転
      msk_img.append(Umsk_size2)
      print(moji[k] + str(i) + ".bmp " + str(j) + "°回転 読み込み")"""
      Uorg_size = cv2.imread (im_data_locate + '/' + moji[k] + '/org' + '/' + str(i) + '.bmp')
      Uorg_size = cv2.resize(Uorg_size,dsize=(512,512))
      trans = cv2.getRotationMatrix2D(center, j , 1)
      org_img.append(cv2.warpAffine(Uorg_size, trans, (width,height)))
      Umsk_size = cv2.imread (im_data_locate + "/" + moji[k] + "/msk" + "/" + str(i) + ".bmp",0)
      Umsk_size = cv2.resize(Umsk_size,dsize=(512,512))
      Umsk_size2 = 255 - Umsk_size   #白黒反転
      msk_img.append(cv2.warpAffine(Umsk_size2, trans, (width,height)))
      print(moji[k] + str(i) + ".bmp " + str(j) + "°回転 読み込み")
      
    #曇り
    k = 2
    for i in range(img_numC):
     for j in range(0,360,180):#j°回転
      """Corg_size = cv2.imread (im_data_locate + "/" + moji[k] + "/org" + "/" + str(i) + ".bmp")
      org_img.append( np.array(cv2.resize(Corg_size,dsize=(512,512))))
      Cmsk_size = cv2.imread (im_data_locate + "/" + moji[k] + "/msk" + "/" + str(i) + ".bmp",0)
      Cmsk_size = cv2.resize(Cmsk_size,dsize=(512,512))
      Cmsk_size2 = 255 - Cmsk_size   #白黒反転
      msk_img.append(Cmsk_size2)
      print(moji[k] + str(i) + ".bmp " + str(j) + "°回転 読み込み")"""
      Corg_size = cv2.imread (im_data_locate + '/' + moji[k] + '/org' + '/' + str(i) + '.bmp')
      Corg_size = cv2.resize(Corg_size,dsize=(512,512))
      trans = cv2.getRotationMatrix2D(center, j , 1)
      org_img.append(cv2.warpAffine(Corg_size, trans, (width,height)))
      Cmsk_size = cv2.imread (im_data_locate + "/" + moji[k] + "/msk" + "/" + str(i) + ".bmp",0)
      Cmsk_size = cv2.resize(Cmsk_size,dsize=(512,512))
      Cmsk_size2 = 255 - Cmsk_size   #白黒反転
      msk_img.append(cv2.warpAffine(Cmsk_size2, trans, (width,height)))
      print(moji[k] + str(i) + ".bmp " + str(j) + "°回転 読み込み")
       
    """if j == 45:
     img = Image.open (im_data_locate + "/山/org" + "/" + str(i) + ".bmp").convert('RGB').resize((512,512)).rotate(j)
     img.show()"""
      
    """#確認
    for i in range(img_numY * 2):
     cv2.imwrite("D:/Blade/Pr/"+ str(i) + "_ex.bmp", org_img[i])
     cv2.imwrite("D:/Blade/Pr1/"+ str(i) + "_ex.bmp", msk_img[i])
      
      
    for i in range(img_numY):  
     tmpm_img.append( np.array( Image.open (im_data_locate + "/msk" + "/" + str(i) + ".bmp").convert('L').resize((512,512)) ) )
     tmpg_img.append( np.array( Image.open (im_data_locate + "/org" + "/" + str(i) + ".bmp").convert('RGB').resize((512,512)) ) )
     cv2.imwrite("D:/Blade/P/"+ str(i) + "_ex.bmp", tmpm_img[i])
     cv2.imwrite("D:/Blade/P1/"+ str(i) + "_ex.bmp", tmpg_img[i])"""

    
        
    #回転画像確認用
    
     
   
    return np.array(org_img),np.array(msk_img)



