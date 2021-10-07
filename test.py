# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 21:40:53 2021

@author: Owner
"""

import os
import numpy as np
import cv2 
from tensorflow.keras.models import load_model
#from PIL import Image
import metrics,Synthetic


#試験用関数
def test(locate, model_locate, test_img_num):
 
 org_img = []
 org_tmp = []
 msk_img = []
 msk_tmp = []
 moji = ["mount","sea","cloud"]
 
 #org_img.append( np.array(Image.open(locate + "/" + moji[locate_number] + "/org/" + str(test_img_num) + ".bmp").convert('RGB').resize((512,512))))
 #msk_img.append( np.array(Image.open(locate + "/" + moji[locate_number] + "/msk/" + str(test_img_num) + ".bmp").convert('L').resize((512,512))))
 org_tmp = cv2.imread (locate + "/" + moji[locate_number] + "/org/" + str(test_img_num) + ".bmp")
 org_img.append(cv2.resize(org_tmp,dsize=(512,512)))
 msk_tmp = cv2.imread (locate + "/" + moji[locate_number] + "/msk/" + str(test_img_num) + ".bmp",0)
 Ymsk_size = cv2.resize(msk_tmp,dsize=(512,512))
 msk_img.append(Ymsk_size)
 
 
 org_img = np.array(org_img) / 255
 msk_img = np.array(msk_img) / 255
 
 
 #モデルの構造の設定(詳しくはgen_model関数へ→)
 if os.path.exists(model_locate + "/model.h5"):
  model = load_model(model_locate + "/model.h5")
 else:
     print("モデルが存在しません")
     
 #モデルサマリー
 print(model.summary());
 
 #モデルの重みの保存
 model.load_weights(model_locate + "/" + "model_weight.h5")
 
 #予測されたマスク画像を保存
 pre_img = model.predict(org_img, verbose=1)

 #print(np.unique(pre_img[0]))
 pre_img = np.clip(pre_img * 255, 0, 255).astype(np.uint8)
 #pre_img = 255 - pre_img
 pre_img = (pre_img > 10) * 255
 #pre_img = np.where(pre_img < 0.1, 0, 255)

 cv2.imwrite(model_locate + "/test_pred_msk_img_ex.bmp", pre_img[0])
 
 #試験に用いた原画像を保存
 org_img = np.clip(org_img * 255, 0, 255).astype(np.uint8)
 cv2.imwrite(model_locate + "/test_org_img_ex.bmp", org_img[0])
 #マスク画像を保存
 #cv2.imwrite(model_locate + "/test_msk_img_ex.bmp", msk_img[0] )
 msk_img = np.clip(msk_img * 255, 0, 255).astype(np.uint8)
 #白黒反転
 msk_img = 255 - msk_img
 cv2.imwrite(model_locate + "/test_msk_img_ex.bmp", msk_img[0])
 """ex_img = Image.fromarray(msk_img[0],'L')
 ex_img = np.clip(ex_img * 255, 0, 255).astype(np.uint8)
 ex_img.save(model_locate + "/test_msk_img_ex.bmp")"""

#画像合成関数に飛ぶ 
 Synthetic.synthetic(model_locate,org_img,pre_img)


 iou = metrics.iou_score(pre_img[0],msk_img[0])
    
 print('iou_score:',iou)  
 #print(msk_img.shape)
 #print(pre_img.shape)
  
#学習後のモデル評価
 score = model.evaluate(org_img, msk_img, verbose=1)
 print('Test loss:', score[0])
 print('Test evaluate_mse:',score[1])

 #print(pre_img[0])

#各種設定

locate = "C:/Blade"
#学習済みのモデルが保存されているフォルダ名
model_locate = "C:/Blade/2021-09-28 12_08_16.662656"

#テストしたい画像絶対ファイルパス    Y:29 ,U:16 ,C:53   
locate_number = 2
test_img_num = 7

#試験の実行
test(locate=locate, model_locate=model_locate, test_img_num=test_img_num)