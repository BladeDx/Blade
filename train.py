# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 13:00:58 2021

@author: Tatsuaki
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 15:30:23 2020

@author: uno
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
import pandas as pd
import cv2 
import time 
#from PIL import Image

#外部ソースファイル
import gen_model, plot, data_load, gen_folder, memo, metrics, Synthetictrain



#学習用関数
def learning(save_locate ,epochs, batch_size, loss_name, img_data_locate, img_height, img_width, img_numY,img_numU,img_numC):
    
 #原画像とマスク画像の読み込み(詳しくはdata_load関数へ→)
 org_img, msk_img = data_load.data_load(img_data_locate, img_numY,img_numU,img_numC)
 
 #正規化
 org_img = org_img / 255.0
 msk_img = msk_img / 255.0
 
 
 
 #訓練データと試験データに9:1で分割
 org_img_train,org_img_test,msk_img_train, msk_img_test = train_test_split(org_img, msk_img, test_size=0.2, random_state=0) 
 
 #デバッグ用
 #print(org_img_train.shape)
 #print(org_img_test.shape)
 #print(org_img_test[0].shape)
 #print(msk_img_train.shape)
 #print(msk_img_test.shape)
 #print(org_img.shape)
 #print(np.unique(org_img))
 #print(np.unique(msk_img))

 """#原画像の一例を保存
 org_img_test = np.clip(org_img_test * 255, 0, 255).astype(np.uint8)
 cv2.imwrite(save_locate + "/org_img_ex.bmp", org_img_test[0])
 #訓練データとするマスク画像の一例を保存
 #cv2.imwrite(save_locate + "/msk_img_ex.bmp", msk_img_test[0])
 ex_img = Image.fromarray(msk_img_test[0],'L')
 ex_img.save(save_locate + "/msk_img_ex.bmp")"""
 
 #モデルの構造の設定(詳しくはgen_model関数へ→)
 model = gen_model.make_Model01(img_height, img_width)
 
 #ネットワークを可視化
 tf.keras.utils.plot_model(model, show_shapes=True)
 #モデルサマリー
 print(model.summary());

 #モデルコンパイル
 model.compile(optimizer= optimizers.Adam(),
                  loss= loss_name,
                  metrics=["mse"])
 
 #モデル学習
 history = model.fit(org_img_train, msk_img_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(org_img_test, msk_img_test)
                    )

 #モデルの保存
 model.save(save_locate + "/model.h5")
 #モデルの重みの保存
 model.save_weights(save_locate + "/" + "model_weight.h5")
 #学習過程をCSVファイルに保存
 hist_df =pd.DataFrame(history.history) 
 with open(save_locate + "/history.csv", mode='w') as f:
   hist_df.to_csv(f)
     
 #学習過程をプロット
 plot.plot_history(history, 
                save_graph_img_path =  save_locate + "/graph.png", 
                fig_size_width = 45, 
                fig_size_height = 30)
    
 #学習後のモデル評価
 score = model.evaluate(org_img_test, msk_img_test, verbose=1)
 print('Test loss:', score[0])
 print('Test evaluate_mse:',score[1])
 
 
 
 #予測されたマスク画像の一例を保存
 pre_img = model.predict(org_img_test,verbose=1)
 pre_img = np.clip(pre_img * 255, 0, 255).astype(np.uint8)
 #pre_img = 255 - pre_img
 #pre_img = (pre_img > 0.5) * 255
 
 for i in range(len(pre_img)):
     for j in range(512):
         for k in range(512):
             if pre_img[i][j][k] < 128:
                pre_img[i][j][k] = 0
             else:
                pre_img[i][j][k] = 255
 
 #ret2,pre_img2 = cv2.threshold(pre_img,0,255,cv2.THRESH_OTSU)
 #print(pre_img)
 #print(pre_img2.shape)
 #pre_img = 255 - pre_img
 
 """for i in range(len(pre_img)):
     ret,pre_img2 = cv2.threshold(pre_img,0,255,cv2.THRESH_OTSU)
     pre_img[i] = (pre_img[i] > ret) * 255"""
     
 
 cv2.imwrite(save_locate + "/pred_msk_img_ex.bmp", pre_img[0])


 #原画像の一例を保存
 org_img_test = np.clip(org_img_test * 255, 0, 255).astype(np.uint8)
 cv2.imwrite(save_locate + "/org_img_ex.bmp", org_img_test[0])
 #訓練データとするマスク画像の一例を保存
 msk_img_test = np.clip(msk_img_test * 255, 0, 255).astype(np.uint8)
 msk_img = 255 - msk_img
 cv2.imwrite(save_locate + "/msk_img_ex.bmp", msk_img_test[0])
 #ex_img = Image.fromarray(msk_img_test[0],'L')
 #ex_img.save(save_locate + "/msk_img_ex.bmp")
 

 for i in range(len(org_img_test)):
     
     cv2.imwrite(save_locate + "/validate_org/" + str(i) + ".bmp", org_img_test[i])
     cv2.imwrite(save_locate + "/validate_msk/" + str(i) + ".bmp", msk_img_test[i])
     cv2.imwrite(save_locate + "/pred_msk/" + str(i) + ".bmp", pre_img[i])


#画像合成関数に飛ぶ 
 #Synthetictrain.synthetic(save_locate,org_img_test,pre_img)
 

 #print(pre_img2[i].shape)
 #print(pre_img.shape)
 iou = 0
#評価IOU
 for i in range(len(pre_img)):
     iou = iou + metrics.iou_score(pre_img[i],msk_img[i])
 
 #print(iou_score)
 global iou_score
 iou_score = iou / len(pre_img) 
 
          
 print("IOU: " + str(iou / len(pre_img)))


#各種設定
locate = "C:/Blade"
#保存フォルダ名
save_locate = gen_folder.gen_folder(locate)

#学習データを保存するディレクトリ作成
if not os.path.exists(save_locate):
     print("ディレクトリを作成します")
     os.makedirs(save_locate + '/validate_msk'); #元マスク画像
     os.makedirs(save_locate + '/validate_org'); #元画像
     os.makedirs(save_locate + '/pred_msk'); #予測マスク画像
     os.makedirs(save_locate + '/synthetic'); #予測マスク画像+元画像
     
#学習エポック数 
epochs = 2000
#学習時のバッチサイズ
batch_size= 4
#損失関数名
loss_name = "binary_crossentropy"

#画像ファイルの場所 org,mskのフォルダがあるパス
img_data_locate = "C:/Blade"
#画像の高さ
img_height = 512
#画像の幅
img_width = 512
#画像の枚数 Y:29 ,U:16 ,C:60(54-60 nemoto)
img_numY = 20
img_numU = 16
img_numC = 54

#iou_score
iou_score = 0

#メモの生成（確認用）
elapsed_time = 0
memo.gen_memo(save_locate, epochs, batch_size, loss_name, img_data_locate, img_height, img_width, img_numY,img_numU,img_numC , elapsed_time , iou_score)

#処理時間計測開始
start = time.time()

#学習の実行
learning(save_locate = save_locate, epochs=epochs,batch_size=batch_size,loss_name=loss_name, img_data_locate =img_data_locate, img_height=img_height, img_width=img_width, img_numY=img_numY, img_numU = img_numU, img_numC=img_numC)

#処理時間計測終了・記録
elapsed_time = (time.time() - start) // 60
print ("elapsed_time:{0}".format(elapsed_time) + "[min]")

#メモの生成
memo.gen_memo(save_locate, epochs, batch_size, loss_name, img_data_locate, img_height, img_width, img_numY,img_numU,img_numC , elapsed_time, iou_score)