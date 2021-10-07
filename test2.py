# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 16:37:16 2021

@author: takase
"""

import os, tkinter, tkinter.filedialog, tkinter.messagebox
import cv2 
import sys
import numpy as np
from tensorflow.keras.models import load_model
#from matplotlib import pyplot as plt

#外部ソースファイル
import metrics,Synthetic



#学習済みのモデルが保存されているフォルダ名
model_locate = "C:/Blade/2021-10-04 20_41_12.615054 souasa 4000"

#変数宣言
org_img = []
org_tmp = []
msk_img = []
msk_tmp = []
keka = []


# ファイル選択ダイアログの表示
root = tkinter.Tk()

#ウィンドウの非表示
root.withdraw()
#ファイルタイプの指定[(ウインドウの見出し,ファイルの拡張子)]
fTyp = [("","*.bmp")]

#実行中のファイルの絶対パスを取得
iDir = os.path.abspath(os.path.dirname(__file__))

#元画像の選択
tkinter.messagebox.showinfo('tk2.py','元画像を選択してください！')
file1 = tkinter.filedialog.askopenfilename(filetypes = fTyp,initialdir = iDir)
if len(file1) == 0:
   tkinter.messagebox.showerror('エラー', '元画像が選択されていません') 
   sys.exit()         

#マスク画像の選択 
tkinter.messagebox.showinfo('tk2.py','マスク画像を選択してください！')
file2 = tkinter.filedialog.askopenfilename(filetypes = fTyp,initialdir = iDir)
if len(file2) == 0:
   tkinter.messagebox.showerror('エラー', 'マスク画像が選択されていません') 
   sys.exit()
   
print(file1)
print(file2)



# 処理ファイル名の出力
org_tmp = cv2.imread (file1)
org_img.append(cv2.resize(org_tmp,dsize=(512,512)))
msk_tmp = cv2.imread (file2,0)
Ymsk_size = cv2.resize(msk_tmp,dsize=(512,512))
msk_img.append(Ymsk_size)


#ここから
#正規化
org_img = np.array(org_img) / 255
msk_img = np.array(msk_img) / 255
 
 
 #モデルの構造の設定(詳しくはgen_model関数へ→)
if os.path.exists(model_locate + "/model.h5"):
 model = load_model(model_locate + "/model.h5")
else:
  tkinter.messagebox.showerror('エラー', 'モデルが存在しません') 
  sys.exit()
  
    
#モデルサマリー
#print(model.summary());
 
#モデルの重みの読み出し
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
cv2.imwrite(model_locate + "/test_msk_img_ex.bmp", msk_img[0] )
"""msk_img = np.clip(msk_img * 255, 0, 255).astype(np.uint8)"""
#白黒反転
msk_img = 255 - msk_img
cv2.imwrite(model_locate + "/test_msk_img_ex.bmp", msk_img[0])
#
"""ex_img = Image.fromarray(msk_img[0],'L')
ex_img = np.clip(ex_img * 255, 0, 255).astype(np.uint8)
ex_img.save(model_locate + "/test_msk_img_ex.bmp")"""

#画像合成関数に飛ぶ 

"""keka.append(Synthetic2.synthetic(model_locate,org_img,pre_img))
keka = np.array(keka)"""

#iou計算
#iou = metrics.iou_score(pre_img[0],msk_img[0])
#print('iou_score:',iou)  

 

#学習後のモデル評価
score = model.evaluate(org_img, msk_img, verbose=1)
print('Test loss:', score[0])
print('Test evaluate_mse:',score[1])

#print(pre_img[0])


#画像合成関数に飛ぶ 
#msk_img = 255 - msk_img
Synthetic.synthetic(model_locate, org_img, pre_img)







  


