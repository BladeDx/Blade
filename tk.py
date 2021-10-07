# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 16:31:29 2021

@author: takase
"""
"""from tkinter import * 
import tkinter.filedialog as tkfd
import os,  tkinter.filedialog, tkinter.messagebox

dataset_path = tkfd.askdirectory()
print(dataset_path)"""

"""img_ids = glob(os.path.join(dataset_path, '/validate_org/', '*'))
img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
img_ids = [p.replace('_Bright', '') for p in img_ids]
img_totalnum = len(img_ids)  # ファイルの画像枚数"""



import os, tkinter, tkinter.filedialog, tkinter.messagebox
import cv2 
import metrics,Synthetic
import numpy as np


locate = "C:/Blade"
#学習済みのモデルが保存されているフォルダ名
model_locate = "C:/Blade/2021-09-21 19_32_56.726585 4000"


org_img = []
org_tmp = []
msk_img = []
msk_tmp = []

# ファイル選択ダイアログの表示
root = tkinter.Tk()
root.withdraw()
fTyp = [("","*")]
iDir = os.path.abspath(os.path.dirname(__file__))
tkinter.messagebox.showinfo('○×プログラム','元画像を選択してください！')
file1 = tkinter.filedialog.askopenfilename(filetypes = fTyp,initialdir = iDir)
tkinter.messagebox.showinfo('○×プログラム','マスク画像を選択してください！')
file2 = tkinter.filedialog.askopenfilename(filetypes = fTyp,initialdir = iDir)

print(file1)
print(file2)

# 処理ファイル名の出力
org_tmp = cv2.imread (file1)
org_img.append(cv2.resize(org_tmp,dsize=(512,512)))
msk_tmp = cv2.imread (file2,0)
Ymsk_size = cv2.resize(msk_tmp,dsize=(512,512))
msk_img.append(Ymsk_size)



org_img = np.array(org_img)
msk_img = np.array(msk_img)

print(org_img.shape)
print(msk_img.shape)

cv2.imshow("a",org_img[0])
cv2.waitKey(0)
cv2.destroyAllWindows()


#画像合成関数に飛ぶ 
msk_img = 255 - msk_img
keka = Synthetic.synthetic(model_locate,org_img,msk_img)
cv2.imshow("a",keka)
cv2.waitKey(0)
cv2.destroyAllWindows()


  


