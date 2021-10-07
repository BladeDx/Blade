# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 13:59:03 2021

@author: takase
"""

import tkinter,tkinter.filedialog, tkinter.messagebox 
import os, sys, cv2
from PIL import Image, ImageTk
import metrics,Synthetic
from tensorflow.keras.models import load_model
import numpy as np




model_locate = "C:/Blade/2021-10-06 13_59_59.006775 normal 2000"


#exit　ボタンクリック時イベント
def btn0_click():
 window.destroy() 
 sys.exit()

#org_img ボタンクリック時イベント
def btn1_click():
    
    #変数宣言
   org_img = []
   org_tmp = []
   msk_img = []
    
    #ファイルタイプの指定[(ウインドウの見出し,ファイルの拡張子)]
   fTyp = [("","*.jpg")]
    
    #実行中のファイルの絶対パスを取得
   iDir = os.path.abspath(os.path.dirname(__file__))
    
    #元画像の選択
   tkinter.messagebox.showinfo('元画像の選択','元画像を選択してください！')
   file1 = tkinter.filedialog.askopenfilename(filetypes = fTyp,initialdir = iDir)
   if len(file1) == 0:
    tkinter.messagebox.showerror('エラー', '元画像が選択されていません') 
    
   print(file1) 
   
   # 処理ファイル名の出力
   
   pil_image1 = Image.open(file1)
   photo_image1 = ImageTk.PhotoImage(image=pil_image1)
   canvas1.create_image(0,0,anchor="nw",image= photo_image1)
   #canvas1.pack()
   #canvas1.mainloop()
   
   org_tmp = cv2.imread (file1)
   org_img.append(cv2.resize(org_tmp,dsize=(512,512)))
   
   #pred_msk 予測
   org_img = np.array(org_img) / 255
   
   #print(org_img)

   
   if os.path.exists(model_locate + "/model.h5"):
      model = load_model(model_locate + "/model.h5")
   else:
      tkinter.messagebox.showerror('エラー', 'モデルが存在しません') 
      sys.exit()
  
 
    #モデルの重みの読み出し
   model.load_weights(model_locate + "/" + "model_weight.h5")
 
    #予測されたマスク画像を保存
   pre_img = model.predict(org_img, verbose=1)

   pre_img = np.clip(pre_img * 255, 0, 255).astype(np.uint8)

   pre_img = (pre_img > 10) * 255

   cv2.imwrite(model_locate + "/test_pred_msk_img_ex.bmp", pre_img[0])
 

    #試験に用いた原画像を保存
   org_img = np.clip(org_img * 255, 0, 255).astype(np.uint8)
   cv2.imwrite(model_locate + "/test_org_img_ex.bmp", org_img[0])

 

   #学習後のモデル評価
   """score = model.evaluate(org_img, msk_img, verbose=1)
   print('Test loss:', score[0])
   print('Test evaluate_mse:',score[1])"""


    #画像合成関数に飛ぶ 
   rgb_pre_image = Synthetic.synthetic(model_locate, org_img[0], pre_img[0])
   
   canvas3.create_image(0,0,anchor="nw",image= rgb_pre_image)
   
   canvas1.mainloop()
   
   
    
   
    
def btn2_click():
    #ファイルタイプの指定[(ウインドウの見出し,ファイルの拡張子)]
   fTyp = [("","*.jpg")]
    
    #実行中のファイルの絶対パスを取得
   iDir = os.path.abspath(os.path.dirname(__file__))
    
    #元画像の選択
   tkinter.messagebox.showinfo('マスク画像の選択','マスク画像を選択してください！')
   file2 = tkinter.filedialog.askopenfilename(filetypes = fTyp,initialdir = iDir)
   if len(file2) == 0:
    tkinter.messagebox.showerror('エラー', 'マスク画像が選択されていません')  
    
   print(file2)    
        
   # 処理ファイル名の出力
   pil_image2 = Image.open(file2)
   photo_image2 = ImageTk.PhotoImage(image=pil_image2)
   canvas2.create_image(0,0,anchor="nw",image= photo_image2)
   #canvas1.pack()
   canvas2.mainloop()
        
   
   




#ウィンドウ作成
window = tkinter.Tk()
window.geometry("1064x1200")
window.title("test")


#キャンバス作成
canvas1 = tkinter.Canvas(window, bg="blue", height=512, width=512)
#キャンバス位置
canvas1.place(x=10,y=0)


#キャンバス作成
canvas2 = tkinter.Canvas(window, bg="blue", height=512, width=512)
#キャンバス位置
canvas2.place(x=542,y=0)


#キャンバス作成
canvas3 = tkinter.Canvas(window, bg="white", height=512, width=512)
#キャンバス位置
canvas3.place(x=266,y= 532)

#終了 ボタンの作成
btn0 = tkinter.Button(window, text='exit',command=btn0_click)
btn0.place(x=911,y=600)

#org_img ボタンの作成
btn1 = tkinter.Button(window, text='org_img',command=btn1_click)
btn1.place(x=133,y=532)

#msk_img ボタンの作成
btn1 = tkinter.Button(window, text='msk_img',command=btn2_click)
btn1.place(x=911,y=532)








#img = tkinter.PhotoImage(file= "",width=512,height=512)
#canvas.create_image(512, 512, image=img, anchor=tkinter.NW)

window.mainloop()