# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 14:04:40 2021

@author: Tatsuaki
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 16:32:54 2020

@author: uno

historyの可視化

"""

import numpy as np
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#グラフの作成
def plot_history(history, save_graph_img_path, fig_size_width, fig_size_height ):
    #mse = history.history['mse']
    #val_mse = history.history['val_mse']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
   
    #epoc数の設定
    #epochs = range(len(mse))
    epochs = range(len(loss))

    #グラフ表示
    plt.figure(figsize=(fig_size_width, fig_size_height))
    plt.rcParams['font.family'] = 'Times New Roman'
    #plt.rcParams['font.size'] = lim_font_size  # 全体のフォント
    #plt.subplot(121)
    
    #グラフタイトルの設定
    #plt.title('Training and Validation RMSE',fontsize=60)
    plt.title('Training and Validation Loss',fontsize=60)
    #y軸を対数軸に
    plt.yscale("log")
    
    #軸ラベルの設定
    plt.xlabel("epochs",fontsize=60)
    plt.ylabel("LOSS",fontsize=60)
    #plt.ylabel("MSE",fontsize=60)
    
    #目盛りラベルのオプション
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)
    
    
    #グラフのプロット
    #plt.plot(epochs, np.sqrt(mse), color = "red", linestyle = "solid" ,label = 'train MSE',alpha = 0.8)
    #plt.plot(epochs, np.sqrt(val_mse), color = "blue", linestyle = "solid" , label= 'valid MSE',alpha = 0.8)
    plt.plot(epochs, loss, color = "red", linestyle = "solid" ,label = 'train LOSS',alpha = 0.8)
    plt.plot(epochs, val_loss, color = "blue", linestyle = "solid" , label= 'valid LOSS',alpha = 0.8)
    
    
    
    #凡例の表示
    plt.legend(fontsize=50)
    plt.rc('legend', fontsize=50)
    
    #グリッド表示
    plt.grid(which = "major", axis = "y", color = "black", alpha = 0.6,
        linestyle = "--", linewidth = 1.0)
    

    #グラフの保存
    plt.savefig(save_graph_img_path)
    
    # バッファ解放    
    plt.close() 
    
