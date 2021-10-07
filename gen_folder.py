# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 15:39:46 2021

@author: Owner
"""

import datetime

def gen_folder(locate):
   
    datetime.date.today()#今日の日付
    datetime.datetime.today()#時間まで
    
    datetime.date.today().year#年
    datetime.date.today().day#日
    
    datetime.datetime.today().microsecond#１秒以下も出る
    
    datetime.date.today().isoformat()#フォーマットの指定
    datetime.datetime.today().strftime("%Y/%m/%d/%H/%M")#フォーマットの指定
    
    
    return  locate  + "/" + str(datetime.datetime.now()).replace(":","_") 