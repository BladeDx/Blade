import copy
from tkinter import *
import tkinter.filedialog as tkfd
import cv2
import numpy as np
import random
from glob import glob
import os
from cts_GUI import cts_GUI
from cts_processing import cts_processing
from cts_CNN_prediction import *


class cts_Main(cts_GUI, cts_processing, CNN_prediction):
    def __init__(self):
        # 事前設定GUIを開く
        self.dataset_select_window()

        self.CNN_prediction_flag = True
        self.dataset_name = os.path.basename(self.dataset_path)

        # 対象フォルダの選択
        self.flu_dataset_names = []
        if self.fluorescent_flag:
            for i in range(self.fluorescent_num):
                dirpath = tkfd.askdirectory()
                self.flu_dataset_names.append(os.path.basename(dirpath))
        print(self.dataset_name)
        print(self.flu_dataset_names)

        # 原画像を読み込み
        self.image_number = 0
        self.imgs = []  # 読み込んだ原画像(MONO)セット
        self.cfps = []  # 読み込んだCFP画像(MONO)セット
        self.mcherrys = []  # 読み込んだmCherry画像(MONO)セット
        self.yfps = []  # 読み込んだYFP画像(MONO)セット
        self.predicts = []  # 読み込んだ or 生成されたCNN予測画像セット
        # self.predicts_hand_flag = []  # predictsに手動処理を加えたかどうか
        self.markers = []  # 生成されるマーカー画像セット
        self.hands = []  # 手動処理が行われた画像セット
        self.segments = []  # 完全にセグメンテーションされた画像セット（無修正、修正画像が混在）[画像、手動処理を加えた画像か(0 or 1)]
        self.imgs_tk = []  # 原画像セットGUI表示用
        self.segments_tk = []  # 予測画像セットGUI表示用
        self.img_ids = glob(os.path.join(self.dataset_path, 'BF', '*'))
        self.img_ids = [os.path.splitext(os.path.basename(p))[0] for p in self.img_ids]
        self.img_ids = [p.replace('_Bright', '') for p in self.img_ids]
        self.img_totalnum = len(self.img_ids)  # ファイルの画像枚数
        self.dsts_tk = []  # 諸々の処理結果画像(COLOR)セットGUI表示用
        print(self.img_ids)

        # 画像処理で用いる閾値（ゴミ取り、円形度）
        self.cleanth = 10
        self.circuth = 0.0
        self.hist_tone = 0
        self.hist_bri_lower = 15  # ヒストグラムの輝度値の下限
        self.hist_bri_upper = 200  # ヒストグラムの輝度値の上限
        self.hist_freq_range_flu1 = 10  # ヒストグラムの各座標の輝度の発生頻度最大のところから何％取るかの範囲
        self.hist_freq_range_flu2 = 10  # ヒストグラムの各座標の輝度の発生頻度最大のところから何％取るかの範囲
        self.hist_freq_range_flu3 = 10  # ヒストグラムの各座標の輝度の発生頻度最大のところから何％取るかの範囲

        # 重みの設定
        self.wc = 20
        self.wp = 1
        self.wa = 2
        # 候補細胞の閾値設定(対象細胞と現画像の細胞の距離がself.r以内の細胞を候補細胞としている）
        self.r = 50
        # アニーリングの閾値設定
        self.reg = 50

        self.mCherry_flag = False
        self.YFP_flag = False

        if "mCherry" in self.flu_dataset_names:
            self.mCherry_flag = True
        if "YFP" in self.flu_dataset_names:
            self.YFP_flag = True
        self.textflag = True  # 画像のテキストの有無
        self.tracking_flag = False  # ラベリング追跡処理のフラグ、0は追跡なし1は追跡あり
        self.seg_mode_flag = False  # モード変更フラグ
        self.labelchange_register_flag = False  # ラベル変更をラベル記録リストself.label_correction_listに入れるかどうか
        self.gui_change_flag = True  # 右側gui画面の画像が画像処理が行われたときに表示させるか
        #self.hist_gui_flag = True  # ヒストグラムの表示を操作するフラグ
        self.data_save_only_flag = False  # データ保存のときのみ行う処理のオンオフ

        self.labelchangeframenum = []  # labelを変更したフレームを入れておくリスト
        self.labelchange = []  # labelを変更した細胞追跡リストを保存しておく[フレーム, self.cell_truck_labellist, self.cell_truck_detaillist]
        self.label_correction_list = []  # 変更したラベルを記録しておくリスト [フレーム、変更前ラベル、変更後ラベル]
        self.hist_label = []  # 細胞ヒストグラムのためのラベルが1以上の領域（つまり細胞）だけを集めたリスト [label, x, y]
        self.hist_mcherry = []  # mcherryヒストグラムのための該当画素のmcherryの輝度値を集めたリスト
        self.hist_yfp = []  # 上記のyfpバージョン

        self.fts = []  # 蛍光画像取得時間

        # 出力画像用のディレクトリ作成
        if not os.path.exists(os.path.join(self.dataset_path, 'Predicts')):
            os.mkdir(os.path.join(self.dataset_path, 'Predicts'))
        if not os.path.exists(os.path.join(self.dataset_path, 'Hands')):
            os.mkdir(os.path.join(self.dataset_path, 'Hands'))

        # 色付け用
        self.colors = []
        for i in range(10000):
            self.colors.append(np.array([random.randint(0, 128), random.randint(0, 255), random.randint(0, 255)]))

        # 自動処理後画像を先にセットしておくかどうか
        self.PreProcess()
        self.PreDataEdit()

        # 原, cfp画像の取得
        for i in range(self.img_totalnum):
            src = cv2.imread(f'{self.dataset_path}/BF/{self.img_ids[i]}_Bright.tif', 0)
            cfp = cv2.imread(f'{self.dataset_path}/CFP/{self.img_ids[i]}_CFP.tif', 0)  # MONOで取得
            if self.mCherry_flag:
                mcherry = cv2.imread(f'{self.dataset_path}/mCherry/'
                                     f'{self.img_ids[i]}_mCherLED.tif', 0)
            if self.YFP_flag:
                yfp = cv2.imread(f'{self.dataset_path}/YFP/'
                                 f'{self.img_ids[i]}_TaYFP.tif', 0)

            if i == 0:
                self.height_src, self.width_src = src.shape
                self.src_sizecount = 0
                while self.height_src > 1000 or self.width_src > 1000:
                    tmp = cv2.resize(src, dsize=None, fx=0.5, fy=0.5)
                    self.height_src, self.width_src = tmp.shape
                    self.src_sizecount += 1

                height_flu, width_flu = cfp.shape
                self.flu_sizecount = 0
                while height_flu > 1000 or width_flu > 1000:
                    tmp = cv2.resize(cfp, dsize=None, fx=0.5, fy=0.5)
                    height_flu, width_flu = tmp.shape
                    self.flu_sizecount += 1

            for sc in range(self.src_sizecount):
                src = cv2.resize(src, dsize=None, fx=0.5, fy=0.5)
            for sc in range(self.flu_sizecount):
                cfp = cv2.resize(cfp, dsize=None, fx=0.5, fy=0.5)
                if self.mCherry_flag:
                    mcherry = cv2.resize(mcherry, dsize=None, fx=0.5, fy=0.5)
                if self.YFP_flag:
                    yfp = cv2.resize(yfp, dsize=None, fx=0.5, fy=0.5)

            _, cfp = cv2.threshold(cfp, 40, 255, cv2.THRESH_BINARY)

            # 切り出し
            center = [self.height_src // 2, self.width_src // 2]
            cfp = cfp[center[0] - 256: center[0] + 256, center[1] - 256: center[1] + 256]
            if self.mCherry_flag:
                mcherry = mcherry[center[0] - 256: center[0] + 256, center[1] - 256: center[1] + 256]
                self.mcherrys.append(copy.deepcopy(mcherry))
            if self.YFP_flag:
                yfp = yfp[center[0] - 256: center[0] + 256, center[1] - 256: center[1] + 256]
                self.yfps.append(copy.deepcopy(yfp))

            self.imgs.append(copy.deepcopy(src))
            self.cfps.append(copy.deepcopy(cfp))
            tk = self.change_tk_format(copy.deepcopy(src), False)
            self.imgs_tk.append(tk)

        # 予測画像の取得
        pred_ids = glob(os.path.join(self.dataset_path, 'Predicts', '*'))
        if pred_ids:
            self.CNN_prediction_flag = False

        if self.CNN_prediction_flag:
            # CNNの処理入れる
            self.CNN_prediction(self.dataset_path, self.img_ids, self.imgs)

        for i in range(self.img_totalnum):
            self.image_number = i
            predict = cv2.imread(
                f'{self.dataset_path}/Predicts/{self.img_ids[i]}_predict.tif', 0)  # MONOで取得
            _, predict = cv2.threshold(predict, 128, 255, cv2.THRESH_BINARY)
            if i == 0:
                self.height_pred, self.width_pred = predict.shape

            # initialize a black image
            dst = np.zeros((self.height_pred, self.width_pred, 3), dtype=np.uint8)

            # dstにpredictをコピー＋CFP領域処理
            # for y in range(self.height_pred):
            #     for x in range(self.width_pred):
            #         dst[y, x, :] = predict[y, x]
            #         if self.cfps[i][y, x] == 255:  # もしcfpの領域内なら青色に表示
            #             dst[y, x] = [255, 0, 0]

            self.predicts.append(copy.deepcopy(predict))
            self.hands.append(copy.deepcopy(predict))
            self.segments.append([copy.deepcopy(predict), 0])
            self.markers.append(copy.deepcopy(predict))
            self.segments_tk.append(self.change_tk_format(copy.deepcopy(predict), True))
            self.dsts_tk.append(copy.deepcopy(dst))

        self.image_number = 0
        self.labeling(self.predicts[self.image_number])

        # 細胞解析支援GUIを起動
        self.cts_window()

    def entry_data(self):
        self.fluorescent_flag = self.fluorescent_flag_BV.get()

        if self.fluorescent_flag:
            self.fluorescent_num = int(self.fluorescent_num_entry.get())

        self.dataset_select_frame.destroy()

    def change_state(self):
        if self.fluorescent_flag_BV.get():
            self.fluorescent_num_entry.configure(state='normal')
        else:
            self.fluorescent_num_entry.configure(state='disabled')

    def dataset_select_window(self):
        self.dataset_select_frame = Toplevel()
        self.dataset_path = tkfd.askdirectory()
        print(self.dataset_path)
        self.dataset_select_frame.title("Dataset Select")
        self.dataset_select_frame.geometry("250x150")
        # 各種ウィジェットの作成
        fluorescent_num_label = Label(self.dataset_select_frame, text="Sets：")
        self.fluorescent_num_entry = Entry(self.dataset_select_frame, width=5, state='disabled')
        self.fluorescent_flag_BV = BooleanVar()
        fluorescent_chb = Checkbutton(self.dataset_select_frame, variable=self.fluorescent_flag_BV,
                                      text='Use of fluorescent images', command=self.change_state)
        imgset_button = Button(self.dataset_select_frame, text="Start", command=self.entry_data, width=15)

        # 各種ウィジェットの設置
        fluorescent_chb.grid(row=0, column=1, padx=5, pady=5)
        fluorescent_num_label.grid(row=1, column=1, sticky="w")
        self.fluorescent_num_entry.grid(row=1, column=1)
        imgset_button.grid(row=2, column=1)

        self.dataset_select_frame.mainloop()

    def cts_window(self):
        self.frame = Toplevel()
        self.frame.title("CellTruckingSystem")
        self.canvas_src = Canvas(self.frame, width=self.width_src, height=self.height_src)
        self.canvas_pred = Canvas(self.frame, width=self.width_pred, height=self.height_pred)
        self.canvas_src.grid(row=1, column=0, columnspan=5, rowspan=1)
        self.canvas_pred.grid(row=1, column=5, columnspan=4, rowspan=1)
        self.image_on_canvas_src = self.canvas_src.create_image(0, 0, anchor=NW, image=self.imgs_tk[self.image_number])
        self.image_on_canvas_pred = self.canvas_pred.create_image(0, 0, anchor=NW,
                                                                  image=self.dsts_tk[self.image_number])

        #self.canvas_pred.bind("<Motion>", self.on_motioned)
        self.canvas_pred.bind("<ButtonPress-1>", self.on_pressed)
        self.canvas_pred.bind("<B1-Motion>", self.on_dragged)
        self.canvas_pred.bind("<ButtonRelease-1>", self.on_released)

        # 現在の画像番号を表示するEntry
        self.message_imgnum = Entry(self.frame, width=40)
        self.message_imgnum.insert(END, f'This image is {self.img_ids[self.image_number]}')
        self.message_imgnum.grid(row=0, column=0, columnspan=1, sticky="w")

        pennames = ["Split", "Join"]

        self.penname = StringVar()
        self.penname.set(pennames[1])
        self.b = OptionMenu(self.frame, self.penname, *pennames)
        self.b.grid(row=3, column=6, columnspan=1, rowspan=1)
        self.penwidth = Scale(self.frame, from_=1, to=5, orient=HORIZONTAL)
        self.penwidth.set(1)
        self.penwidth.grid(row=4, column=6, columnspan=1, rowspan=1)

        # 「消す」ボタン
        self.button_erase = Button(self.frame, text=u'erase', width=15)
        self.button_erase.bind("<Button-1>", self.erase)
        self.button_erase.grid(row=5, column=6, columnspan=1, rowspan=1)

        self.button_last = Button(
            self.frame, text="⇛", command=self.onLastButton, width=5)
        self.button_last.grid(row=2, column=6, columnspan=1, rowspan=1, sticky="")
        self.button_next5 = Button(
            self.frame, text="⇒", command=self.onNext5Button, width=5)
        self.button_next5.grid(row=2, column=6, columnspan=1, rowspan=1, sticky="w")
        self.button_next = Button(
            self.frame, text="→", command=self.onNextButton, width=5)
        self.button_next.grid(row=2, column=5, columnspan=1, rowspan=1, sticky="w")
        self.button_back = Button(
            self.frame, text="←", command=self.onBackButton, width=5)
        self.button_back.grid(row=2, column=4, columnspan=1, rowspan=1, sticky="e")
        self.button_back5 = Button(
            self.frame, text="⇐", command=self.onBack5Button, width=5)
        self.button_back5.grid(row=2, column=4, columnspan=1, rowspan=1, sticky="")
        self.button_first = Button(
            self.frame, text="⇚", command=self.onFirstButton, width=5)
        self.button_first.grid(row=2, column=4, columnspan=1, rowspan=1, sticky="w")

        # 開始番号
        self.slbl = Label(self.frame, text="Start Num")
        self.slbl.grid(row=3, column=8, sticky="w")
        # 開始番号入力欄
        self.sEnt = Entry(self.frame, width=5)
        self.sEnt.insert("end", "1")
        self.sEnt.grid(row=3, column=8, sticky="")
        # 終了番号
        self.flbl = Label(self.frame, text="End Num")
        self.flbl.grid(row=4, column=8, sticky="w")
        # 開始番号入力欄
        self.fEnt = Entry(self.frame, width=5)
        self.fEnt.insert("end", str(self.img_totalnum))
        self.fEnt.grid(row=4, column=8, sticky="")

        # ラべルの変更
        # 開始番号
        self.blbl = Label(self.frame, text="Label to be changed")
        self.blbl.grid(row=3, column=7, sticky="w")
        # 開始番号入力欄
        self.bEnt = Entry(self.frame, width=5)
        self.bEnt.insert("end", "0")
        self.bEnt.grid(row=3, column=7, sticky="e")
        # 終了番号
        self.albl = Label(self.frame, text="Changed label")
        self.albl.grid(row=4, column=7, sticky="w")
        # 終了番号入力欄
        self.aEnt = Entry(self.frame, width=5)
        self.aEnt.insert("end", "0")
        self.aEnt.grid(row=4, column=7, sticky="e")
        # ラべル変更
        self.button_labelch = Button(
            self.frame, text="Label Change", command=self.LabelChangeButton, width=10)
        self.button_labelch.grid(row=5, column=7, columnspan=1, rowspan=1, sticky="")

        # スレッショルドのスライダを作る
        tones = [1, 2, 4, 8]

        # self.hist_lab = Label(self.frame, text="Histogram")
        # self.hist_lab.grid(row=2, column=0, sticky="w")
        self.tone_lab = Label(self.frame, text="Shades of luminance")
        self.tone_lab.grid(row=2, column=0, sticky="w")
        self.tone_num = StringVar()
        self.tone_num.set(tones[0])
        self.tone_OM = OptionMenu(self.frame, self.tone_num, *tones)
        self.tone_OM.grid(row=2, column=0, columnspan=1, rowspan=1)
        self.hist_bri_lowersl = Scale(self.frame, label='bright_lower', orient='h',
                                      from_=0, to=255, length=256, command=self.SliderHistBrightLower)
        self.hist_bri_lowersl.set(self.hist_bri_lower)
        self.hist_bri_lowersl.grid(row=3, column=0, rowspan=2, sticky="w")
        self.hist_freq_range_flu1sl = Scale(self.frame, label="freq_" + self.flu_dataset_names[0], orient='h',
                                            from_=1, to=100, length=100, command=self.SliderHistFrequencyRangeflu1)
        self.hist_freq_range_flu1sl.set(self.hist_freq_range_flu1)
        self.hist_freq_range_flu1sl.grid(row=5, column=0, columnspan=2, rowspan=2, sticky="w")
        self.hist_freq_range_flu2sl = Scale(self.frame, label="freq_" + self.flu_dataset_names[1], orient='h',
                                            from_=1, to=100, length=100, command=self.SliderHistFrequencyRangeflu2)
        self.hist_freq_range_flu2sl.set(self.hist_freq_range_flu2)
        self.hist_freq_range_flu2sl.grid(row=7, column=0, columnspan=2, rowspan=2, sticky="w")
        # self.hist_freq_range_flu3sl = Scale(self.frame, label='frequency_flu3', orient='h',
        #                                     from_=1, to=100, length=100, command=self.SliderHistFrequencyRangeflu3)
        # self.hist_freq_range_flu3sl.set(self.hist_freq_range_flu3)
        # self.hist_freq_range_flu3sl.grid(row=8, column=0, columnspan=2, rowspan=2, sticky="")

        # セグメンテーションモードに切り替え
        self.button_segmode = Button(
            self.frame, text="Fix Mode", command=self.SegmentationModeButton, width=15)
        self.button_segmode.grid(row=3, column=4, columnspan=1, rowspan=1)
        # ラべリングモードに切り替え
        self.button_labelmode = Button(
            self.frame, text="Truck Mode", command=self.TruckingModeButton, width=15)
        self.button_labelmode.grid(row=4, column=4, columnspan=1, rowspan=1)

        # セグメンテーション画像を保存
        self.button_segsave = Button(
            self.frame, text="Segment Save", command=self.SegmentationImageSaveButton, width=15)
        self.button_segsave.grid(row=7, column=6, columnspan=1, rowspan=1)
        # セグメンテーション画像の消去
        self.button_segdel = Button(
            self.frame, text="Segment Del", command=self.SegmentationImageDeleteButton, width=15)
        self.button_segdel.grid(row=6, column=8, columnspan=1, rowspan=1)
        # セグメンテーション画像（軌跡）を保存している画像に置き換える
        self.button_seginstall = Button(
            self.frame, text="Segment DL", command=self.SegmentationImageDownloadButton, width=15)
        self.button_seginstall.grid(row=5, column=8, columnspan=1, rowspan=1)

        # 画像のtext表示の切り替え
        self.button_text = Button(
            self.frame, text="text", command=self.TextButton, width=5)
        self.button_text.grid(row=2, column=8, columnspan=1, rowspan=1, sticky="w")

        # 透過した原画像をセグメンテーション画像に貼り付ける
        # self.button_paste = Button(
        #     self.frame, text="src paste", command=self.PasteButton, width=5)
        # self.button_paste.grid(row=2, column=7, columnspan=1, rowspan=1, sticky="w")

        # 画像保存
        self.button_imgcripsave = Button(
            self.frame, text="画像保存", command=self.ImageSaveButton, width=10)
        self.button_imgcripsave.grid(row=7, column=8, columnspan=1, rowspan=1)

        # 分析データ保存
        # self.button_graph = Button(
        #     self.frame, text="ヒストグラム表示", command=self.HistButton, width=15)
        # self.button_graph.grid(row=8, column=7, columnspan=1, rowspan=1)

        # 分析データ保存
        self.button_graph = Button(
            self.frame, text="Data Save", command=self.DataSaveButton, width=15)
        self.button_graph.grid(row=8, column=8, columnspan=1, rowspan=1)

        self.frame.mainloop()


# ----------------------------------------------------------------------
if __name__ == "__main__":
    cts_Main()
    # Windowを生成する。
    # Windowについて : https://kuroro.blog/python/116yLvTkzH2AUJj8FHLx/
    # root = Tk()
    # app = cell_tracking_app_Main(master=root)

    # Windowをループさせて、継続的にWindow表示させる。
    # mainloopについて : https://kuroro.blog/python/DmJdUb50oAhmBteRa4fi/
   # app.mainloop()

