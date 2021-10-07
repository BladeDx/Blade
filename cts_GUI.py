import csv
import math
from tkinter import *
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import copy
import pandas as pd
import matplotlib.pyplot as plt
import datetime


class cts_GUI():
    def PreProcess(self):
        # PreDataファイルから変更済みラベルをself.label_correction_listに保存
        if os.path.exists(f'{self.dataset_path}/PreData.csv'):
            self.df = pd.read_csv(f'{self.dataset_path}/PreData.csv')
            for i in range(len(self.df)):
                if self.dataset_name == str(self.df.iat[i, 0]):
                    label_correction_list = self.df.iat[i, 1]
                    # print(label_correction_list)
                    if label_correction_list != "[]":
                        label_correction_list = ''.join(x for x in label_correction_list if x not in "[]")
                        label_correction_list = label_correction_list.split(",")
                        label_correction_list = [int(x) for x in label_correction_list]
                        self.label_correction_list = np.array(label_correction_list).reshape(-1, 3).tolist()
                        # print(self.label_correction_list)

        else:
            self.df = pd.DataFrame([[self.dataset_name, "[]"]],
                                   columns=["Stack Number", "Label Correction Log[frame, before_lab, after_lab]"])
            self.df.to_csv(f'{self.dataset_path}/PreData.csv', index=False)

        if os.path.exists(f'{self.dataset_path}/c1 Regions.csv'):
            df = pd.read_csv('./#35_c1/c1 Regions.csv')
            for i in range(len(df)):
                df.iloc[i, 1] = df.iloc[i, 1][:df.iloc[i, 1].find(".")]
                # print(df.iloc[i, 1])
                ftdatetime = datetime.datetime.strptime(df.iloc[i, 1], '%Y-%m-%dT%H:%M:%S')
                self.fts.append(ftdatetime)
        else:
            for i in range(self.img_totalnum):
                self.fts.append(0)

    def PreDataEdit(self):
        self.df_pidx = 0  # dfの該当の画像セットがある列番号
        self.df = pd.read_csv(f'{self.dataset_path}/PreData.csv')
        for i in range(len(self.df)):
            if self.dataset_name == str(self.df.iat[i, 0]):
                self.df_pidx = i
                break

            elif i == len(self.df) - 1:
                self.df_pidx = i
                self.df_add = pd.DataFrame([[self.dataset_name, "[]"]],
                                           columns=["Stack Number",
                                                    "Label Correction Log[frame, before_lab, after_lab]"])
                self.df_add.to_csv(f'{self.dataset_path}/PreData.csv', mode='a', header=False, index=False)

    def ImageZoom(self):
        crop_img = copy.deepcopy(self.hands[self.image_number])
        crop_img = self.srcPaste(crop_img)
        crop_img = crop_img[self.sy - 25: self.sy + 25, self.sx - 25: self.sx + 25]
        height = crop_img.shape[0]
        width = crop_img.shape[1]
        zoom_img = cv2.resize(crop_img, (int(width * 10), int(height * 10)))
        self.canvas_src.itemconfig(self.image_on_canvas_src, image=self.change_tk_format(zoom_img, False))

    def change_tk_format(self, img, srcpaste_flag):
        if srcpaste_flag:
            img = self.srcPaste(img)
        if img.ndim == 3:  # もしカラー画像なら
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # imreadはBGRなのでRGBに変換
        image_pil = Image.fromarray(img)  # RGBからPILフォーマットへ変換
        self.tk = ImageTk.PhotoImage(image_pil)  # ImageTkフォーマットへ変換

        return self.tk

    def srcPaste(self, img):  # 512x512の画像限定
        center = [self.height_src // 2, self.width_src // 2]
        # 512x512に切り出し
        src_crop = self.imgs[self.image_number][center[0] - 256: center[0] + 256, center[1] - 256: center[1] + 256]
        # コントラスト低減
        transparent = 0.2 * img + 1.0
        transparent = np.clip(transparent, 0, 255).astype(np.uint8)
        dst = cv2.add(transparent, src_crop)
        return dst
        #self.canvas_pred.itemconfig(self.image_on_canvas_pred, image=self.change_tk_format(dst))

    # def on_motioned(self, event):
    #     sx = event.x
    #     sy = event.y
    #
    #     if not self.seg_mode_flag and self.flu_dataset_names:
    #         if sx < 512 and sy < 512:
    #             if self.labelImg[sy, sx] >= 1:
    #                 # self.hist_gui_flagで細胞内で連続でグラフが描画されないようにする
    #                 if self.hist_gui_flag:
    #                     # 初期化
    #                     self.hist_mcherry = np.array([])
    #                     self.hist_yfp = np.array([])
    #
    #                     for item in self.hist_label:
    #                         if self.labelImg[sy, sx] == item[0]:
    #                             #print(item[0])
    #                             # 該当座標の輝度値を格納
    #                             if self.mCherry_flag:
    #                                 #print(self.mcherrys[self.image_number][item[2], item[1]])
    #                                 self.hist_mcherry = np.append(self.mcherrys[self.image_number][item[2], item[1]],
    #                                                               self.hist_mcherry)
    #
    #                             if self.YFP_flag:
    #                                 #print(self.yfps[self.image_number][item[2], item[1]])
    #                                 self.hist_yfp = np.append(self.yfps[self.image_number][item[2], item[1]],
    #                                                           self.hist_yfp)
    #                     #print("pre_mc:" + str(self.hist_mcherry))
    #                     #print("pre_yfp:" + str(self.hist_yfp))
    #
    #                     tone = int(self.tone_num.get())
    #                     r = int(256 / tone)
    #
    #                     if self.mCherry_flag:
    #                         for i in range(r):
    #                             r1 = i * tone
    #                             r2 = (i+1) * tone
    #                             #print(r2)
    #                             self.hist_mcherry = np.where((r1 <= self.hist_mcherry) & (self.hist_mcherry < r2),
    #                                                          i, self.hist_mcherry)
    #                         self.hist_mcherry *= tone
    #                         #print("mc:" + str(self.hist_mcherry))
    #
    #                     if self.YFP_flag:
    #                         for i in range(r):
    #                             r1 = i * tone
    #                             r2 = (i + 1) * tone
    #                             self.hist_yfp = np.where((r1 <= self.hist_yfp) & (self.hist_yfp < r2),
    #                                                      i, self.hist_yfp)
    #                         self.hist_yfp *= tone
    #                         #print("yfp:" + str(self.hist_yfp))
    #
    #
    #
    #                     # ヒストグラムの取得
    #                     if self.mCherry_flag:
    #                         mc_hist, mc_bins = np.histogram(self.hist_mcherry, bins=np.arange(0, 255))
    #
    #                     if self.YFP_flag:
    #                         yfp_hist, yfp_bins = np.histogram(self.hist_yfp, bins=np.arange(0, 255))
    #
    #
    #                     # ヒストグラムの表示
    #                     if self.mCherry_flag:
    #                         plt.title("mCherry")
    #                         plt.xlabel('Brightness value')
    #                         plt.ylabel('Frequency')
    #                         plt.xlim(self.hist_bri_lower, self.hist_bri_upper)
    #                         plt.ylim(0, 30)
    #                         plt.plot(mc_hist)
    #                         plt.show()
    #                     if self.YFP_flag:
    #                         plt.title("YFP")
    #                         plt.xlabel('Brightness value')
    #                         plt.ylabel('Frequency')
    #                         plt.xlim(self.hist_bri_lower, self.hist_bri_upper)
    #                         plt.ylim(0, 30)
    #                         plt.plot(yfp_hist)
    #                         plt.show()
    #
    #                     self.hist_gui_flag = False  # １度表示したらグラフ描画しない
    #
    #             else:
    #                 self.hist_gui_flag = True  # 画面外に出たらグラフ描画できるようにする

    def on_pressed(self, event):
        self.sx = event.x
        self.sy = event.y
        if self.seg_mode_flag:
            if self.penname.get() == "Split":
                pencolor = "black"
            elif self.penname.get() == "Join":
                pencolor = "white"

            self.oval = self.canvas_pred.create_oval(self.sx, self.sy, event.x, event.y, outline=pencolor, width=1)

            if self.penname.get() == "Split":
                if self.penwidth.get() == 1:
                    self.hands[self.image_number][self.sy][self.sx] = 0

                if self.penwidth.get() == 3:
                    self.hands[self.image_number][self.sy - 1:self.sy + 2, self.sx - 1:self.sx + 2] = 0

                if self.penwidth.get() == 5:
                    self.hands[self.image_number][self.sy - 2:self.sy + 3, self.sx - 2:self.sx + 3] = 0

            elif self.penname.get() == "Join":
                if self.penwidth.get() == 1:
                    self.hands[self.image_number][self.sy][self.sx] = 255

                if self.penwidth.get() == 3:
                    self.hands[self.image_number][self.sy - 1:self.sy + 2, self.sx - 1:self.sx + 2] = 255

                if self.penwidth.get() == 5:
                    self.hands[self.image_number][self.sy - 2:self.sy + 3, self.sx - 2:self.sx + 3] = 255

            if self.seg_mode_flag and 25 <= self.sy < self.height_pred - 25 and 25 <= self.sx < self.width_pred - 25:
                self.ImageZoom()

        else:
            if self.flu_dataset_names:
                if self.sx < 512 and self.sy < 512:
                    if self.labelImg[self.sy, self.sx] >= 1:
                        # self.hist_gui_flagで細胞内で連続でグラフが描画されないようにする
                        # if self.hist_gui_flag:
                        # 初期化
                        self.hist_mcherry = np.array([])
                        self.hist_yfp = np.array([])

                        for item in self.hist_label:
                            if self.labelImg[self.sy, self.sx] == item[0]:
                                # print(item[0])
                                # 該当座標の輝度値を格納
                                if self.mCherry_flag:
                                    # print(self.mcherrys[self.image_number][item[2], item[1]])
                                    self.hist_mcherry = np.append(self.mcherrys[self.image_number][item[2], item[1]],
                                                                  self.hist_mcherry)

                                if self.YFP_flag:
                                    # print(self.yfps[self.image_number][item[2], item[1]])
                                    self.hist_yfp = np.append(self.yfps[self.image_number][item[2], item[1]],
                                                              self.hist_yfp)
                        # print("pre_mc:" + str(self.hist_mcherry))
                        # print("pre_yfp:" + str(self.hist_yfp))

                        tone = int(self.tone_num.get())
                        r = int(256 / tone)

                        if self.mCherry_flag:
                            for i in range(r):
                                r1 = i * tone
                                r2 = (i + 1) * tone
                                # print(r2)
                                self.hist_mcherry = np.where((r1 <= self.hist_mcherry) & (self.hist_mcherry < r2),
                                                             i, self.hist_mcherry)
                            self.hist_mcherry *= tone
                            # print("mc:" + str(self.hist_mcherry))

                        if self.YFP_flag:
                            for i in range(r):
                                r1 = i * tone
                                r2 = (i + 1) * tone
                                self.hist_yfp = np.where((r1 <= self.hist_yfp) & (self.hist_yfp < r2),
                                                         i, self.hist_yfp)
                            self.hist_yfp *= tone
                            # print("yfp:" + str(self.hist_yfp))

                        # ヒストグラムの取得
                        if self.mCherry_flag:
                            mc_hist, mc_bins = np.histogram(self.hist_mcherry, bins=np.arange(0, 255))

                        if self.YFP_flag:
                            yfp_hist, yfp_bins = np.histogram(self.hist_yfp, bins=np.arange(0, 255))

                        # ヒストグラムの表示
                        if self.mCherry_flag:
                            plt.title("mCherry")
                            plt.xlabel('Brightness value')
                            plt.ylabel('Frequency')
                            plt.xlim(self.hist_bri_lower, self.hist_bri_upper)
                            plt.ylim(0, 30)
                            plt.plot(mc_hist)
                            plt.show()
                        if self.YFP_flag:
                            plt.title("YFP")
                            plt.xlabel('Brightness value')
                            plt.ylabel('Frequency')
                            plt.xlim(self.hist_bri_lower, self.hist_bri_upper)
                            plt.ylim(0, 30)
                            plt.plot(yfp_hist)
                            plt.show()

                        # self.hist_gui_flag = False  # １度表示したらグラフ描画しない
                    #
                    # else:
                    #     self.hist_gui_flag = True  # 画面外に出たらグラフ描画できるようにする

    def on_dragged(self, event):
        if self.seg_mode_flag:
            if self.penname.get() == "Split":
                pencolor = "black"
            elif self.penname.get() == "Join":
                pencolor = "white"
            else:
                return

            self.line = self.canvas_pred.create_line(self.sx, self.sy, event.x, event.y,
                                                     fill=pencolor, width=self.penwidth.get())

            self.sx = event.x
            self.sy = event.y

            if self.penname.get() == "Split":
                if self.penwidth.get() == 1:
                    self.hands[self.image_number][self.sy][self.sx] = 0

                if self.penwidth.get() == 3:
                    self.hands[self.image_number][self.sy - 1:self.sy + 2, self.sx - 1:self.sx + 2] = 0

                if self.penwidth.get() == 5:
                    self.hands[self.image_number][self.sy - 2:self.sy + 3, self.sx - 2:self.sx + 3] = 0

            elif self.penname.get() == "Join":
                if self.penwidth.get() == 1:
                    self.hands[self.image_number][self.sy][self.sx] = 255

                if self.penwidth.get() == 3:
                    self.hands[self.image_number][self.sy - 1:self.sy + 2, self.sx - 1:self.sx + 2] = 255
                    # self.hands[self.image_number][self.sy - 1][self.sx - 1] = 255
                    # self.hands[self.image_number][self.sy - 1][self.sx] = 255
                    # self.hands[self.image_number][self.sy][self.sx - 1] = 255
                    # self.hands[self.image_number][self.sy - 1][self.sx + 1] = 255
                    # self.hands[self.image_number][self.sy][self.sx + 1] = 255
                    # self.hands[self.image_number][self.sy + 1][self.sx + 1] = 255
                    # self.hands[self.image_number][self.sy + 1][self.sx] = 255
                    # self.hands[self.image_number][self.sy + 1][self.sx - 1] = 255

                if self.penwidth.get() == 5:
                    self.hands[self.image_number][self.sy - 2:self.sy + 3, self.sx - 2:self.sx + 3] = 255


            if self.seg_mode_flag and 25 <= self.sy < self.height_pred - 25 and 25 <= self.sx < self.width_pred - 25:
                self.ImageZoom()

    def on_released(self, event):
        self.canvas_src.itemconfig(self.image_on_canvas_src, image=self.imgs_tk[self.image_number])

    def erase(self, event):
        self.canvas_pred.delete("all")
        self.hands[self.image_number] = copy.deepcopy(self.segments[self.image_number][0])

        # 画像の表示
        if self.seg_mode_flag:
            tk = self.segments_tk[self.image_number]
        else:
            tk = self.dsts_tk[self.image_number]

        self.image_on_canvas_pred = self.canvas_pred.create_image(0, 0, anchor=NW, image=tk)
        self.canvas_pred.itemconfig(self.image_on_canvas_pred, image=tk)

    def DisplayUpdate(self, tracking):
        self.tracking_flag = tracking
        self.labeling(self.segments[self.image_number][0])
        self.LogLabelChange(self.image_number, self.label_correction_list)

        # 表示画像を更新
        self.canvas_src.itemconfig(self.image_on_canvas_src, image=self.imgs_tk[self.image_number])
        if not self.seg_mode_flag:
            self.canvas_pred.itemconfig(self.image_on_canvas_pred, image=self.dsts_tk[self.image_number])
        else:
            self.canvas_pred.itemconfig(self.image_on_canvas_pred, image=self.segments_tk[self.image_number])

        # フレーム番号更新
        self.message_imgnum.delete(0, END)
        self.message_imgnum.insert(END, f'This image is {self.img_ids[self.image_number]}')

    # フレーム移動
    def onBackButton(self):
        # 最後の画像に戻る
        if self.image_number == 0:
            self.image_number = self.img_totalnum - 1
        else:
            # 一つ戻る
            self.image_number -= 1

        # 表示画像を更新
        self.DisplayUpdate(tracking=True)

    def onBack5Button(self):
        # 最後の画像に戻る
        if self.image_number < 5:
            self.image_number = self.img_totalnum - 1
        else:
            # 一つ戻る
            self.image_number -= 5

        # 表示画像を更新
        self.DisplayUpdate(tracking=True)

    def onFirstButton(self):
        # 最初の画像に戻る
        self.image_number = 0

        # 表示画像を更新
        self.DisplayUpdate(tracking=False)

    def onNextButton(self):
        if self.image_number == self.img_totalnum - 1:
            self.image_number = 0
        else:
            self.image_number += 1

        # 表示画像を更新
        self.DisplayUpdate(tracking=True)

    def onNext5Button(self):
        # 最初の画像に戻る
        if self.image_number + 5 > self.img_totalnum - 1:
            self.image_number = 0

        else:
            # 5つ進む
            self.image_number += 5

        # 表示画像を更新
        self.DisplayUpdate(tracking=True)

    def onLastButton(self):
        # 最後の画像にとぶ
        self.image_number = self.img_totalnum - 1

        # 表示画像を更新
        self.DisplayUpdate(tracking=False)

    def SliderHistBrightLower(self, args):
        self.hist_bri_lower = int(self.hist_bri_lowersl.get())

    def SliderHistFrequencyRangeflu1(self, args):
        self.hist_freq_range_flu1 = int(self.hist_freq_range_flu1sl.get())

    def SliderHistFrequencyRangeflu2(self, args):
        self.hist_freq_range_flu2 = int(self.hist_freq_range_flu2sl.get())

    # def SliderHistFrequencyRangeflu3(self, args):
    #     self.hist_freq_range_flu3 = int(self.hist_freq_range_flu3sl.get())

    def LabelChangeButton(self):
        if self.bEnt.get() == "" or not self.bEnt.get().isdecimal():
            # self.systext.insert("1.0", "Input cell label changing before number\n")
            return
        if self.aEnt.get() == "" or not self.aEnt.get().isdecimal():
            # self.systext.insert("1.0", "Input cell label changing after number\n")
            return

        self.label_bnum = int(self.bEnt.get())  # before label number
        self.label_anum = int(self.aEnt.get())  # after label number

        self.labelchange_register_flag = True
        self.LabelChange(self.label_bnum, self.label_anum)

    def LogLabelChange(self, image_number, label_correction_list):
        # print("LogLabelChange!!")
        if label_correction_list:
            for item in label_correction_list:
                if item[0] == image_number:
                    self.LabelChange(item[1], item[2])

    def LabelChange(self, bef_label, aft_label):
        if bef_label - 1 not in self.cell_truck_labellist:
            self.labelchange_register_flag = False
            return

        # 変更後のラベルが重複している場合、重複していると注意表示し、操作を止める
        if aft_label - 1 in self.cell_truck_labellist:
            self.labelchange_register_flag = False
            return

        if self.labelchange_register_flag:
            # ラベル変更リストに追加
            self.label_correction_list.append([self.image_number, bef_label, aft_label])

            # csvファイルの該当箇所に変更したリストを保存
            self.df = pd.read_csv(f'{self.dataset_path}/PreData.csv')
            self.df.iat[self.df_pidx, 1] = self.label_correction_list
            self.df.to_csv(f'{self.dataset_path}/PreData.csv', index=False)
            self.labelchange_register_flag = False

        self.pre_update_labellist_num = self.cell_truck_labellist.index(bef_label - 1)  # 詳細ラベルリスト内のラベルの順番を取得
        self.cell_truck_labellist[self.pre_update_labellist_num] = aft_label - 1
        self.cell_truck_detaillist[self.pre_update_labellist_num][0] = aft_label - 1
        if self.image_number in self.labelchangeframenum:
            idx = self.labelchangeframenum.index(self.image_number)
            self.labelchange[idx][1] = self.cell_truck_labellist
            self.labelchange[idx][2] = self.cell_truck_detaillist
        else:
            self.labelchangeframenum.append(self.image_number)
            self.labelchange.append([self.image_number, self.cell_truck_labellist, self.cell_truck_detaillist])

        # ラベル変更処理を施したうえで再度同じ画像をラベリング処理
        self.labeling(self.segments[self.image_number][0])

        if self.gui_change_flag:
            self.canvas_pred.itemconfig(self.image_on_canvas_pred, image=self.dsts_tk[self.image_number])

    def DataSaveButton(self):
        self.data_save_only_flag = True
        # global pre_cell_detaillist, pre_cell_labelnum
        self.cx = []
        self.cy = []
        pre_cell_trucklist = []  # 1つ前のフレームの細胞番号のリスト、ベクトルを求めるのに用いる
        pre_cell_detaillist = []
        # self.dotcolor = []
        # self.graphlabellist = []  # グラフにプロットする細胞のリスト

        if self.sEnt.get() == "" or not self.sEnt.get().isdecimal():
            # self.systext.insert("1.0", "Input image start number\n")
            return
        self.m_snum = int(self.sEnt.get())
        if self.fEnt.get() == "" or not self.fEnt.get().isdecimal():
            # self.systext.insert("1.0", "Input image last number\n")
            return
        self.m_fnum = int(self.fEnt.get())

        if 0 < self.m_snum < self.m_fnum <= self.img_totalnum and self.m_fnum > 0:
            self.cellcsvpath = f'{self.dataset_path}/{self.dataset_name}_{self.m_snum}-{self.m_fnum}' \
                               f'_bl{self.hist_bri_lower}_tone{int(self.tone_num.get())}_AnalysisData.csv'
            self.cellcsvfile = open(self.cellcsvpath, 'w')
            # header を設定
            headernames = ["Fluorescence No.", "Fluorescence Time[d:h:m:s]", "Cell No.", "Parent No.",
                           "Division Time[h:m:s]", "Cell division flag", "Generation", "Gx[pix]", "Gy[pix]",
                           "I1", "I2", "I3", "I4", "I5", "I6"]
            writer = csv.DictWriter(self.cellcsvfile, fieldnames=headernames)
            writer.writeheader()

            self.tracking_flag = False
            self.gui_change_flag = False
            for imgcount in range(self.m_snum - 1, self.m_fnum):
                self.image_number = imgcount
                self.labeling(self.segments[imgcount][0])

                self.LogLabelChange(self.image_number, self.label_correction_list)
                if self.image_number in self.labelchangeframenum:
                    idx = self.labelchangeframenum.index(self.image_number)
                    self.cell_truck_labellist = self.labelchange[idx][1]
                    self.cell_truck_detaillist = self.labelchange[idx][2]

                # for label in range(self.n):
                #     # 背景は抜く
                #     label += 1
                #     for item in self.hist_label:
                #         if label == item[0]:
                #             # 該当座標の輝度値を格納
                #             if self.mCherry_flag:
                #                 self.hist_mcherry = np.append(self.mcherrys[self.image_number][item[2], item[1]],
                #                                               self.hist_mcherry)
                #             if self.YFP_flag:
                #                 self.hist_yfp = np.append(self.yfps[self.image_number][item[2], item[1]],
                #                                           self.hist_yfp)

                self.tracking_flag = True

                self.renum_cell_truck_detaillist = sorted(self.cell_truck_detaillist, key=lambda x: x[0])
                self.renum_cell_truck_labellist = sorted(self.cell_truck_labellist)

                for item in self.renum_cell_truck_detaillist:
                    cell_vector_flag = True
                    writer = csv.writer(self.cellcsvfile, lineterminator='\n')
                    # self.graphlabellist.append(item[0])
                    self.cx.append(float(item[2]))
                    self.cy.append(float(item[3]))

                    # ベクトルの計算
                    # print(len(pre_cell_trucklist), pre_cell_trucklist)
                    # print(len(pre_cell_detaillist), pre_cell_detaillist)
                    # if item[0] in pre_cell_trucklist:
                    # pre_cell_labelnum = pre_cell_trucklist.index(item[0])
                    # フレーム間の同じラベルの細胞の距離r
                    # r = math.sqrt(
                    #    math.pow(math.fabs(item[2] - pre_cell_detaillist[pre_cell_labelnum][2]), 2) +
                    #    math.pow(math.fabs(item[3] - pre_cell_detaillist[pre_cell_labelnum][3]), 2))
                    # 距離が長すぎるのは前フレームの細胞が消失して、新しい細胞がそのラベルで割り当てられる可能性が
                    # 考えられるため、距離が短いものだけベクトル表示する
                    # if r < 30:
                    #    cell_vector_flag = False
                    #    cell_vector_x = item[2] - pre_cell_detaillist[pre_cell_labelnum][2]
                    #    cell_vector_y = item[3] - pre_cell_detaillist[pre_cell_labelnum][3]

                    # csvファイルに書き込む要素(蛍光番号,蛍光時刻,細胞番号,親番号,出生時刻,年齢,分裂回数,世代,重心x,重心y,蛍光強度1~6
                    # 新しい細胞の場合ベクトルが出ないので
                    # if cell_vector_flag:
                    if item[12] != 0:
                        if isinstance(item[12], datetime.time):
                            item[12] = item[12].strftime('%H:%M:%S')
                        else:
                            item[12] = item[12].strftime('%d.%H:%M:%S')

                    writer.writerow(
                        ["%03d" % (imgcount + 1), self.fts[self.image_number].strftime('%Y-%m-%d %H:%M:%S'),
                         "%03d" % (item[0] + 1), "%03d" % item[10],
                         item[12], item[13], item[14], round(item[2], 2), round(item[3], 2),
                         round(item[15], 2), round(item[16], 2), round(item[17], 2), round(item[18], 2),
                         round(item[19], 2), round(item[20], 2)])
                    # else:
                    #     writer.writerow(
                    #         [imgcount, item[0] + 1, round(item[2], 2), round(item[3], 2), item[7], round(item[6]),
                    #          round(cell_vector_x, 2), round(cell_vector_y, 2), item[10],
                    #          imgcount * self.shoot_interval * 15])

                print(imgcount)

        self.gui_change_flag = True
        self.data_save_only_flag = False
        self.cellcsvfile.close()
        self.canvas_pred.itemconfig(self.image_on_canvas_pred, image=self.dsts_tk[self.image_number])

    def TextButton(self):
        if not self.seg_mode_flag:
            if self.textflag:
                self.textflag = False
            else:
                self.textflag = True

            self.labeling(self.segments[self.image_number][0])
            self.canvas_pred.itemconfig(self.image_on_canvas_pred, image=self.dsts_tk[self.image_number])

    def TruckingModeButton(self):
        self.seg_mode_flag = False

        self.labeling(self.segments[self.image_number][0])

        # if self.imagelab[self.image_number][1] != 1:
        #     self.imagelab[self.image_number][0] = self.dst
        #     cv2.imwrite(self.dircommonpath + "/" + self.labpath + "/" +
        #                 self.filenameMaker(self.labprocess, 0, self.image_number), self.ldst)

        self.canvas_pred.itemconfig(self.image_on_canvas_pred, image=self.dsts_tk[self.image_number])

    def SegmentationModeButton(self):
        self.seg_mode_flag = True

        # self.seg = self.segments[self.image_number]
        #
        # # 画像の表示
        # self.img_srgb = cv2.cvtColor(self.seg, cv2.COLOR_BGR2RGB)
        # self.img_spil = Image.fromarray(self.img_srgb)
        # self.img_stk = ImageTk.PhotoImage(self.img_spil)

        self.canvas_pred.itemconfig(self.image_on_canvas_pred, image=self.segments_tk[self.image_number])

    def SegmentationImageSaveButton(self):
        if self.seg_mode_flag == 1:
            self.segments[self.image_number][0] = copy.deepcopy(self.hands[self.image_number])
            self.segments_tk[self.image_number] = self.change_tk_format(self.hands[self.image_number], True)
            self.segments[self.image_number][1] = 1
            # self.img_fs = cv2.cvtColor(self.segments[self.image_number][0], cv2.COLOR_GRAY2BGR)
            cv2.imwrite(f'{self.dataset_path}/Hands/{self.img_ids[self.image_number]}_hand.tif',
                        self.segments[self.image_number][0])
            self.canvas_pred.itemconfig(self.image_on_canvas_pred, image=self.segments_tk[self.image_number])

    def SegmentationImageDeleteButton(self):
        if self.sEnt.get() == "" or not self.sEnt.get().isdecimal():
            # self.systext.insert("1.0", "Input fiber label changing before number\n")
            return
        if self.fEnt.get() == "" or not self.fEnt.get().isdecimal():
            # self.systext.insert("1.0", "Input fiber label changing after number\n")
            return
        self.label_snum = int(self.sEnt.get())
        self.label_fnum = int(self.fEnt.get())

        for i in range(self.label_snum - 1, self.label_fnum):
            if self.segments[i][1] == 1:
                self.segments[i][0] = copy.deepcopy(self.predicts[i])
                self.hands[i] = copy.deepcopy(self.predicts[i])
                self.segments[i][1] = 0
                self.segments_tk[i] = self.change_tk_format(self.predicts[i], True)

        if self.label_snum - 1 <= self.image_number < self.label_fnum:
            if self.seg_mode_flag:
                self.canvas_pred.itemconfig(self.image_on_canvas_pred, image=self.segments_tk[self.image_number])

            else:
                self.labeling(self.segments[self.image_number][0])
                self.image_on_canvas_pred = self.canvas_pred.create_image(0, 0, anchor=NW,
                                                                          image=self.dsts_tk[self.image_number])

    def SegmentationImageDownloadButton(self):
        if self.sEnt.get() == "" or not self.sEnt.get().isdecimal():
            # self.systext.insert("1.0", "Input fiber label changing before number\n")
            return
        if self.fEnt.get() == "" or not self.fEnt.get().isdecimal():
            # self.systext.insert("1.0", "Input fiber label changing after number\n")
            return
        self.label_snum = int(self.sEnt.get())
        self.label_fnum = int(self.fEnt.get())
        for i in range(self.label_snum - 1, self.label_fnum):
            # もしセグメント画像ファイルに該当番号の画像があるなら画像をグレイに直してセグメンと画像バッファに更新
            if os.path.exists(f'{self.dataset_path}/Hands/{self.img_ids[i]}_hand.tif'):
                segd = cv2.imread(f'{self.dataset_path}/Hands/{self.img_ids[i]}_hand.tif', 0)
                _, segd = cv2.threshold(segd, 128, 255, cv2.THRESH_BINARY)
                self.segments[i][0] = segd
                self.hands[i] = segd
                self.segments_tk[i] = self.change_tk_format(segd, True)
                self.segments[i][1] = 1

        # 画像の表示
        if self.seg_mode_flag:
            self.canvas_pred.itemconfig(self.image_on_canvas_pred, image=self.segments_tk[self.image_number])

        else:
            self.labeling(self.hands[self.image_number])
            self.image_on_canvas_pred = self.canvas_pred.create_image(0, 0, anchor=NW,
                                                                      image=self.dsts_tk[self.image_number])

    def HistButton(self):
        # 初期化
        self.hist_mcherry = []
        self.hist_yfp = []

        for label in range(self.n):
            # 背景は抜く
            label += 1
            for item in self.hist_label:
                if label == item[0]:
                    # 該当座標の輝度値を格納
                    self.hist_mcherry.append(self.mcherrys[self.image_number][item[2], item[1]])
                    self.hist_yfp.append(self.yfps[self.image_number][item[2], item[1]])

            # ヒストグラムの取得
            mc_hist, mc_bins = np.histogram(np.array(self.hist_mcherry), bins=np.arange(256 + 1))
            yfp_hist, yfp_bins = np.histogram(np.array(self.hist_yfp), bins=np.arange(256 + 1))

            print(mc_bins)
            print(yfp_bins)

            # ヒストグラムの表示
            plt.plot(mc_hist)
            plt.show()
            plt.plot(yfp_hist)
            plt.show()

    def ImageSaveButton(self):
        # make marker
        k = 0.7
        res = np.zeros(self.segments[self.image_number][0].shape, dtype=np.uint8)
        n, img = cv2.connectedComponents(self.segments[self.image_number][0], connectivity=4)
        # blob = cv2.connectedComponentsWithStats(self.segments[self.image_number][0], connectivity=4)
        # n = blob[0]
        # img = copy.deepcopy(blob[1])
        # data = np.delete(blob[2], 0, 0)
        # print(data[:, 4])

        # deal with each cell
        for label in np.unique(img):
            if label == 0:  # ignore the background
                continue
            hand = ((label == img) * 1).astype(np.uint8)  # 細胞1個だけにする
            _, hand = cv2.threshold(hand, 0, 255, cv2.THRESH_BINARY)

            # find the maximal disk
            dist = cv2.distanceTransform(hand, cv2.DIST_L2, 3)  # 距離変換
            dmax = cv2.minMaxLoc(dist)[1]  # 類似度の最も高い値を取る（セルマスクに含まれる最大の円盤の直径）
            # compute the dSE
            dSE = int((1 - k) * dmax)
            #print(dSE)
            if dSE == 0:
                continue
            # get a circle structure and erode
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dSE, dSE))  # dSEの大きさの楕円カーネルを形成
            hand = cv2.erode(hand, kernel)  # 細胞を収縮

            # select only the largest connected component(because the erotion will split(erode) some cell to multible component)
            # erodionで細胞が分割される可能性があるのでラベリングして面積最大のものだけ残す
            nb, cc = cv2.connectedComponents(hand, connectivity=4)
            if nb != 1:
                max_size = 0
                max_cc = 0
                for i in range(1, nb):
                    component = (cc == i) * 1  # ラベリング画像からiラベルのpixelだけ配列に入れる
                    if (np.sum(component)) > max_size:  # 最大面積ならmax~に代入
                        max_size = np.sum(component)
                        max_cc = i
                hand = ((max_cc == cc) * 1).astype(np.uint8)  # 面積最大のラベルだけにする

            # add each eroded cell to the final image
            res = cv2.add(res, hand)
        # transfrom to binary image
        _, marker = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY)

        # hand & marker img save
        cv2.imwrite(f'../cell-detction-using-u-net-framework-master/oki_cell/train/images/'
                    f'#39_{self.img_ids[self.image_number]}.tif', self.imgs[self.image_number])
        cv2.imwrite(f'../cell-detction-using-u-net-framework-master/oki_cell/train/masks_new/'
                    f'#39_{self.img_ids[self.image_number]}_mask.tif', self.segments[self.image_number][0])
        cv2.imwrite(f'../cell-detction-using-u-net-framework-master/oki_cell/train/markers/'
                    f'#39_{self.img_ids[self.image_number]}_marker.tif', marker)
