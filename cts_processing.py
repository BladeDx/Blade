# 自動セグメンテーション処理の追加
# 5枚飛ばししてからnextボタンを押すとinitlabel=0でラべリングを行ってしまうバグ(initlabel=1になってくれるのが理想，初期状態のラべリング処理だと思われていた部分のinitlabel=0が原因だと考えられる)の解消
# tkinter * opencvを実装，ファイルをGUIタブ上で開く設定の追加，保存の形式設定，label関数の処理を早くする
import collections
from tkinter import *
from PIL import Image, ImageTk
import cv2
import numpy as np
import copy
import math
import datetime


# ----------------------------------------------------------------------

class cts_processing():
    # ラべリング処理
    def labeling(self, bin):
        dst = np.zeros((self.height_pred, self.width_pred, 3), np.uint8)
        I1 = 0
        I2 = 0
        I3 = 0
        I4 = 0
        I5 = 0
        I6 = 0

        # labeling
        self.nLabels, self.labelImg = cv2.connectedComponents(bin, connectivity=4)  # ラべル情報が乗っている2次元配列の取得
        self.blabel = cv2.connectedComponentsWithStats(bin, connectivity=4)  # ブロブ解析による各オブジェクトの詳細情報の取得

        # ブロブ解析
        self.n = self.blabel[0] - 1
        self.data = np.delete(self.blabel[2], 0, 0)
        self.center = np.delete(self.blabel[3], 0, 0)
        self.cell_existence = []
        for i in range(self.n):
            self.cell_existence.append(1)

        """
        print(u"ブロブの個数:", n)
        print(u"各ブロブの外接矩形の左上x座標", data[:, 0])
        print(u"各ブロブの外接矩形の左上y座標", data[:, 1])
        print(u"各ブロブの外接矩形の幅", data[:, 2])
        print(u"各ブロブの外接矩形の高さ", data[:, 3])
        print(u"各ブロブの面積", data[:, 4])
        print(u"各ブロブの中心座標:\n", center)
        """

        # 面積が小さい細胞核＋ゴミの消去
        for i in range(self.nLabels - 1):
            self.cell_existence[i], self.n = self.area_filter(self.n, self.cell_existence[i], self.data[i, 4],
                                                              self.cleanth)

        # ヒストグラム作成のための輝度情報を入手
        # 生死フラグに背景分の要素を追加する必要があるため，リストの最前列に0を追加+rastalabellistをlabellistに合わせるため+1する
        fe = self.cell_existence.copy()
        fe.insert(0, 0)
        self.hist_label = []
        for y in range(self.height_pred):
            for x in range(self.width_pred):
                # 生きている(cell_existence = 1)細胞のラベルのみランダムで配色，
                if fe[self.labelImg[y, x]] == 1:
                    # ヒストグラムに使う配列に格納, labelImgはラスタスキャンのラベル画像のため第1カラムはラスタスキャンのラベルであることに注意
                    self.hist_label.append([self.labelImg[y, x], x, y])

        # labelImgをリスト型に(周囲を-1で囲み周囲長を求める時のindex error対策を行っている)
        labelImg_list = self.labelImg.tolist()
        minusone_list = []
        for i in range(self.width_pred + 2):
            minusone_list.append(-1)
        for j in range(self.height_pred):
            labelImg_list[j].insert(0, -1)
            labelImg_list[j].append(-1)
        labelImg_list.insert(0, minusone_list)
        labelImg_list.append(minusone_list)

        # 各ラベルの周囲長を求める
        self.perimeter_list = []  # 背景周囲長は含まない
        self.area_around_list = []  # 周囲の面積数のリスト
        for region in range(1, self.nLabels):  # 背景ラベル(0)は考えない
            around, area_around = self.btrace(region, labelImg_list)
            self.perimeter_list.append(around)
            self.area_around_list.append(area_around)

        # 円形度
        self.circularity_list = []
        for i in range(self.nLabels - 1):
            self.cell_existence[i], self.n, circularity = self.circularity_filter(i, self.n, self.cell_existence[i],
                                                                                  self.perimeter_list[i],
                                                                                  self.data[i, 4],
                                                                                  self.circuth,
                                                                                  self.area_around_list[i])

            self.circularity_list.append(circularity)

        # 2回目以降の処理
        if self.tracking_flag:
            self.f_list = []
            self.cell_rasta_labellist = []  # 追跡しているラベルの元のラベルリスト（ラスタスキャンで振られた）リスト，list内にラべルがあるかの判定で1次リストを使用
            self.pre_cell_truck_detaillist = copy.deepcopy(self.cell_truck_detaillist)

            # 細胞の前後処理(追跡リストの更新)
            for i in range(len(self.pre_cell_truck_detaillist)):
                for j in range(self.nLabels - 1):
                    if self.cell_existence[j] == 0: continue
                    self.center_distance = math.sqrt(
                        math.pow(math.fabs(self.pre_cell_truck_detaillist[i][2] - self.center[j][0]), 2) + math.pow(
                            math.fabs(self.pre_cell_truck_detaillist[i][3] - self.center[j][1]), 2))
                    # 対象細胞のr近傍が1000以内の細胞を候補細胞とする
                    if self.center_distance < self.r:
                        # 候補細胞のf値をfminを導出するためのlist、f_listに追加しておく
                        self.f = self.wc * self.center_distance + self.wp * \
                                 math.fabs(self.pre_cell_truck_detaillist[i][6] - self.perimeter_list[j]) + self.wa * \
                                 math.fabs(self.pre_cell_truck_detaillist[i][7] - self.data[j, 4])

                        self.f_list.append(
                            [self.pre_cell_truck_detaillist[i][0], j, self.f, self.pre_cell_truck_detaillist[i][9]])

            # print("self.f_list " + str(self.f_list))

            # アニーリング処理（fをregずつあげて候補ラベルの中から１つに絞っていく）
            self.aperture_f = self.reg
            #
            self.fcp = copy.deepcopy(self.f_list)
            # self.f_list.sort(key=lambda x: x[1])
            # self.fcp.sort(key=lambda y: y[1])
            self.fmin_list = []
            self.outf_list = []
            self.fmin_object_labelnum_list = []
            self.fmin_candidate_labelnum_list = []
            self.cell_existence_list = []  # 候補細胞が周囲にない対象ラベル（前画像では生きているが、現画像では死んでしまっている細胞）を消すためのリスト

            while len(self.fcp) != 0:
                for self.fcp_item in self.fcp[:]:
                    # fminとなる細胞ラベルの前後関係が決まった時の後の処理だからfmin決まってからでないとそのままでは使えない
                    # 消失した細胞の近くに生きている細胞がある状況で，ラベルが重なることを防ぐ
                    if self.aperture_f > self.fcp_item[2]:
                        # 対象細胞、候補細胞が重複する場合、そのFリスト要素は削除。
                        # ただし、ラベル修正によって変わった対象細胞のラベルが除かれないようにする必要があるまたラベル修正後の細胞核が消滅した場合の追跡ラベルリストの削除処理も考える必要がある
                        if (not (self.fcp_item[0] in self.fmin_object_labelnum_list) and
                            not (self.fcp_item[1] in self.fmin_candidate_labelnum_list)) or \
                                self.fcp_item[3] == 1:  # and \1
                            # self.fcp_item[0] in self.label_correction_list:#これではほかの候補もfminlistに入り込むためダメ
                            # self.fmin = self.fcp_item
                            self.fmin_list.append(self.fcp_item)
                            self.fmin_object_labelnum_list.append(self.fcp_item[0])
                            self.fmin_candidate_labelnum_list.append(self.fcp_item[1])
                            # for self.fcp_item in self.fcp:
                            #    if self.fcp_item[0] == i:

                        else:
                            self.outf_list.append(self.fcp_item)

                        self.fcp.remove(self.fcp_item)
                self.aperture_f += self.reg

            # print("self.fmin_list " + str(self.fmin_list))
            # print("outside self.fmin_list " + str(self.outf_list))
            for i in range(len(self.fmin_list)):

                # データ保存のときのみ行う処理
                if self.fluorescent_flag and self.data_save_only_flag:
                    # 蛍光輝度の計算
                    I1 = self.fluorescence_intensity(self.fmin_list[i][1] + 1)

                # if self.aperture_f > self.fcp_item[2] and not (self.fcp_item[1] in self.cell_rasta_labellist):
                self.cell_rasta_labellist.append(self.fmin_list[i][1])
                self.cell_existence_list.append([self.fmin_list[i][0], self.fmin_list[i][1]])

                # truckdetaillistの更新，truckdetaillistとtrucklabellistの順番が一緒である(?)ことから
                self.trucklabelnum = self.cell_truck_labellist.index(self.fmin_list[i][0])
                self.cell_truck_detaillist[self.trucklabelnum][1] = self.fmin_list[i][1]
                self.cell_truck_detaillist[self.trucklabelnum][2] = self.center[self.fmin_list[i][1]][0]
                self.cell_truck_detaillist[self.trucklabelnum][3] = self.center[self.fmin_list[i][1]][1]
                self.cell_truck_detaillist[self.trucklabelnum][4] = self.data[self.fmin_list[i][1], 0]
                self.cell_truck_detaillist[self.trucklabelnum][5] = self.data[self.fmin_list[i][1], 1]
                self.cell_truck_detaillist[self.trucklabelnum][6] = self.perimeter_list[self.fmin_list[i][1]]
                self.cell_truck_detaillist[self.trucklabelnum][7] = self.data[self.fmin_list[i][1], 4]
                self.cell_truck_detaillist[self.trucklabelnum][8] = self.circularity_list[self.fmin_list[i][1]]
                self.cell_truck_detaillist[self.trucklabelnum][9] = 0
                self.cell_truck_detaillist[self.trucklabelnum][10] = self.pre_cell_truck_detaillist[self.trucklabelnum][
                    10]
                self.cell_truck_detaillist[self.trucklabelnum][11] = self.pre_cell_truck_detaillist[self.trucklabelnum][
                    11]
                self.cell_truck_detaillist[self.trucklabelnum][12] = 0
                self.cell_truck_detaillist[self.trucklabelnum][13] = self.pre_cell_truck_detaillist[self.trucklabelnum][
                    13]
                self.cell_truck_detaillist[self.trucklabelnum][14] = self.pre_cell_truck_detaillist[self.trucklabelnum][
                    14]
                self.cell_truck_detaillist[self.trucklabelnum][15] = I1
                self.cell_truck_detaillist[self.trucklabelnum][16] = I2
                self.cell_truck_detaillist[self.trucklabelnum][17] = I3
                self.cell_truck_detaillist[self.trucklabelnum][18] = I4
                self.cell_truck_detaillist[self.trucklabelnum][19] = I5
                self.cell_truck_detaillist[self.trucklabelnum][20] = I6

                # self.fcp.remove(self.fcp_item)
            # print("self.cell_rasta_labellist " + str(self.cell_rasta_labellist))
            # print("self.cell_truck_labellist " + str(self.cell_truck_labellist))
            # print("self.cell_truck_detaillist " + str(self.cell_truck_detaillist))

            # 間違ってリストの中に含まれている対象、候補細胞ラベルを削除
            # fminlistの対象外となったラベルの中で、その対象細胞が追跡対象ラベルリストに含まれているかつfminlist内の対象細胞に含まれていない場合
            for self.outf_item in self.outf_list:
                if self.outf_item[0] in self.cell_truck_labellist and not self.outf_item[
                                                                              0] in self.fmin_object_labelnum_list:
                    # truckdetaillistとtrucklabellistの順番が一緒である(?)ことから
                    del (self.cell_truck_detaillist[self.cell_truck_labellist.index(self.outf_item[0])])
                    self.cell_truck_labellist.remove(self.outf_item[0])
            # print("self.cell_truck_labellist delete1" + str(self.cell_truck_labellist))
            # print("self.cell_truck_detaillist delete1" + str(self.cell_truck_detaillist))

            self.cell_truck_detaillist_01 = []
            for i in range(len(self.cell_truck_detaillist)):
                self.cell_truck_detaillist_01.append(
                    [self.cell_truck_detaillist[i][0], self.cell_truck_detaillist[i][1]])
            for self.cell_truck_detailitem_01 in self.cell_truck_detaillist_01[:]:
                if not self.cell_truck_detailitem_01 in self.cell_existence_list:
                    # del (
                    # self.cell_truck_detaillist[self.cell_truck_detaillist_01.index(self.cell_truck_detailitem_01)])
                    for self.cell_truck_detailitem in self.cell_truck_detaillist[:]:
                        if self.cell_truck_detailitem[0] == self.cell_truck_detailitem_01[0] and \
                                self.cell_truck_detailitem[1] == self.cell_truck_detailitem_01[1]:
                            self.cell_truck_detaillist.remove(self.cell_truck_detaillist[
                                                                  self.cell_truck_detaillist.index(
                                                                      self.cell_truck_detailitem)])
                            self.cell_truck_labellist.remove(self.cell_truck_detailitem_01[0])
            #         print("delete" + str(self.cell_truck_detailitem_01))
            # print("self.cell_rasta_labellist delete2" + str(self.cell_rasta_labellist))
            # print("self.cell_truck_labellist delete2" + str(self.cell_truck_labellist))
            # print("self.cell_truck_detaillist delete2" + str(self.cell_truck_detaillist))
            # print("self.pre_cell_truck_detaillist " + str(self.pre_cell_truck_detaillist))

            # 新規細胞ラべるをトラックリストに追加
            parent_cell_label = 0
            generation = 0
            for i in range(self.nLabels - 1):
                if self.cell_existence[i] == 1 and not (i in self.cell_rasta_labellist):
                    for j in range(self.n):
                        if not (j in self.cell_truck_labellist):
                            self.cell_rasta_labellist.append(i)
                            self.cell_truck_labellist.append(j)
                            # データ保存のときのみ行う処理
                            if self.fluorescent_flag and self.data_save_only_flag:
                                I1 = self.fluorescence_intensity(i + 1)
                                # 親番号、分裂回数、世代の更新
                                parent_cell_candidate_label = []
                                for k in range(len(self.pre_cell_truck_detaillist)):
                                    self.center_distance = math.sqrt(
                                        math.pow(
                                            math.fabs(self.pre_cell_truck_detaillist[k][2] - self.center[i][0]), 2)
                                        + math.pow(math.fabs(self.pre_cell_truck_detaillist[k][3] - self.center[i][1]),
                                                   2))
                                    # print("r = " + str(self.center_distance))
                                    # 対象細胞のr近傍が50以内の細胞を候補細胞とする
                                    if self.center_distance < 50:
                                        parent_cell_candidate_label.append(
                                            [self.pre_cell_truck_detaillist[k][0], self.center_distance])
                                print("parent candidate: " + str(parent_cell_candidate_label))
                                if parent_cell_candidate_label:
                                    # 細胞間距離が最小となる候補ラベルを親細胞ラベルにする
                                    parent_cell_candidate_label = np.array(parent_cell_candidate_label)
                                    parent_cell_label = int(parent_cell_candidate_label[np.argmin(
                                        parent_cell_candidate_label, axis=0)[1]][0])
                                    self.cell_truck_detaillist[self.cell_truck_labellist.index(parent_cell_label)][13] \
                                        += 1
                                    self.cell_truck_detaillist[self.cell_truck_labellist.index(parent_cell_label)][14] \
                                        += 1
                                    generation = self.cell_truck_detaillist[
                                        self.cell_truck_labellist.index(parent_cell_label)][14]

                                    # 分裂時間の計算
                                    cdt = self.fts[self.image_number] - \
                                          self.cell_truck_detaillist[
                                              self.cell_truck_labellist.index(parent_cell_label)][11]
                                    m, s = divmod(cdt.seconds, 60)
                                    h, m = divmod(m, 60)
                                    # 分裂時間の出力（day以下が欲しい情報なのでyearとmonthは適当に入れる）
                                    if cdt.days == 0:
                                        cdtdatetime = datetime.time(h, m, s)

                                    else:
                                        cdtdatetime = datetime.datetime(self.fts[0].year, self.fts[0].month, cdt.days,
                                                                        h, m, s)

                                    self.cell_truck_detaillist[self.cell_truck_labellist.index(parent_cell_label)][12] \
                                        = cdtdatetime

                                    # 親細胞も分裂したら新生細胞と捉えるため出生時刻を現フレームの蛍光時刻にする
                                    self.cell_truck_detaillist[self.cell_truck_labellist.index(parent_cell_label)][11] \
                                        = self.fts[self.image_number]
                                    # 表示されるラベルは現ラベルに+1したもの
                                    parent_cell_label += 1
                                    print("parent: " + str(parent_cell_label))

                                    print("cell division time:" + str(cdtdatetime))

                            self.cell_truck_detaillist.append([j, i, self.center[i][0], self.center[i][1],
                                                               self.data[i, 0], self.data[i, 1], self.perimeter_list[i],
                                                               self.data[i, 4], self.circularity_list[i], 0,
                                                               parent_cell_label, self.fts[self.image_number], 0, 0,
                                                               generation, I1, I2, I3, I4, I5, I6])

                            break

            # print("self.cell_rasta_labellist add" + str(self.cell_rasta_labellist))
            # print("self.cell_truck_labellist add" + str(self.cell_truck_labellist))
            # print("self.cell_truck_detaillist add" + str(self.cell_truck_detaillist))

            self.cell_rasta_labellist = [x[1] for x in self.cell_truck_detaillist]

            # print("self.cell_rasta_labellist sort" + str(self.cell_rasta_labellist))

        # 初回の処理
        if not self.tracking_flag:
            self.cell_truck_detaillist = []
            # self.cell_truck_detaillist = [[-1 for i in range(8)] for j in range(self.n)]  # 前画像のcelllabelの情報を記録しておくフラグ(初期化)
            self.cell_truck_labellist = []  # 現在追跡しているラベルリスト，list内にラべルがあるかの判定で1次リストを使用
            self.cell_rasta_labellist = []  # 追跡しているラベルの元のラベルリスト（ラスタスキャンで振られた）リスト，list内にラべルがあるかの判定で1次リストを使用

            for i in range(self.nLabels - 1):
                if self.cell_existence[i] == 1:
                    self.cell_truck_labellist.append(i)
                    self.cell_rasta_labellist.append(i)
                    # データ保存のときのみ行う処理
                    if self.fluorescent_flag and self.data_save_only_flag:
                        I1 = self.fluorescence_intensity(i + 1)
                    """
                    self.cell_truck_detaillist[i][0] = i #現在ラベル
                    #仮ラベル（ラスタスキャンで決まったラベル）
                    self.cell_truck_detaillist[i][2] = self.center[i][0] #
                    self.cell_truck_detaillist[i][3] = self.center[i][1] #
                    self.cell_truck_detaillist[i][4] = self.data[i, 0] #
                    self.cell_truck_detaillist[i][5] = self.data[i, 1] #
                    self.cell_truck_detaillist[i][6] = self.perimeter[i]
                    self.cell_truck_detaillist[i][7] = self.data[i, 4]
                    """
                    # 0,現在ラベル番号 1,ラスタスキャンラベル番号 2,重心x 3,重心y 4,外接矩形の左上x座標 5,外接矩形の左上y座標
                    # 6,周囲長 7,面積 8,円形度 9,ラベル修正オブジェクトフラグ（ラベル修正した細胞かどうか,1以上が修正）10,親番号
                    # 11,出生時刻(絶対時刻) 12,分裂時刻（相対時刻） 13,分裂回数 14,世代 15~20,相対蛍光強度（avg(x) / avg(mcherry)）
                    self.cell_truck_detaillist.append([i, -1, self.center[i][0], self.center[i][1],
                                                       self.data[i, 0], self.data[i, 1], self.perimeter_list[i],
                                                       self.data[i, 4], self.circularity_list[i], 0, 0,
                                                       self.fts[self.image_number], 0, 0, 0, I1, I2, I3, I4, I5, I6])
        #   円形度確認
        # for item in self.cell_truck_detaillist:
        #     print("["+str(item[0])+"]perimeter,area,circularity: "+str(item[6])+","+str(item[7])+","+str(item[8]))

        # dstにあらかじめ用意したランダムのRGB値で配色
        # 生死フラグに背景分の要素を追加する必要があるため，リストの最前列に0を追加+rastalabellistをlabellistに合わせるため+1する
        self.cell_existence.insert(0, 0)
        self.cell_rasta_labellist_plus1 = list(map(lambda x: x + 1, self.cell_rasta_labellist))
        # self.cell_rasta_labellist_plus1 = [i + 1 for i in self.cell_rasta_labellist]

        # self.hist_label = []
        for y in range(self.height_pred):
            for x in range(self.width_pred):
                # 生きている(cell_existence = 1)細胞のラベルのみランダムで配色，
                if self.cell_existence[self.labelImg[y, x]] == 1:
                    # # ヒストグラムに使う配列に格納
                    # self.hist_label.append([self.labelImg[y, x], x, y])

                    dst[y, x] = self.colors[
                        self.cell_truck_detaillist[self.cell_rasta_labellist_plus1.index(self.labelImg[y, x])][0]]
                    if self.cfps[self.image_number][y, x] == 255:  # もしcfpの領域内なら青色に表示
                        dst[y, x] = [255, 0, 0]

                else:
                    dst[y, x] = [0, 0, 0]
        # print(cell_existence)

        # 背景分の生死フラグの削除
        self.cell_existence.pop(0)

        """
        for i in range(0, self.nLabels - 1):
            if self.cell_existence[i] == 1:
                x = self.data[i, 0]
                y = self.data[i, 1]
                if x == 0 or y == 0:
                    cv2.putText(dst, str(i+1), (x + 10, y + 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
                else:
                    cv2.putText(dst, str(i+1), (x - 10, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
        """

        # TxsR領域が含まれている細胞の総数
        # TxsR_n = 0

        if self.textflag:
            for i in range(len(self.cell_truck_detaillist)):
                x = self.cell_truck_detaillist[i][4]
                y = self.cell_truck_detaillist[i][5]
                # if self.cell_truck_detaillist[i][10] == 0:  # 通常は黄色文字表示
                if x == 0 or y == 0:
                    cv2.putText(dst, str(self.cell_truck_detaillist[i][0] + 1), (x + 5, y + 5),
                                cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 255))
                else:
                    cv2.putText(dst, str(self.cell_truck_detaillist[i][0] + 1), (x - 5, y - 5),
                                cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 255))
                # else:  # TxsRの領域内の場合赤文字表示
                #     TxsR_n += 1
                #     if x == 0 or y == 0:
                #         cv2.putText(dst, str(self.cell_truck_detaillist[i][0] + 1), (x + 10, y + 20),
                #                     cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 0, 255))
                #     else:
                #         cv2.putText(dst, str(self.cell_truck_detaillist[i][0] + 1), (x - 10, y - 5),
                #                     cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 0, 255))

            # オブジェクトの総数を黄文字で表示
            cv2.putText(dst, "Cells : " + str(self.n), (400, 500), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
            # cv2.putText(dst, "Dye Cells : " + str(TxsR_n), (450, 470), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))

        img_drgb = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
        img_dpil = Image.fromarray(img_drgb)
        img_dtk = ImageTk.PhotoImage(img_dpil)
        self.dsts_tk[self.image_number] = img_dtk

    def fluorescence_intensity(self, label):
        # 蛍光強度
        # 初期化
        self.hist_mcherry = np.array([])
        self.hist_yfp = np.array([])

        for item in self.hist_label:
            if label == item[0]:
                # 該当座標の輝度値を格納
                if self.mCherry_flag and self.hist_bri_lower < self.mcherrys[self.image_number][item[2], item[1]]:
                    self.hist_mcherry = np.append(self.mcherrys[self.image_number][item[2], item[1]],
                                                  self.hist_mcherry)
                if self.YFP_flag and self.hist_bri_lower < self.yfps[self.image_number][item[2], item[1]]:
                    self.hist_yfp = np.append(self.yfps[self.image_number][item[2], item[1]],
                                              self.hist_yfp)

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

        # print("label: " + str(label))
        c_mcherry = collections.Counter(self.hist_mcherry)
        c_yfp = collections.Counter(self.hist_yfp)
        # print(c_mcherry)
        # print(c_yfp)
        if c_mcherry and c_yfp:
            if self.flu_dataset_names[0] == "mCherry":
                range_mcherry = int(c_mcherry.most_common()[0][1] * ((100 - self.hist_freq_range_flu1) / 100))
                range_yfp = int(c_yfp.most_common()[0][1] * ((100 - self.hist_freq_range_flu2) / 100))
            else:
                range_mcherry = int(c_mcherry.most_common()[0][1] * ((100 - self.hist_freq_range_flu2) / 100))
                range_yfp = int(c_yfp.most_common()[0][1] * ((100 - self.hist_freq_range_flu1) / 100))
            # print(range_mcherry)
            # print(range_yfp)
            c_mcherry_bind = {key: value for key, value in c_mcherry.items() if value >= range_mcherry}
            c_yfp_bind = {key: value for key, value in c_yfp.items() if value >= range_yfp}
            # print(c_mcherry_bind)
            # print(c_yfp_bind)
            mc_sum = 0
            mc_fre_sum = 0
            yfp_sum = 0
            yfp_fre_sum = 0
            for bri in range(int(min(c_mcherry_bind.keys())), int(max(c_mcherry_bind.keys()) + 1)):
                if bri in c_mcherry:
                    mc_sum += c_mcherry[bri] * bri
                    mc_fre_sum += c_mcherry[bri]
            for bri in range(int(min(c_yfp_bind.keys())), int(max(c_yfp_bind.keys()) + 1)):
                if bri in c_yfp:
                    yfp_sum += c_yfp[bri] * bri
                    yfp_fre_sum += c_yfp[bri]

            avg_mc = mc_sum / mc_fre_sum
            avg_yfp = yfp_sum / yfp_fre_sum

            # print(avg_mc)
            # print(avg_yfp)

            intensity = avg_yfp / avg_mc

        else:
            intensity = -1

        return intensity

    def btrace(self, region, labelImg_list):
        around = 0.0
        area_around = 0
        for y in range(self.height_pred):
            for x in range(self.width_pred):
                if labelImg_list[y][x] == region:
                    sy = y
                    sx = x
                    cy = y
                    cx = x
                    move_direct = 0

                    while True:
                        # current x,yが領域外に行く時はindex errorにならないよう対応
                        # （おそらく-1はheight,width(最後尾)に対応しているためこの条件文でよい）
                        # if cy != self.height and cx != self.width:
                        # dst[cy, cx] = [0, 255, 255]

                        tmp_direct = (move_direct + 5) % 8
                        tracecount = 0
                        while True:
                            if tmp_direct == 0:
                                nx = cx - 1
                                ny = cy
                                tracecount = tracecount + 1
                            # print(0)
                            elif tmp_direct == 1:
                                nx = cx - 1
                                ny = cy + 1
                                tracecount = tracecount + 1
                            # print(1)
                            elif tmp_direct == 2:
                                nx = cx
                                ny = cy + 1
                                tracecount = tracecount + 1
                            elif tmp_direct == 3:
                                nx = cx + 1
                                ny = cy + 1
                                tracecount = tracecount + 1
                            # print(3)
                            elif tmp_direct == 4:
                                nx = cx + 1
                                ny = cy
                                tracecount = tracecount + 1
                            # print(4)
                            elif tmp_direct == 5:
                                nx = cx + 1
                                ny = cy - 1
                                tracecount = tracecount + 1
                            # print(5)
                            elif tmp_direct == 6:
                                nx = cx
                                ny = cy - 1
                                tracecount = tracecount + 1
                            # print(6)
                            else:
                                nx = cx - 1
                                ny = cy - 1
                                tracecount = tracecount + 1
                            # print(7)
                            if tracecount == 8:  #
                                # print("label[" + str(region) + "]trace finished")
                                return around, area_around
                            if labelImg_list[ny][nx] == region:
                                break
                            tmp_direct = (tmp_direct + 1) % 8
                        cx = nx
                        cy = ny
                        move_direct = tmp_direct
                        if move_direct % 2 == 0:
                            around = around + 1.0
                        else:
                            around = around + math.sqrt(2)
                        area_around += 1
                        if cx == sx and cy == sy:  # 1周して同じ地点に戻ってきたらbreak
                            # print("label[" + str(region) + "]trace finished")
                            return around, area_around
        return around, area_around

    def area_filter(self, exist_labels, cell_existence, area, cleanth):
        if area < cleanth:
            cell_existence = 0  # 背景は含めていないためcell_existence[0]→1つ目の細胞の生死フラグであることに注意
            exist_labels -= 1

        return cell_existence, exist_labels

    def circularity_filter(self, lnum, exist_labels, cell_existence, perimeter, area, circuth, area_around):
        old_cell_circularity = 0.0
        cell_circularity = 0.0
        # 面積の小さい細胞は除外
        if cell_existence == 1 and perimeter != 0:  # 細胞の面積処理後にperimeterが0になるオブジェクトが残るわけないのでおかしいから直す
            old_cell_circularity = 4 * math.pi * area / math.pow(perimeter, 2)
            cell_circularity = 4 * math.pi * (float(area - area_around) + float(area_around) / 2.0) / math.pow(
                perimeter, 2)
            # 正規化するため1を超えるオブジェクト（本当ならこれも１に抑えたいが…）を１にする
            if cell_circularity > 1:
                cell_circularity = 1
            # if cell_circularity >= 1:
            #     print("[" + str(lnum + 1) + "] old cell_circularity:" + str(old_cell_circularity))
            #     print("[" + str(lnum + 1) + "] new cell_circularity:" + str(cell_circularity))
            #     print("[" + str(lnum + 1) + "] new cell_circularity_detail:" + str(perimeter), str(area), str(area_around))

            if cell_circularity <= circuth:
                cell_existence = 0
                exist_labels -= 1
                # print("[" + str(i + 1) + "] lost by circularity = " + str(cell_circularity))
                self.systext.insert("1.0", "[" + str(lnum + 1) + "] lost by circularity = " +
                                    "{0:.5f}".format(cell_circularity) + "\n")
        return cell_existence, exist_labels, cell_circularity

    def has_duplicates(self, seq):
        seen = []
        unique_list = [x for x in seq if x not in seen and not seen.append(x)]
        return len(seq) != len(unique_list)
