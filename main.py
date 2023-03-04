import pickle
import numpy as np
import math
import time
import serial
import threading
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

port = 'COM8'
baudrate = 115200

print('port: ', port)
print('baudrate: ', baudrate)

save_path = r'C:\Users\user\PycharmProjects\threading\ecmo_ai_model_221007.pickle'

with open(save_path, 'rb') as f:
    trainingdb = pickle.load(f)

w1 = trainingdb['heart']['w1']
w2 = trainingdb['heart']['w2']
w3 = trainingdb['heart']['w3']
w4 = trainingdb['ecmo']['w4']
w5 = trainingdb['ecmo']['w5']
w6 = trainingdb['ecmo']['w6']

b1 = trainingdb['heart']['b1']
b2 = trainingdb['heart']['b2']
b3 = trainingdb['heart']['b3']
b4 = trainingdb['ecmo']['b4']
b5 = trainingdb['ecmo']['b5']
b6 = trainingdb['ecmo']['b6']


def NDivision(arr, n, ref):
    outp = np.array([])
    for i in range(0, len(arr)):
        if i % n == ref:
            outp = np.append(outp, arr[i])
    return outp


def DNN(x, h1_w, h1_b, h2_w, h2_b, o_w, o_b):
    def Dnn_Relu(arr):
        for i in range(0, arr.shape[0]):
            if arr[i] < 0:
                arr[i] = 0
            else:
                pass
        return arr

    z1 = np.matmul(x, h1_w) + h1_b
    z1 = np.array(z1.reshape(z1.shape[0] * z1.shape[1]))
    z1 = Dnn_Relu(z1)
    z2 = np.matmul(z1, h2_w) + h2_b
    z2 = np.array(z2.reshape(z2.shape[0] * z2.shape[1]))
    z2 = Dnn_Relu(z2)
    z = np.matmul(z2, o_w) + o_b
    z = np.array(z.reshape(z.shape[0] * z.shape[1]))
    outp = np.zeros(len(z))
    outp[np.argmax(z)] = 1

    return outp


def differentiate(current_val, previous_val):
    if previous_val is None:
        return 0.0
    else:
        return (current_val - previous_val) * 0.004


ser = serial.Serial(port, baudrate, timeout=0.5)
print('serial connect success')
print(ser)

ser.write(b'\x80\x73\x73\x8f')  # monitor에 's' 보내기

R = ''
ibp_diff = ''
ibp_val = ''
last_ibp_val = 0


class DataPreprocessor(QThread):
    def __init__(self, parent):
        super().__init__()
        self.running = True

        self.parent = parent
        # self.ibp_wave_arr = np.array([])
        # self.serial_loop_n = 0

    def run(self):
        global R, ibp_val, last_ibp_val, ibp_diff

        while self.running:

            if ser.readable():
                a = ser.readline()

                R += a.hex()

                for i in range(0, len(R), 2):
                    if R[i:i + 2] == '80':

                        j_ref = 0
                        for j in range(i + 2, len(R), 2):
                            if R[j:j + 2] == '8f':

                                # spo_h = R[i + 6:i + 8]
                                # spo_l = R[i + 8:i + 10]
                                #
                                # spo_val = (int(spo_h, 16) & int('01111111', 2)) * (2 ** 7) + \
                                #           (int(spo_l, 16) & int('01111111', 2)) - 1000

                                ibp_h = R[i + 10:i + 12]
                                ibp_l = R[i + 12:i + 14]

                                ibp_val = (int(ibp_h, 16) & int('01111111', 2)) * (2 ** 7) + \
                                          (int(ibp_l, 16) & int('01111111', 2)) - 512

                                if ibp_val is not None:
                                    ibp_diff = differentiate(ibp_val, last_ibp_val)
                                    # print('Differentiated IBP value:', ibp_diff)
                                    last_ibp_val = ibp_val
                                else:
                                    print('Error: ibp_val is None')

                                sac_sig = R[i + 22:i + 24]

                                pump_sig = format(int(sac_sig, 16), 'b').zfill(8)
                                pump1 = pump_sig[7]
                                pump2 = pump_sig[6]

                                # print('pump1, pump2: ', pump1, pump2)

                                self.parent.ibp_wave_arr = np.append(self.parent.ibp_wave_arr, ibp_diff)
                                if len(self.parent.ibp_wave_arr) > 3000:
                                    self.parent.ibp_wave_arr = np.array(self.parent.ibp_wave_arr[1::])

                                self.parent.pump1_arr = np.append(self.parent.pump1_arr, pump1)
                                if len(self.parent.pump1_arr) > 3000:
                                    self.parent.pump1_arr = np.array(self.parent.pump1_arr[1::])

                                self.parent.pump2_arr = np.append(self.parent.pump2_arr, pump2)
                                if len(self.parent.pump2_arr) > 3000:
                                    self.parent.pump2_arr = np.array(self.parent.pump2_arr[1::])

                                self.parent.serial_loop_n += 1

                                j_ref = 1
                                break

                        if j_ref == 0:
                            R = R[i::]
                            break

    def resume(self):
        self.running = True

    def pause(self):
        self.running = False


save_outp_h = ''
save_outp_e = ''


class ApplyDNN(QThread):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

    def run(self):

        global save_outp_h, save_outp_e
        i = 0

        while 1:
            serial_loop_n = self.parent.serial_loop_n
            ibp_wave_arr = self.parent.ibp_wave_arr
            pump1_arr = self.parent.pump1_arr
            pump2_arr = self.parent.pump2_arr

            if self.parent.dnn_loop_n >= serial_loop_n:
                time.sleep(0.1)
                continue

            if serial_loop_n % 4 == 0:

                index = -1 * (serial_loop_n - self.parent.dnn_loop_n)

                ibp_tmp = np.array(ibp_wave_arr[index - 125 * 4: index])
                # pump1_tmp = np.array(pump1_arr[index - 125: index])
                # pump2_tmp = np.array(pump2_arr[index - 125: index])
                pump1_tmp = np.array(pump1_arr[index - 90 * 4: index])
                pump2_tmp = np.array(pump2_arr[index - 90 * 4: index])

                # print(ibp_tmp)

                # diff, sac1, sac2 각각 16분주 중 (div_i) 번째 분주 값 가져오기
                for div_i in range(0, 1):
                    diff = NDivision(ibp_tmp, 4, div_i)
                    sac1 = NDivision(pump1_tmp, 4, div_i)
                    sac2 = NDivision(pump2_tmp, 4, div_i)

                    save_diff = np.array([])
                    save_sac1 = np.array([])
                    save_sac2 = np.array([])
                    save_outp_h = np.array([])
                    save_outp_e = np.array([])

                    # diff 필요한 범위만 가져온 뒤 표준화
                    # diff_inp = diff[serial_loop_n - 125:serial_loop_n]
                    diff_inp = np.array(diff)
                    # save_diff_val = np.array(diff_inp)
                    diff_inp_min = np.min(diff_inp)
                    diff_inp_max = np.max(diff_inp)

                    # diff_inp = (diff_inp - diff_inp_min) / (diff_inp_max - diff_inp_min)
                    if diff_inp_max == diff_inp_min:
                        diff_inp = np.zeros_like(diff_inp)
                    else:
                        diff_inp = (diff_inp - diff_inp_min) / (diff_inp_max - diff_inp_min)

                    # sac1 필요한 범위만 가져오기
                    # sac1_inp = sac1[serial_loop_n - 90:serial_loop_n]
                    sac1_inp = np.array(sac1)
                    # save_sac1_val = np.array(sac1_inp)
                    # for sac_i in range(0, len(sac1_inp)):
                    #     if sac1_inp[sac_i] >= 0.5:
                    #         sac1_inp[sac_i] = 1
                    #     else:
                    #         sac1_inp[sac_i] = 0

                    # sac2 필요한 범위만 가져오기
                    # sac2_inp = sac2[serial_loop_n - 90:serial_loop_n]
                    sac2_inp = np.array(sac2)
                    # save_sac2_val = np.array(sac2_inp)
                    # for sac_i in range(0, len(sac2_inp)):
                    #     if sac2_inp[sac_i] >= 0.5:
                    #         sac2_inp[sac_i] = 1
                    #     else:
                    #         sac2_inp[sac_i] = 0

                    inp = np.hstack((diff_inp, sac1_inp, sac2_inp))

                    # DNN 적용
                    outp_h = DNN(inp, w1, b1, w2, b2, w3, b3)
                    outp_e = DNN(inp, w4, b4, w5, b5, w6, b6)

                    # print(outp_h, outp_e)

                    # DNN 출력값 중 마지막 값(파형 없음을 나타내는) 제거
                    outp_h = np.array(outp_h[0:-1])
                    outp_e = np.array(outp_e[0:-1])

                    # 출력값 누적
                    if i == 0:
                        save_diff = np.array(diff)
                        save_sac1 = np.hstack((np.zeros(35), save_sac1))
                        save_sac2 = np.hstack((np.zeros(35), save_sac2))
                        save_outp_h = np.hstack((np.zeros(65), outp_h, np.zeros(30)))
                        save_outp_e = np.hstack((np.zeros(65), outp_e, np.zeros(30)))
                    else:
                        save_diff = np.append(save_diff, diff[-1])

                        save_sac1 = np.append(save_sac1, save_sac1[-1])
                        save_sac2 = np.append(save_sac2, save_sac2[-1])

                        save_outp_h = np.append(save_outp_h, 0)
                        save_outp_h[-60:-30] = save_outp_h[-60:-30] + outp_h

                        save_outp_e = np.append(save_outp_e, 0)
                        save_outp_e[-60:-30] = save_outp_e[-60:-30] + outp_e

                    self.parent.dnn_loop_n += 1

                    a1 = save_outp_h[-60]
                    a2 = save_outp_e[-60]

                    print(a1, a2)

    def resume(self):
        pass

    def pause(self):
        pass


class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.ibp_wave_arr = np.array([])
        self.pump1_arr = np.array([])
        self.pump2_arr = np.array([])
        self.serial_loop_n = 0
        self.dnn_loop_n = 0

        self.controller = DataPreprocessor(self)
        self.controller.start()

        self.applydnn = ApplyDNN(self)
        self.applydnn.start()

        # window size
        self.resize(300, 100)

        btn1 = QPushButton("resume", self)
        btn1.move(10, 10)
        btn2 = QPushButton("pause", self)
        btn2.move(10, 50)

        # 시그널-슬롯 연결하기
        btn1.clicked.connect(self.resume)
        btn2.clicked.connect(self.pause)

    def resume(self):
        self.controller.resume()
        self.controller.start()

    def pause(self):
        self.controller.pause()


app = QApplication(sys.argv)
mywindow = MyWindow()
mywindow.show()
app.exec_()