import pickle
import numpy as np
from math import factorial
import time
import serial
import threading

# from scipy.signal import savgol_filter

port = 'COM8'
baudrate = 115200

print('port: ', port)
print('baudrate: ', baudrate)

R = ''
ser = serial.Serial(port, baudrate, timeout=0.001)
print('serial connect success')
print(ser)

ser.write(b'\x80\x73\x73\x8f')  # monitor에 's' 보내기

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


def moving_average(data, window_size=5):
    filtered_data = np.sum(data[-window_size:]) / window_size
    return filtered_data


def differentiate(current_val, previous_val):
    if previous_val is None:
        return 0.0
    else:
        return (current_val - previous_val) * 0.016


class SerialReceiver(threading.Thread):

    def __init__(self, lock, sema1, sema2):
        super().__init__()

        self.ibp_val = ''
        self.flow_val = ''
        self.pump1 = ''
        self.pump2 = ''

        self.lock = lock
        self.sema1 = sema1
        self.sema2 = sema2

    def run(self):

        print('Serial Receiver start')

        global R

        while 1:

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

                                self.ibp_val = (int(ibp_h, 16) & int('01111111', 2)) * (2 ** 7) + \
                                               (int(ibp_l, 16) & int('01111111', 2)) - 512

                                flow_h = R[i + 18:i + 20]
                                flow_l = R[i + 20:i + 22]

                                self.flow_val = (int(flow_h, 16) & int('01111111', 2)) * (2 ** 7) + \
                                                (int(flow_l, 16) & int('01111111', 2)) - 512

                                sac_sig = R[i + 22:i + 24]

                                pump_sig = format(int(sac_sig, 16), 'b').zfill(8)

                                self.pump1 = pump_sig[7]
                                self.pump2 = pump_sig[6]

                                with self.lock:
                                    self.sema1.release()

                                j_ref = 1
                                break

                        if j_ref == 0:
                            R = R[i::]
                            break


class DataPreprocessor(threading.Thread):
    def __init__(self, parent, lock, sema1, sema2):
        super().__init__()

        self.parent = parent

        self.ibp_wave_arr = ([])
        self.ibp_filtered_arr = ([])
        self.ibp_diff_tmp_arr = ([])
        self.ibp_diff_arr = ([])
        self.pump1_arr = ([])
        self.pump2_arr = ([])
        self.flow_arr = ([])
        self.serial_loop_n = 0
        self.sync_loop_n = 0

        self.lock = lock
        self.sema1 = sema1
        self.sema2 = sema2

    def run(self):

        print('Data Preprocessor start')

        self.sema1.acquire()

        while 1:

            with self.lock:

                ibp_val = self.parent.ibp_val
                flow_val = self.parent.flow_val
                pump1 = self.parent.pump1
                pump2 = self.parent.pump2

                if pump1 == '0':
                    pump1 = 1
                else:
                    pump1 = 0

                if pump2 == '0':
                    pump2 = 1
                else:
                    pump2 = 0

                self.ibp_wave_arr = np.append(self.ibp_wave_arr, ibp_val)
                if len(self.ibp_wave_arr) > 100:
                    self.ibp_wave_arr = np.array(self.ibp_wave_arr[1::])
                # print("ibp: ", *self.ibp_wave_arr)

                if len(self.ibp_wave_arr) >= 10:

                    self.ibp_filtered_arr = np.append(self.ibp_filtered_arr, moving_average(self.ibp_wave_arr))
                    if len(self.ibp_filtered_arr) > 100:
                        self.ibp_filtered_arr = np.array(self.ibp_filtered_arr[1::])
                    # print("ibp_filtered: ", *self.ibp_filtered_arr)

                    if len(self.ibp_filtered_arr) >= 2:

                        self.ibp_diff_tmp_arr = np.append(self.ibp_diff_tmp_arr,
                                                          differentiate(self.ibp_filtered_arr[-1],
                                                                        self.ibp_filtered_arr[-2]))
                        if len(self.ibp_diff_tmp_arr) > 500:
                            self.ibp_diff_tmp_arr = np.array(self.ibp_diff_tmp_arr[1::])
                        # print("ibp_diff: ", *self.ibp_diff_arr)

                        if self.serial_loop_n % 4 == 0:

                            self.ibp_diff_arr = np.append(self.ibp_diff_arr, self.ibp_diff_tmp_arr[-1])
                            if len(self.ibp_diff_arr) > 500:  # 데이터 하나씩 들어 가는지 확인
                                self.ibp_diff_arr = np.array(self.ibp_diff_arr[1::])
                            print("ibp_diff_arr: ", *self.ibp_diff_arr)

                            self.pump1_arr = np.append(self.pump1_arr, pump1)
                            if len(self.pump1_arr) > 500:
                                self.pump1_arr = np.array(self.pump1_arr[1::])
                            # print('pump1: ', *self.pump1_arr)

                            self.pump2_arr = np.append(self.pump2_arr, pump2)
                            if len(self.pump2_arr) > 500:
                                self.pump2_arr = np.array(self.pump2_arr[1::])
                            # print('pump2: ', *self.pump2_arr)

                            self.flow_arr = np.append(self.flow_arr, flow_val)
                            if len(self.flow_arr) > 500:
                                self.flow_arr = np.array(self.flow_arr[1::])

                            print("flow: ", *self.flow_arr)
                            print('')

                            if len(self.ibp_diff_arr) > 125:
                                self.sema2.release()

                            self.sync_loop_n += 1

                self.serial_loop_n += 1


save_diff = np.array([])
save_sac1 = np.array([])
save_sac2 = np.array([])
save_outp_h = np.array([])
save_outp_e = np.array([])


class ApplyDNN(threading.Thread):
    def __init__(self, parent, lock, sema2):
        super().__init__()

        self.parent = parent
        self.dnn_loop_n = 0

        self.lock = lock
        self.sema2 = sema2

        # self.lock = threading.Lock()
        # self.processing_signal = False

    def run(self):

        global save_diff, save_sac1, save_sac2, save_outp_h, save_outp_e

        print("DNN apply start")

        self.sema2.acquire()

        while 1:

            with self.lock:

                ibp_diff_arr = self.parent.ibp_diff_arr
                pump1_arr = self.parent.pump1_arr
                pump2_arr = self.parent.pump2_arr

                # index = -1 * (serial_loop_n - self.parent.dnn_loop_n)

                # print('')

                if len(ibp_diff_arr) >= 125:

                    ibp_tmp = np.array(ibp_diff_arr[-126:-1], dtype='float32')
                    pump1_tmp = np.array(pump1_arr[-91:-1], dtype='float32')
                    pump2_tmp = np.array(pump2_arr[-91:-1], dtype='float32')

                    # print(11111, len(ibp_tmp), len(pump1_tmp), len(pump2_tmp))

                    # diff 필요한 범위만 가져온 뒤 표준화
                    diff_inp = ibp_tmp
                    save_diff_val = np.array(diff_inp)

                    diff_inp_min = np.min(diff_inp)
                    diff_inp_max = np.max(diff_inp)

                    diff_principle = diff_inp_max - diff_inp_min

                    if diff_principle == 0:
                        diff_inp = (diff_inp - diff_inp_min) / 0.001
                    else:
                        diff_inp = (diff_inp - diff_inp_min) / (diff_inp_max - diff_inp_min)

                    # print(diff_inp)

                    # sac1 필요한 범위만 가져오기
                    sac1_inp = pump1_tmp
                    save_sac1_val = np.array(sac1_inp)

                    # sac2 필요한 범위만 가져오기
                    sac2_inp = pump2_tmp
                    save_sac2_val = np.array(sac2_inp)

                    inp = np.hstack((diff_inp, sac1_inp, sac2_inp))
                    # inp = np.reshape(inp, (1, len(inp)))                  # shape change
                    # print('input shape: ', inp.shape)
                    # print('inp: ', *inp)

                    # print('diff_inp: ', *diff_inp)
                    # print('diff_inp: ', len(diff_inp))
                    # print('sac1_inp: ', *sac1_inp)
                    # print('sac1_inp: ', len(sac1_inp))
                    # print('sac2_inp: ', *sac2_inp)
                    # print('sac2_inp: ', len(sac2_inp))

                    # DNN 적용
                    outp_h = DNN(inp, w1, b1, w2, b2, w3, b3)
                    outp_e = DNN(inp, w4, b4, w5, b5, w6, b6)
                    # outp_h = np.reshape(outp_h, (1, len(outp_h)))
                    # outp_e = np.reshape(outp_e, (1, len(outp_e)))
                    # print('output shape: ', outp_h.shape, outp_e.shape)
                    # print('output shape_h: ', *outp_h)
                    # print('output shape_e: ', *outp_e)

                    # DNN 출력값 중 마지막 값(파형 없음을 나타내는) 제거
                    outp_h = np.array(outp_h[0:-1])
                    outp_e = np.array(outp_e[0:-1])
                    # print('output shape_h_d: ', *outp_h)
                    # print('output shape_e_d: ', *outp_e)

                    # 출력값 누적
                    if self.dnn_loop_n == 0:
                        save_diff = np.array(save_diff_val)
                        save_sac1 = np.hstack((np.zeros(35), save_sac1_val))
                        save_sac2 = np.hstack((np.zeros(35), save_sac2_val))

                        save_outp_h = np.hstack((np.zeros(65), outp_h, np.zeros(30)))
                        # print('save_outp_h_1', save_outp_h)
                        save_outp_e = np.hstack((np.zeros(65), outp_e, np.zeros(30)))
                    else:
                        save_diff = np.append(save_diff, save_diff_val[-1])
                        save_sac1 = np.append(save_sac1, save_sac1_val[-1])
                        save_sac2 = np.append(save_sac2, save_sac2_val[-1])

                        save_outp_h = np.append(save_outp_h, 0)
                        # print('save_outp_h_2', save_outp_h)

                        save_outp_h[-60:-30] = save_outp_h[-60:-30] + outp_h
                        # print('save_outp_h_3', save_outp_h)

                        save_outp_e = np.append(save_outp_e, 0)
                        save_outp_e[-60:-30] = save_outp_e[-60:-30] + outp_e

                    a1 = save_outp_h[-60]
                    a2 = save_outp_e[-60]
                    print(a1, a2)
                    # print('save_output shape: ', save_outp_h.shape, save_outp_e.shape)
                    # print('save_output_h: ', *save_outp_h)
                    # print('save_output_e: ', *save_outp_e)
                    print('')

                    self.dnn_loop_n += 1

                else:
                    pass


class MyProgram:
    def __init__(self):
        lock = threading.Lock()
        sema1 = threading.Semaphore(0)
        sema2 = threading.Semaphore(0)

        self.thread1 = SerialReceiver(lock, sema1, sema2)
        self.thread2 = DataPreprocessor(self.thread1, lock, sema1, sema2)
        self.thread3 = ApplyDNN(self.thread2, lock, sema2)

    def run(self):
        self.thread1.start()
        self.thread2.start()
        # self.thread3.start()

        self.thread1.join()
        self.thread2.join()
        # self.thread3.join()


my_program = MyProgram()
my_program.run()
