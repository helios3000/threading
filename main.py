# import pickle
# import numpy as np
# import time
# import serial
# import threading
# import queue
# from collections import deque
#
#
# port = 'COM8'
# baudrate = 115200
#
# print('port: ', port)
# print('baudrate: ', baudrate)
#
# R = ''
# ser = serial.Serial(port, baudrate, timeout=0.001)
# print('serial connect success')
# print(ser)
#
# ser.write(b'\x80\x73\x73\x8f')  # monitor에 's' 보내기
#
# save_path = r'C:\Users\user\PycharmProjects\threading\ecmo_ai_model_221007.pickle'
#
# with open(save_path, 'rb') as f:
#     trainingdb = pickle.load(f)
#
# w1 = trainingdb['heart']['w1']
# w2 = trainingdb['heart']['w2']
# w3 = trainingdb['heart']['w3']
# w4 = trainingdb['ecmo']['w4']
# w5 = trainingdb['ecmo']['w5']
# w6 = trainingdb['ecmo']['w6']
#
# b1 = trainingdb['heart']['b1']
# b2 = trainingdb['heart']['b2']
# b3 = trainingdb['heart']['b3']
# b4 = trainingdb['ecmo']['b4']
# b5 = trainingdb['ecmo']['b5']
# b6 = trainingdb['ecmo']['b6']
#
#
# def NDivision(arr, n, ref):
#     outp = np.array([])
#     for i in range(0, len(arr)):
#         if i % n == ref:
#             outp = np.append(outp, arr[i])
#     return outp
#
#
# def DNN(x, h1_w, h1_b, h2_w, h2_b, o_w, o_b):
#     def Dnn_Relu(arr):
#         for i in range(0, arr.shape[0]):
#             if arr[i] < 0:
#                 arr[i] = 0
#             else:
#                 pass
#         return arr
#
#     z1 = np.matmul(x, h1_w) + h1_b
#     z1 = np.array(z1.reshape(z1.shape[0] * z1.shape[1]))
#     z1 = Dnn_Relu(z1)
#     z2 = np.matmul(z1, h2_w) + h2_b
#     z2 = np.array(z2.reshape(z2.shape[0] * z2.shape[1]))
#     z2 = Dnn_Relu(z2)
#     z = np.matmul(z2, o_w) + o_b
#     z = np.array(z.reshape(z.shape[0] * z.shape[1]))
#     outp = np.zeros(len(z))
#     outp[np.argmax(z)] = 1
#
#     return outp
#
#
# def moving_average(data, window_size):
#     return sum(list(data)[:5]) / window_size
#
#
# def differentiate(current_val, previous_val):
#     if previous_val is None:
#         return 0.0
#     else:
#         return (current_val - previous_val) * 0.016
#
#
# def derivation(data):
#     return (data[0] - data[-1]) * 0.016
#
#
# class SerialReceiver(threading.Thread):
#
#     def __init__(self, ibp_raw_queue, pump1_raw_queue, pump2_raw_queue, flow_raw_queue, lock, sema1):
#         super().__init__()
#
#         # self.ibp_val = ''
#         # self.flow_val = ''
#         # self.pump1 = ''
#         # self.pump2 = ''
#
#         # self.ibp_raw_queue = deque(maxlen=100)
#         # self.pump1_raw_queue = deque(maxlen=100)
#         # self.pump2_raw_queue = deque(maxlen=100)
#         # self.flow_raw_queue = deque(maxlen=100)
#
#         self.ibp_raw_queue = ibp_raw_queue
#         self.pump1_raw_queue = pump1_raw_queue
#         self.pump2_raw_queue = pump2_raw_queue
#         self.flow_raw_queue = flow_raw_queue
#
#         self.lock = lock
#         self.sema1 = sema1
#
#         self.serial_loop_n = 0
#
#     def run(self):
#
#         print('Serial Receiver start')
#
#         global R, ibp_val, pump1, pump2, flow
#
#         while 1:
#
#             if ser.readable():
#                 a = ser.readline()
#
#                 R += a.hex()
#
#                 for i in range(0, len(R), 2):
#                     if R[i:i + 2] == '80':
#
#                         j_ref = 0
#                         for j in range(i + 2, len(R), 2):
#                             if R[j:j + 2] == '8f':
#                                 # spo_h = R[i + 6:i + 8]
#                                 # spo_l = R[i + 8:i + 10]
#                                 #
#                                 # spo_val = (int(spo_h, 16) & int('01111111', 2)) * (2 ** 7) + \
#                                 #           (int(spo_l, 16) & int('01111111', 2)) - 1000
#
#                                 ibp_h = R[i + 10:i + 12]
#                                 ibp_l = R[i + 12:i + 14]
#
#                                 ibp_val = (int(ibp_h, 16) & int('01111111', 2)) * (2 ** 7) + \
#                                           (int(ibp_l, 16) & int('01111111', 2)) - 512
#
#                                 flow_h = R[i + 18:i + 20]
#                                 flow_l = R[i + 20:i + 22]
#
#                                 flow_val = (int(flow_h, 16) & int('01111111', 2)) * (2 ** 7) + \
#                                            (int(flow_l, 16) & int('01111111', 2)) - 512
#
#                                 sac_sig = R[i + 22:i + 24]
#
#                                 pump_sig = format(int(sac_sig, 16), 'b').zfill(8)
#
#                                 pump1 = pump_sig[7]
#                                 pump2 = pump_sig[6]
#
#                                 self.ibp_raw_queue.put(ibp_val)
#                                 self.pump1_raw_queue.put(pump1)
#                                 self.pump2_raw_queue.put(pump2)
#                                 self.flow_raw_queue.put(flow_val)
#
#                                 self.sema1.release()
#
#                                 self.serial_loop_n += 1
#
#                                 # print('raw: ', *list(self.ibp_raw_queue))
#
#                                 # time.sleep(0.01)
#
#                                 j_ref = 1
#                                 break
#
#                         if j_ref == 0:
#                             R = R[i::]
#                             break
#
#
# class DataModifier(threading.Thread):
#     def __init__(self, ibp_raw_queue, pump1_raw_queue, pump2_raw_queue, flow_raw_queue, ibp_filtered_queue,
#                  pump1_sync_queue, pump2_sync_queue, flow_sync_queue, lock, sema1, sema2):
#         super().__init__()
#
#         self.ibp_raw_queue = ibp_raw_queue
#         self.pump1_raw_queue = pump1_raw_queue
#         self.pump2_raw_queue = pump2_raw_queue
#         self.flow_raw_queue = flow_raw_queue
#
#         self.ibp_buffer = deque(maxlen=10)
#         self.pump1_buffer = deque(maxlen=10)
#         self.pump2_buffer = deque(maxlen=10)
#         self.flow_buffer = deque(maxlen=10)
#
#         self.ibp_filtered_queue = ibp_filtered_queue
#         self.pump1_sync_queue = pump1_sync_queue
#         self.pump2_sync_queue = pump2_sync_queue
#         self.flow_sync_queue = flow_sync_queue
#
#         # self.ibp_filtered_queue = deque(maxlen=10)
#         # self.pump1_sync_queue = deque(maxlen=10)
#         # self.pump2_sync_queue = deque(maxlen=10)
#         # self.flow_sync_queue = deque(maxlen=10)
#
#         self.lock = lock
#         self.sema1 = sema1
#         self.sema2 = sema2
#
#     def run(self):
#
#         print('Data Modifier start')
#
#         while 1:
#
#             self.sema1.acquire()
#             time.sleep(0.01)
#
#             with self.lock:
#
#                 # print('raw: ', *list(self.ibp_raw_queue))
#
#                 ibp_raw_data = self.ibp_raw_queue.get()
#                 pump1_raw_data = self.pump1_raw_queue.get()
#                 pump2_raw_data = self.pump2_raw_queue.get()
#                 flow_raw_data = self.flow_raw_queue.get()
#
#                 self.ibp_buffer.append(ibp_raw_data)
#
#                 # if pump1_raw_data == '0':
#                 #     self.pump1_buffer.append(1)
#                 # else:
#                 #     self.pump1_buffer.append(0)
#
#                 pump1_int_data = int(pump1_raw_data)
#                 pump1_int_data ^= 1
#                 self.pump1_buffer.append(pump1_int_data)
#
#                 # if pump2_raw_data == '0':
#                 #     self.pump1_buffer.append(1)
#                 # else:
#                 #     self.pump1_buffer.append(0)
#
#                 pump2_int_data = int(pump2_raw_data)
#                 pump2_int_data ^= 1
#                 self.pump2_buffer.append(pump2_int_data)
#
#                 self.flow_buffer.append(flow_raw_data)
#
#                 if len(self.ibp_buffer) >= 5:
#                     self.ibp_filtered_queue.put(moving_average(self.ibp_buffer, 5))  # moving_average(data, size)
#                     self.pump1_sync_queue.put(self.pump1_buffer.pop())
#                     self.pump2_sync_queue.put(self.pump2_buffer.pop())
#                     self.flow_sync_queue.put(self.flow_buffer.pop())
#
#                     # print('MA: ', *list(self.ibp_filtered_queue))
#                     # print('')
#
#                     self.sema2.release()
#
#                 else:
#                     pass
#
#
# class IbpDerivation(threading.Thread):
#     def __init__(self, ibp_filtered_queue, pump1_sync_queue, pump2_sync_queue, flow_sync_queue, ibp_diff_queue,
#                  pump1_queue, pump2_queue, flow_queue, lock, sema2, sema3):
#         super().__init__()
#
#         self.ibp_filtered_queue = ibp_filtered_queue
#         self.pump1_sync_queue = pump1_sync_queue
#         self.pump2_sync_queue = pump2_sync_queue
#         self.flow_sync_queue = flow_sync_queue
#
#         self.ibp_filtered_buffer = deque(maxlen=2)
#         self.pump1_sync_buffer = deque(maxlen=2)
#         self.pump2_sync_buffer = deque(maxlen=2)
#         self.flow_sync_buffer = deque(maxlen=2)
#
#         self.ibp_diff_queue = ibp_diff_queue
#         self.pump1_queue = pump1_queue
#         self.pump2_queue = pump2_queue
#         self.flow_queue = flow_queue
#
#         # self.ibp_diff_queue = deque(maxlen=10)
#         # self.pump1_queue = deque(maxlen=10)
#         # self.pump2_queue = deque(maxlen=10)
#         # self.flow_queue = deque(maxlen=10)
#
#         self.lock = lock
#         self.sema2 = sema2
#         self.sema3 = sema3
#
#     def run(self):
#
#         print('ibp Derivation start')
#
#         while 1:
#
#             self.sema2.acquire()
#             time.sleep(0.01)
#
#             with self.lock:
#
#                 ibp_filtered_data = self.ibp_filtered_queue.get()
#                 pump1_sync_data = self.pump1_sync_queue.get()
#                 pump2_sync_data = self.pump2_sync_queue.get()
#                 flow_sync_data = self.flow_sync_queue.get()
#
#                 self.ibp_filtered_buffer.append(ibp_filtered_data)
#                 self.pump1_sync_buffer.append(pump1_sync_data)
#                 self.pump2_sync_buffer.append(pump2_sync_data)
#                 self.flow_sync_buffer.append(flow_sync_data)
#
#                 if len(self.ibp_filtered_buffer) == 2:
#                     self.ibp_diff_queue.put(derivation(self.ibp_filtered_buffer))
#                     self.pump1_queue.put(self.pump1_sync_buffer.pop())
#                     self.pump2_queue.put(self.pump2_sync_buffer.pop())
#                     self.flow_queue.put(self.flow_sync_buffer.pop())
#
#                     # print(self.ibp_diff_queue)
#
#                     self.sema3.release()
#
#                 else:
#                     pass
#
#
# class DataAppender(threading.Thread):
#     def __init__(self, ibp_diff_queue, pump1_queue, pump2_queue, flow_queue, ibp_diff_ringbuffer, ibp_pump1_ringbuffer,
#                  ibp_pump2_ringbuffer, flow_ringbuffer, lock, sema3, sema4):
#         super().__init__()
#
#         self.ibp_diff_queue = ibp_diff_queue
#         self.pump1_queue = pump1_queue
#         self.pump2_queue = pump2_queue
#         self.flow_queue = flow_queue
#
#         self.ibp_diff_ringbuffer = ibp_diff_ringbuffer
#         self.ibp_pump1_ringbuffer = ibp_pump1_ringbuffer
#         self.ibp_pump2_ringbuffer = ibp_pump2_ringbuffer
#         self.flow_ringbuffer = flow_ringbuffer
#
#         # self.ibp_diff_ringbuffer = deque(maxlen=200)
#         # self.ibp_pump1_ringbuffer = deque(maxlen=200)
#         # self.ibp_pump2_ringbuffer = deque(maxlen=200)
#         # self.flow_ringbuffer = deque(maxlen=200)
#
#         self.lock = lock
#         self.sema3 = sema3
#         self.sema4 = sema4
#
#     def run(self):
#
#         print('Data Appender start')
#
#         while 1:
#
#             self.sema3.acquire()
#             time.sleep(0.01)
#
#             with self.lock:
#
#                 ibp_diff_data = self.ibp_diff_queue.get()
#                 pump1_data = self.pump1_queue.get()
#                 pump2_data = self.pump2_queue.get()
#                 flow_data = self.flow_queue.get()
#
#                 self.ibp_diff_ringbuffer.append(ibp_diff_data)
#                 self.ibp_pump1_ringbuffer.append(pump1_data)
#                 self.ibp_pump2_ringbuffer.append(pump2_data)
#                 self.flow_ringbuffer.append(flow_data)
#
#                 if len(self.ibp_diff_ringbuffer) >= 125:
#                     self.sema4.release()
#
#
# save_diff = np.array([])
# save_sac1 = np.array([])
# save_sac2 = np.array([])
# save_outp_h = np.array([])
# save_outp_e = np.array([])
#
#
# class ApplyDNN(threading.Thread):
#     def __init__(self, ibp_diff_ringbuffer, ibp_pump1_ringbuffer, ibp_pump2_ringbuffer, flow_ringbuffer, lock, sema4):
#         super().__init__()
#
#         self.ibp_diff_ringbuffer = ibp_diff_ringbuffer
#         self.ibp_pump1_ringbuffer = ibp_pump1_ringbuffer
#         self.ibp_pump2_ringbuffer = ibp_pump2_ringbuffer
#         self.flow_ringbuffer = flow_ringbuffer
#
#         self.dnn_loop_n = 0
#
#         self.lock = lock
#         self.sema4 = sema4
#
#         # self.lock = threading.Lock()
#         # self.processing_signal = False
#
#     def run(self):
#
#         global save_diff, save_sac1, save_sac2, save_outp_h, save_outp_e
#
#         print("DNN apply start")
#
#         while 1:
#
#             self.sema4.acquire()
#             time.sleep(0.01)
#
#             with self.lock:
#
#                 ibp_diff_arr = self.ibp_diff_ringbuffer
#                 pump1_arr = self.ibp_pump1_ringbuffer
#                 pump2_arr = self.ibp_pump2_ringbuffer
#                 flow_arr = self.flow_ringbuffer
#
#                 # index = -1 * (serial_loop_n - self.parent.dnn_loop_n)
#
#                 # print('')
#
#                 if len(ibp_diff_arr) >= 125:
#
#                     ibp_tmp = np.array(list(ibp_diff_arr)[-125:], dtype='float32')
#                     pump1_tmp = np.array(list(pump1_arr)[-90:], dtype='float32')
#                     pump2_tmp = np.array(list(pump2_arr)[-90:], dtype='float32')
#
#                     # print(11111, len(ibp_tmp), len(pump1_tmp), len(pump2_tmp))
#
#                     print('ibp: ', *ibp_tmp)
#                     print('pump1: ', *pump1_tmp)
#                     print('pump2: ', *pump2_tmp)
#
#                     # diff 필요한 범위만 가져온 뒤 표준화
#                     diff_inp = ibp_tmp
#                     save_diff_val = np.array(diff_inp)
#
#                     diff_inp_min = np.min(diff_inp)
#                     diff_inp_max = np.max(diff_inp)
#
#                     diff_principle = diff_inp_max - diff_inp_min
#
#                     if diff_principle == 0:
#                         diff_inp = (diff_inp - diff_inp_min) / 0.001
#                     else:
#                         diff_inp = (diff_inp - diff_inp_min) / (diff_inp_max - diff_inp_min)
#
#                     # print(diff_inp)
#
#                     # sac1 필요한 범위만 가져오기
#                     sac1_inp = pump1_tmp
#                     save_sac1_val = np.array(sac1_inp)
#
#                     # sac2 필요한 범위만 가져오기
#                     sac2_inp = pump2_tmp
#                     save_sac2_val = np.array(sac2_inp)
#
#                     inp = np.hstack((diff_inp, sac1_inp, sac2_inp))
#                     # inp = np.reshape(inp, (1, len(inp)))                  # shape change
#                     # print('input shape: ', inp.shape)
#                     # print('inp: ', *inp)
#
#                     # print('diff_inp: ', *diff_inp)
#                     # print('diff_inp: ', len(diff_inp))
#                     # print('sac1_inp: ', *sac1_inp)
#                     # print('sac1_inp: ', len(sac1_inp))
#                     # print('sac2_inp: ', *sac2_inp)
#                     # print('sac2_inp: ', len(sac2_inp))
#
#                     # DNN 적용
#                     outp_h = DNN(inp, w1, b1, w2, b2, w3, b3)
#                     outp_e = DNN(inp, w4, b4, w5, b5, w6, b6)
#                     # outp_h = np.reshape(outp_h, (1, len(outp_h)))
#                     # outp_e = np.reshape(outp_e, (1, len(outp_e)))
#                     # print('output shape: ', outp_h.shape, outp_e.shape)
#                     # print('output shape_h: ', *outp_h)
#                     # print('output shape_e: ', *outp_e)
#
#                     # DNN 출력값 중 마지막 값(파형 없음을 나타내는) 제거
#                     outp_h = np.array(outp_h[0:-1])
#                     outp_e = np.array(outp_e[0:-1])
#                     # print('output shape_h_d: ', *outp_h)
#                     # print('output shape_e_d: ', *outp_e)
#
#                     # 출력값 누적
#                     if self.dnn_loop_n == 0:
#                         save_diff = np.array(save_diff_val)
#                         save_sac1 = np.hstack((np.zeros(35), save_sac1_val))
#                         save_sac2 = np.hstack((np.zeros(35), save_sac2_val))
#
#                         save_outp_h = np.hstack((np.zeros(65), outp_h, np.zeros(30)))
#                         # print('save_outp_h_1', save_outp_h)
#                         save_outp_e = np.hstack((np.zeros(65), outp_e, np.zeros(30)))
#                     else:
#                         save_diff = np.append(save_diff, save_diff_val[-1])
#                         save_sac1 = np.append(save_sac1, save_sac1_val[-1])
#                         save_sac2 = np.append(save_sac2, save_sac2_val[-1])
#
#                         save_outp_h = np.append(save_outp_h, 0)
#                         # print('save_outp_h_2', save_outp_h)
#
#                         save_outp_h[-60:-30] = save_outp_h[-60:-30] + outp_h
#                         # print('save_outp_h_3', save_outp_h)
#
#                         save_outp_e = np.append(save_outp_e, 0)
#                         save_outp_e[-60:-30] = save_outp_e[-60:-30] + outp_e
#
#                     a1 = save_outp_h[-60]
#                     a2 = save_outp_e[-60]
#                     print(a1, a2)
#                     # print('save_output shape: ', save_outp_h.shape, save_outp_e.shape)
#                     # print('save_output_h: ', *save_outp_h)
#                     # print('save_output_e: ', *save_outp_e)
#                     print('')
#
#                     self.dnn_loop_n += 1
#
#                     if self.dnn_loop_n == 12000:
#                         save_data = np.vstack((save_diff, save_sac1, save_sac2, save_outp_h, save_outp_e)).T
#                         np.savetxt(r'C:\Users\user\Desktop\ecmo_ai_apply_230413.csv',
#                                    save_data, fmt='%s',
#                                    delimiter=",")
#
#                 else:
#                     pass
#
#
# class ProgramInitiator:
#     def __init__(self):
#         # from Serial Receiver
#         ibp_raw_queue = queue.Queue(maxsize=500)
#         pump1_raw_queue = queue.Queue(maxsize=500)
#         pump2_raw_queue = queue.Queue(maxsize=500)
#         flow_raw_queue = queue.Queue(maxsize=500)
#
#         # from Data Modifier
#         ibp_filtered_queue = queue.Queue(maxsize=500)
#         pump1_sync_queue = queue.Queue(maxsize=500)
#         pump2_sync_queue = queue.Queue(maxsize=500)
#         flow_sync_queue = queue.Queue(maxsize=500)
#
#         # from ibp Derivation
#         ibp_diff_queue = queue.Queue(maxsize=500)
#         pump1_queue = queue.Queue(maxsize=500)
#         pump2_queue = queue.Queue(maxsize=500)
#         flow_queue = queue.Queue(maxsize=500)
#
#         # from Data Appender
#         ibp_diff_ringbuffer = deque(maxlen=200)
#         ibp_pump1_ringbuffer = deque(maxlen=200)
#         ibp_pump2_ringbuffer = deque(maxlen=200)
#         flow_ringbuffer = deque(maxlen=200)
#
#         lock = threading.Lock()
#         sema1 = threading.Semaphore(0)
#         sema2 = threading.Semaphore(0)
#         sema3 = threading.Semaphore(0)
#         sema4 = threading.Semaphore(0)
#
#         self.thread1 = SerialReceiver(ibp_raw_queue, pump1_raw_queue, pump2_raw_queue, flow_raw_queue, lock, sema1)
#         self.thread2 = DataModifier(ibp_raw_queue, pump1_raw_queue, pump2_raw_queue, flow_raw_queue, ibp_filtered_queue,
#                                     pump1_sync_queue, pump2_sync_queue, flow_sync_queue, lock, sema1, sema2)
#         self.thread3 = IbpDerivation(ibp_filtered_queue, pump1_sync_queue, pump2_sync_queue, flow_sync_queue,
#                                      ibp_diff_queue, pump1_queue, pump2_queue, flow_queue, lock, sema2, sema3)
#         self.thread4 = DataAppender(ibp_diff_queue, pump1_queue, pump2_queue, flow_queue, ibp_diff_ringbuffer,
#                                     ibp_pump1_ringbuffer, ibp_pump2_ringbuffer, flow_ringbuffer, lock, sema3, sema4)
#         self.thread5 = ApplyDNN(ibp_diff_ringbuffer, ibp_pump1_ringbuffer, ibp_pump2_ringbuffer, flow_ringbuffer, lock,
#                                 sema4)
#
#     def run(self):
#         self.thread1.start()
#         self.thread2.start()
#         # self.thread3.start()
#         # self.thread4.start()
#         # self.thread5.start()
#
#         self.thread1.join()
#         self.thread2.join()
#         # self.thread3.join()
#         # self.thread4.join()
#         # self.thread5.join()
#
#
# initiator = ProgramInitiator()
# initiator.run()


import pickle
import numpy as np
import time
import serial
import threading


port = 'COM8'
baudrate = 115200

print('port: ', port)
print('baudrate: ', baudrate)

R = ''
ser = serial.Serial(port, baudrate, timeout=0.0001)
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


def moving_average(data, window_size=7):
    filtered_data = np.sum(data[-window_size:]) / window_size
    return filtered_data


def moving_average_dnn(x, window_size):
    weights = np.repeat(1.0, window_size)/window_size
    return np.convolve(x, weights, 'valid')


def derivation(current_val, previous_val):
    if previous_val is None:
        return 0.0
    else:
        return (current_val - previous_val) * 0.0016


class SerialReceiver(threading.Thread):

    def __init__(self, lock, sema1, evt1):
        super().__init__()

        self.ibp_val = ''
        self.flow_val = ''
        self.pump1 = ''
        self.pump2 = ''

        self.lock = lock
        self.sema1 = sema1
        self.evt1 = evt1

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
    def __init__(self, parent, lock, sema1, evt1, evt2):
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
        self.evt1 = evt1
        self.evt2 = evt2

    def run(self):

        print('Data Preprocessor start')

        while 1:

            self.sema1.acquire()
            time.sleep(0.001)                                # thread1 & thread2, 0.001일 때 60bpm 67Hz

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
                if len(self.ibp_wave_arr) > 7:
                    self.ibp_wave_arr = np.array(self.ibp_wave_arr[1::])
                # print("ibp: ", *self.ibp_wave_arr)
                # print(time.time())

                # synchronization
                # self.pump1_arr = np.append(self.pump1_arr, pump1)
                # if len(self.pump1_arr) > 200:
                #     self.pump1_arr = np.array(self.pump1_arr[1::])
                # # print('pump1: ', *self.pump1_arr)
                #
                # self.pump2_arr = np.append(self.pump2_arr, pump2)
                # if len(self.pump2_arr) > 200:
                #     self.pump2_arr = np.array(self.pump2_arr[1::])
                # # print('pump2: ', *self.pump2_arr)
                #
                # self.flow_arr = np.append(self.flow_arr, flow_val)
                # if len(self.flow_arr) > 200:
                #     self.flow_arr = np.array(self.flow_arr[1::])

                # original code before synchronization
                if len(self.ibp_wave_arr) >= 7:

                    self.ibp_filtered_arr = np.append(self.ibp_filtered_arr, moving_average(self.ibp_wave_arr))
                    if len(self.ibp_filtered_arr) > 10:
                        self.ibp_filtered_arr = np.array(self.ibp_filtered_arr[1::])
                    # print("ibp_filtered: ", *self.ibp_filtered_arr)
                    # print(time.time())

                    if len(self.ibp_filtered_arr) >= 2:

                        self.ibp_diff_tmp_arr = np.append(self.ibp_diff_tmp_arr, derivation(self.ibp_filtered_arr[-1],
                                                          self.ibp_filtered_arr[-2]))
                        if len(self.ibp_diff_tmp_arr) > 10:
                            self.ibp_diff_tmp_arr = np.array(self.ibp_diff_tmp_arr[1::])
                        # print("ibp_diff: ", *self.ibp_diff_arr)
                        # print(time.time())
                        # print('')

                        '''
                        데이터 저장
                        '''
                        self.ibp_diff_arr = np.append(self.ibp_diff_arr, self.ibp_diff_tmp_arr[-1])
                        if len(self.ibp_diff_arr) > 300:
                            self.ibp_diff_arr = ([])
                        # print("ibp_diff_arr: ", *self.ibp_diff_arr)

                        self.pump1_arr = np.append(self.pump1_arr, pump1)
                        if len(self.pump1_arr) > 300:
                            self.pump1_arr = ([])
                        # print('pump1: ', *self.pump1_arr)

                        self.pump2_arr = np.append(self.pump2_arr, pump2)
                        if len(self.pump2_arr) > 300:
                            self.pump2_arr = ([])
                        # print('pump2: ', *self.pump2_arr)

                        self.flow_arr = np.append(self.flow_arr, flow_val)
                        if len(self.flow_arr) > 300:
                            self.flow_arr = ([])
                        # print("flow: ", *self.flow_arr)
                        # print('')

                        if len(self.ibp_diff_arr) == 300:
                            self.evt2.set()
                            self.evt1.clear()
                            self.evt1.wait()

                        self.sync_loop_n += 1
                        # print('DataPreprocessor:', self.sync_loop_n)
                        print('ibp: ', len(self.ibp_diff_arr))

                self.serial_loop_n += 1


class ApplyDNN(threading.Thread):
    def __init__(self, parent, lock, evt1, evt2):
        super().__init__()

        self.parent = parent
        self.dnn_loop_n = 0

        self.save_diff = np.array([])
        self.save_sac1 = np.array([])
        self.save_sac2 = np.array([])
        self.save_outp_h = np.array([])
        self.save_outp_e = np.array([])

        self.cp_est_outp_h_tmp = ([])
        self.cp_est_outp_e_tmp = ([])

        self.heart_decision = ([])
        self.ecmo_decision = ([])

        self.lock = lock
        self.evt1 = evt1
        self.evt2 = evt2

        # self.lock = threading.Lock()
        # self.processing_signal = False

    def run(self):

        global i, j, bpm_1, bpm_2, bpm

        k = 0

        print("DNN apply start")

        while 1:

            self.evt2.wait()

            ibp_diff_arr = self.parent.ibp_diff_arr
            pump1_arr = self.parent.pump1_arr
            pump2_arr = self.parent.pump2_arr

            # index = -1 * (serial_loop_n - self.parent.dnn_loop_n)

            # print('')

            for i in range(0, 174):

                ibp_tmp = np.array(ibp_diff_arr[i:i+127], dtype='float32')
                pump1_tmp = np.array(pump1_arr[i+37:i+127], dtype='float32')
                pump2_tmp = np.array(pump2_arr[i+37:i+127], dtype='float32')

                # print('ibp_tmp', *ibp_tmp)

                # print(11111, len(ibp_tmp), len(pump1_tmp), len(pump2_tmp))

                # diff 필요한 범위만 가져온 뒤 표준화
                diff_inp_tmp = ibp_tmp

                diff_inp_min = np.min(diff_inp_tmp)
                diff_inp_max = np.max(diff_inp_tmp)

                diff_principle = diff_inp_max - diff_inp_min

                if diff_principle == 0:
                    diff_inp_tmp = (diff_inp_tmp - diff_inp_min) / 0.1
                else:
                    diff_inp_tmp = (diff_inp_tmp - diff_inp_min) / (diff_inp_max - diff_inp_min)

                diff_inp = moving_average_dnn(diff_inp_tmp, 3)
                save_diff_val = np.array(diff_inp)

                # print('diff_inp', *diff_inp)

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
                    self.save_diff = np.array(save_diff_val)
                    self.save_sac1 = np.hstack((np.zeros(35), save_sac1_val))
                    self.save_sac2 = np.hstack((np.zeros(35), save_sac2_val))

                    self.save_outp_h = np.hstack((np.zeros(65), outp_h, np.zeros(30)))
                    # print('save_outp_h_1', save_outp_h)
                    self.save_outp_e = np.hstack((np.zeros(65), outp_e, np.zeros(30)))
                else:
                    self.save_diff = np.append(self.save_diff, save_diff_val[-1])
                    self.save_sac1 = np.append(self.save_sac1, save_sac1_val[-1])
                    self.save_sac2 = np.append(self.save_sac2, save_sac2_val[-1])

                    self.save_outp_h = np.append(self.save_outp_h, 0)
                    # print('save_outp_h_2', save_outp_h)

                    self.save_outp_h[-60:-30] = self.save_outp_h[-60:-30] + outp_h
                    # print('save_outp_h_3', save_outp_h)

                    self.save_outp_e = np.append(self.save_outp_e, 0)
                    self.save_outp_e[-60:-30] = self.save_outp_e[-60:-30] + outp_e

                a1 = self.save_outp_h[-60]
                a2 = self.save_outp_e[-60]

                # print(a1, a2)
                # print('save_output shape: ', save_outp_h.shape, save_outp_e.shape)
                # print('save_output_h: ', *save_outp_h)
                # print('save_output_e: ', *save_outp_e)
                # print('')

                self.dnn_loop_n += 1
                # print('ApplyAI:', self.dnn_loop_n, i)
                print('i:', i)
                if i == 173:

                    # cp_est_outp_h_tmp = self.save_outp_h.copy()
                    cp_est_outp_e_tmp = self.save_outp_e.copy()
                    #
                    # cp_est_outp_h_tmp = np.array(cp_est_outp_h_tmp)
                    # cp_est_outp_e_tmp = np.array(cp_est_outp_e_tmp)
                    #
                    # indices_above_5_h = np.where(cp_est_outp_h_tmp > 5)
                    # indices_above_5_e = np.where(cp_est_outp_e_tmp > 5)
                    #
                    # for index in indices_above_5_h:
                    #     cp_est_outp_h_tmp[index - 4:index + 4] = 0
                    #
                    # for index in indices_above_5_e:
                    #     cp_est_outp_e_tmp[index - 4:index + 4] = 0
                    #
                    # '''
                    # bpm 계산
                    # '''
                    # for search_i in range(0, len(cp_est_outp_h_tmp)):
                    #
                    #     if cp_est_outp_h_tmp[search_i] == 1:
                    #         bpm_1 = search_i
                    #
                    #         for search_j in range (search_i+1, len(cp_est_outp_h_tmp)):
                    #
                    #             if cp_est_outp_h_tmp[search_j] == 1:
                    #                 bpm_2 = search_j
                    #                 break
                    #
                    #     break
                    #
                    # bpm = round(60 / (bpm_2 - bpm_1) * 0.015)
                    #
                    # '''
                    # lead/lag 판정
                    # '''
                    #
                    # for search_i in range(0, len(cp_est_outp_h_tmp)):
                    #
                    #     if cp_est_outp_h_tmp[search_i] == 1:
                    #
                    #         for search_j in range(0, len(cp_est_outp_e_tmp)):
                    #
                    #             if cp_est_outp_e_tmp[search_j] == 1:
                    #
                    #                 if search_i - search_j > (bpm * 0.3):
                    #                     cp_est_outp_e_tmp[search_j] = 'lead'
                    #                     i = search_i
                    #                     break
                    #                 if search_i - search_j < (bpm * 0.3):
                    #                     cp_est_outp_e_tmp[search_j] = 'lead'
                    #                     i = search_i
                    #                     break

                    save_data = np.vstack((self.save_diff, self.save_sac1, self.save_sac2, self.save_outp_h,
                                           self.save_outp_e, cp_est_outp_e_tmp)).T

                    save_file = fr'C:\Users\user\Desktop\CO_est\ecmo_ai_apply_230509_80_-1_{k}.csv'
                    np.savetxt(save_file, save_data, fmt='%s', delimiter=",")
                    k += 1
                    print('k:', k)

                    self.save_diff = np.array([])
                    self.save_sac1 = np.array([])
                    self.save_sac2 = np.array([])
                    self.save_outp_h = np.array([])
                    self.save_outp_e = np.array([])

                    self.cp_est_outp_h_tmp.clear()
                    self.cp_est_outp_e_tmp.clear()

                    self.evt2.clear()
                    self.evt1.set()





                # ECMO est를 통한 co/counter-pulsation 예측
                # self.cp_est_outp_tmp.clear()
                # cp_est_outp_tmp = self.save_outp_e.copy()
                #
                # for i in range(len(cp_est_outp_tmp)):
                #     if cp_est_outp_tmp[i] >= 3:
                #         cp_est_outp_tmp[i] = 1
                #         for j in range(-9, 10):
                #             if 0 <= i + j < len(cp_est_outp_tmp) and j != 0:
                #                 cp_est_outp_tmp[i + j] = 0
                #     else:
                #         cp_est_outp_tmp[i] = 0
                #
                # # Convert cp_est_outp_tmp to updated_outp_e_array.
                # updated_outp_e_array = np.array(cp_est_outp_tmp)
                #
                # # Check for 1s in updated_outp_e_array and update based on self.save_outp_h.
                # for i in range(len(updated_outp_e_array)):
                #     if updated_outp_e_array[i] == 1:
                #         found_1_in_outp_h = any(map(lambda x: x >= 1, self.save_outp_h[max(0, i - 5):i + 4]))
                #         if found_1_in_outp_h:
                #             updated_outp_e_array[i] = 3
                #         else:
                #             updated_outp_e_array[i] = 4




                # # Heart est를 통한 co/counter-pulsation 예측
                # cp_est_outp_tmp = self.save_outp_h.copy()
                # for j in range(len(cp_est_outp_tmp)):
                #
                #     if cp_est_outp_tmp[i] >= 5:
                #
                #         bpm_1 = i
                #
                #         if cp_est_outp_tmp[i+1] >= 5:
                #
                #             bpm_2 = i
                #
                #     bpm = (bpm_2 - bpm_1)
                #
                #     if cp_est_outp_tmp[i] >= 5 and sum(cp_est_outp_tmp[-5:]) == 0:
                #         found_1_in_sac1 = any(map(lambda x: x >= 1, self.save_outp_h[max(0, i - 14):i + 13]))
                #         found_1_in_sac2 = any(map(lambda x: x >= 1, self.save_outp_h[max(0, i - 14):i + 13]))
                #         if found_1_in_sac1:
                #             cp_est_outp_tmp[i] = 1
                #         else:
                #             cp_est_outp_tmp[i] = 0
                #         if found_1_in_sac2:
                #             cp_est_outp_tmp[i] = 1
                #         else:
                #             cp_est_outp_tmp[i] = 0
                #
                # if self.dnn_loop_n == 1000:
                #
                #     save_data = np.vstack((self.save_diff, self.save_sac1, self.save_sac2, self.save_outp_h,
                #                            self.save_outp_e, cp_est_outp_tmp)).T
                #
                #     np.savetxt(r'C:\Users\user\Desktop\CP_est_data\ecmo_ai_apply_230508_sac.csv', save_data,
                #                fmt='%s', delimiter=",")



class MyProgram:
    def __init__(self):
        lock = threading.Lock()
        sema1 = threading.Semaphore(0)
        evt1 = threading.Event()
        evt2 = threading.Event()

        self.thread1 = SerialReceiver(lock, sema1, evt1)
        self.thread2 = DataPreprocessor(self.thread1, lock, sema1, evt1, evt2)
        self.thread3 = ApplyDNN(self.thread2, lock, evt1, evt2)

    def run(self):
        self.thread1.start()
        self.thread2.start()
        self.thread3.start()

        self.thread1.join()
        self.thread2.join()
        self.thread3.join()


my_program = MyProgram()
my_program.run()
