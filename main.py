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


try:
    ser = serial.Serial(port, baudrate, timeout=0.5)
    print('serial connect success')
    print(ser)

    ser.write(b'\x80\x73\x73\x8f')  # monitor에 's' 보내기

    R = ''

    class DataPreprocessor(QThread):
        def __init__(self):
            super().__init__()
            self.running = True

        def run(self):
            global R
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
                                    print('ibp_val:', ibp_val)

                                    j_ref = 1
                                    break

                            if j_ref == 0:
                                R = R[i::]
                                break

        def resume(self):
            self.running = True

        def pause(self):
            self.running = False


    class MyWindow(QMainWindow):
        def __init__(self):
            super().__init__()

            self.controller = DataPreprocessor()
            self.controller.start()

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

except serial.serialutil.SerialException:
    print('serial connect error')
