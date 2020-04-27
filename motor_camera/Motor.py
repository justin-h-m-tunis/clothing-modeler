import time
import serial


class Motor:
    def __init__(self, macrostep_time, total_macrosteps, baudrate, com, onSerialFail=lambda: None):
        #sets delay to be the total time taken for one step
        self.delay = macrostep_time
        #sets total macrosteps
        self.total = total_macrosteps
        #opens serial communication on com at baudrate
        while True:
            try:
                self.ser = serial.Serial(port=com,baudrate=baudrate)
            except serial.SerialException:
                self.ser = None
                onSerialFail()
                time.sleep(3) # wait for 3 seconds and retry
                continue
            break
        print("Connection success!")

    # Completes n macrosteps of the motor, checks cond before each step
    # it is a blocking call, so should be parellelized with cond = camera.captureRGBD()
    def macrostep(self, n=1, cond = lambda n: True, updateFn=lambda n: None):
        for i in range(n):
            t = time.process_time_ns()
            while (time.process_time_ns() - t) < self.delay*1e6 or not cond(i):
                pass
            if not self.ser is None:
                updateFn(i)
                self.ser.write('1'.encode())
                self.ser.write('\n'.encode())

    def fullRotation(self, cond=lambda n: True, updateFn=lambda n: print("Step "+str(n) + " completed!")):  # Executes a full rotation of the motor
        self.macrostep(self.total,cond=cond,updateFn=updateFn)