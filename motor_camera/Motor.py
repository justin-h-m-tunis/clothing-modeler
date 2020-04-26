import time
import serial
import gui_controller


class Motor:
    def __init__(self, macrostep_time, total_macrosteps, baudrate, com, onSerialFail=lambda: None):
        #sets delay to be the total time taken for one step
        self.delay = macrostep_time
        #sets total macrosteps
        self.total = total_macrosteps
        #opens serial communication on com at baudrate
        try:
            self.ser = serial.Serial(port=com,baudrate=baudrate)
        except serial.SerialException:
            self.ser = None
            onSerialFail()

    # Completes n macrosteps of the motor, checks cond before each step
    # it is a blocking call, so should be parellelized with cond = camera.captureRGBD()
    def macrostep(self, n=1, cond = lambda x: True):
        curr_step = 0
        for i in range(n):
            t = time.process_time_ns()
            while (time.process_time_ns() - t) < self.delay*1e6 or not cond(i):
                pass
            if not self.ser is None:
                print("stepped!")
                curr_step += 1
                gui_controller.update_progress(curr_step, self.total)
                self.ser.write('1'.encode())
                self.ser.write('\n'.encode())

    def fullRotation(self, cond=lambda: True):  # Executes a full rotation of the motor
        self.macrostep(self.total,cond)
