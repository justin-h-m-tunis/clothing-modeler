import gui_controller
import random

class CallBack():
    def __init__(self, control_instance):
        super().__init__()
        print("called")
        self.control = control_instance
        self.control.update_progress(random.randint(0,100), 100)
        # self.control.create_progress_bar()

    