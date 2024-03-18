import threading

class local_QA():
    def __init__(self, N):
        self.deleted = False
        self.ready = False
        self.lock = threading.Lock()


    def delete(self):
        with self.lock:
            self.deleted = True
            self.ready = False


    def do_local_QA(self):
        pass
