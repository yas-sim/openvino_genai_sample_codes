import openvino_genai as ov_genai
import threading

class OpenVINO_GenAI_Streamer:
    def __init__(self, model_path, device='CPU', config={}):
        self.pipe = ov_genai.LLMPipeline(model_path, device=device, config=config)
        self.queue = []
        self.queue_lock = threading.Lock()
        self.llm_thread = None
        self.abort_flag = False
    
    def callback(self, token):
        self.queue_lock.acquire()
        self.queue.append(token)
        self.queue_lock.release()
        return self.abort_flag

    def __iter__(self):
        return self
    
    def __next__(self):
        while True:
            self.queue_lock.acquire()
            queue_len = len(self.queue)
            if queue_len > 0:
                break
            self.queue_lock.release()
            if self.llm_thread is not None:
                if self.llm_thread.is_alive() == False:
                    self.llm_thread.join()
                    self.llm_thread = None
                    raise StopIteration
        token = self.queue.pop(0)
        self.queue_lock.release()
        return token

    def generate(self, prompt, **kwargs):
        self.abort_flag = False
        kwargs['streamer'] = self.callback
        self.llm_thread = threading.Thread(target=self.pipe.generate, args=(prompt, ), kwargs=kwargs)
        self.llm_thread.start()

    def stop_generation(self):
        self.abort_flag = True
