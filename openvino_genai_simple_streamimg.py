import openvino_genai as ov_genai

def callback(token):
    print(token, end='', flush=True)

model_path = 'TinyLlama-1.1B-Chat-v1.0'
pipe = ov_genai.LLMPipeline(model_path, device='CPU', config={'CACHE_DIR':'./cache'})
pipe.generate('What is sushi?\n', streamer=callback, max_new_tokens=300, temperature=0.9, top_p=0.85)
