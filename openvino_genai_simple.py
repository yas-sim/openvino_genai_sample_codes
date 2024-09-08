import openvino_genai as ov_genai

# optimum-cli export openvino --trust-remote-code --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 TinyLlama-1.1B-Chat-v1.0
model_path = 'TinyLlama-1.1B-Chat-v1.0'
pipe = ov_genai.LLMPipeline(model_path, device='CPU', config={'CACHE_DIR':'./cache'})
res = pipe.generate('What is sushi?\n', max_new_tokens=300, temperature=0.9, top_p=0.85)
print(res)