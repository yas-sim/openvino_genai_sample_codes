from ov_genai_utils.streamer import *

model_path = 'TinyLlama-1.1B-Chat-v1.0'

ovs = OpenVINO_GenAI_Streamer(model_path, device='CPU', config={'CACHE_DIR':'./cache'})
ovs.generate(prompt='What is sushi?\n', max_new_tokens=300, temperature=0.9, top_p=0.85)
for token in ovs:
    print(token, end='', flush=True)
