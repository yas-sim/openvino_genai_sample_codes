# OpenVINO.genai simple sample codes  

|#|File name|Description|
|---|---|---|
|1|openvino_gengi_simple.py|Most simple LLM Q&A sample code|
|2|openvino_genai_simple_streaming.py|Simple LLM Q&A sample code. Streaming output enabled.|
|3|openvino_genai_streamer_test.py|Simple LLM Q&A sample code with vLLM-like iterator-based streaming output enabled.|

## Model preparation

The sample programs above requires LLM model to run.  
You can prepare the LLM model in OpenVINO IR format with `optimum-cli` command in the `optimum-intel` PyPI package.  
You can download the `TinyLlama-1.1B` model with following command.  

```sh
pip install optimum-intel
optimum-cli export openvino --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --trust-remote-code "TinyLlama-1.1B-Chat-v1.0"
```
Please refer to the original [OpenVINO.genai GitHub repo](https://github.com/openvinotoolkit/openvino.genai) for further details.  

