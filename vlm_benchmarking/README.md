# VLM Benchmaking

### Environment Setup

Create a .env file inside the src/ directory to store your Hugging Face API token:

src/.env
```
HF_TOKEN=your_huggingface_token_here
```

The script automatically loads this token for model access.

### Running the Benchmark
```
cd vlm_benchmarking/src
python3 vlm_auto.py
```

### Running with VLLM and Openai endpoint
1. Run the docker container first in a separate terminal and do not change this terminal
```
docker run --runtime nvidia --gpus all     -v ~/.cache/huggingface:/root/.cache/huggingface     --env "HF_TOKEN=$ENTER YOUR HUGGING FACE TOKEN HERE$"     -p 8000:8000  --env "TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas"    --ipc=host     vllm/vllm-openai:latest     --model leon-se/ForestFireVLM-3B

```

2. Run the python file after sourcing into the python virtual environment:
```
cd vlm_benchmarking/src
python3 vlm_auto_openai_vllm.py
```