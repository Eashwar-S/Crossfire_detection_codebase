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