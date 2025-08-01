## Agentic RAG [LlamaIndex + vLLM + Flask]
Story: https://euro-vision.atlassian.net/browse/MLAIHO-63

Python Version: 3.10

### Getting the ENV up
It's a bit tricky to boot up the environemnt as ChromaDB is not behaving well with uv. We have to run the environment setup commands in below order
- `uv venv --python 3.10`
- `source .venv/bin/activate`
- `python -m ensurepip --upgrade`
- `python -m pip install -r requirements.txt`

### Artifacts
- Embedding model is running in a RunPod pod
- LLM model is running in another RunPod pod
- ChromaDB is installed locally

### Model Serving from RunPod Steps
1. Create a Runpod pod
2. Open up 8000 port in the pod configuration
3. Configure vLLM in the pod

    -  UV Install - `curl -LsSf https://astral.sh/uv/install.sh | sh`
    - vLLM Install
        - `uv venv --python 3.12 --seed`
        - `source .venv/bin/activate`
        - `uv pip install vllm --torch-backend=auto`

4. Serve Embedding Model `BAAI/bge-small-en-v1.5` using `nohup vllm serve BAAI/bge-small-en-v1.5`
5. Serve Qwen Model `Qwen/Qwen2.5-Coder-7B-Instruct` using `nohup vllm serve Qwen/Qwen2.5-Coder-7B-Instruct`

### Using vLLM Models
```
llm = MyLLM(
    model_id="Qwen/Qwen2.5-Coder-7B-Instruct"
    , modelProvider='Qwen'
    , inferencePlatformType='vLLM'
    , inferenceNodeSupplier='runpod'
    , runpod_inference_id=os.environ.get("VLLM_LLM_INFERENCE_NODE_ID")
)
embedding_model = MyEmbedder(
    inferenceNodeSupplier='runpod'
    , inferencePlatformType='vllm'
    , runpod_inference_id = os.environ.get('VLLM_EMBEDDING_MODEL_INFERENCE_NODE_ID')
)
```
