import yaml
from pydantic import BaseModel
from typing import List, Literal, Optional


class VectorDBConfig(BaseModel):
    allowed_data_genres: List[str] = [
        "educational"
        , "story"
        , "cooking"
        , "persona"
    ]

class EmbeddingModelConfig(BaseModel):
    provider: Literal['baai', 'openai']
    model_name: str = 'text-embedding-3-small'
    # Literal[
    #     'baai/bge-small-en-v1.5'
    #     , 'text-embedding-3-small'
    # ]
    inference_platform: Optional[Literal['vllm', 'ollama', 'openai', 'google']] = None
    inference_node_supplier: Optional[Literal['runpod', 'ec2', 'openai', 'google']] = None

class LLMConfig(BaseModel):
    provider: Literal['qwen', 'openai', 'groq', 'google']
    model_name: str = 'gpt-4.1'
    # Literal[
    #     'qwen/qwen2.5-coder-7b-instruct'
    #     , 'openai:gpt-4.1'
    # ]
    inference_platform: Optional[Literal['vllm', 'ollama', 'openai', 'google']] = None
    inference_node_supplier: Optional[Literal['runpod', 'ec2', 'openai', 'google']] = None
    temperature: Optional[int] = 0
    max_output_tokens: Optional[int] = 2048

class RetrieverConfig(BaseModel):
    top_k: Optional[int] = 3

class AppConfig(BaseModel):
    vector_db: VectorDBConfig
    embedding_model: EmbeddingModelConfig
    llm: LLMConfig
    retriever: RetrieverConfig


def __load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as file:
        config=yaml.safe_load(file)
    return config

def get_app_config():
    declared_config = __load_config()
    embedding_model_name = declared_config['model_selector']['embedding_model']
    llm_name = declared_config['model_selector']['llm']
    app_config = AppConfig(
        vector_db = VectorDBConfig()
        , embedding_model = EmbeddingModelConfig(
            provider=declared_config['embedding_model'][embedding_model_name]['provider'].lower()
            , model_name=declared_config['embedding_model'][embedding_model_name]['model_name']
            , inference_platform=declared_config['embedding_model'][embedding_model_name]['inference_platform'].lower()
            , inference_node_supplier=declared_config['embedding_model'][embedding_model_name]['inference_node_supplier'].lower()
        )
        , llm = LLMConfig(
            provider=declared_config['llm'][llm_name]['provider'].lower()
            , model_name=declared_config['llm'][llm_name]['model_name']
            , inference_platform=declared_config['llm'][llm_name]['inference_platform'].lower()
            , inference_node_supplier=declared_config['llm'][llm_name]['inference_node_supplier'].lower()
        )
        , retriever = RetrieverConfig()
    )
    return app_config


if __name__ == '__main__':
    print(get_app_config())