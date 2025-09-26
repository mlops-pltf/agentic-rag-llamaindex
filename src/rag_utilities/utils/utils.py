import os
import yaml
from pydantic import BaseModel, ConfigDict
from typing import List, Literal, Optional, Tuple, Union
from rag_utilities.llm import get_llm, get_embedder
from rag_utilities.llm import RunPodLLamaIndexAgentQwenLLM, RunPodLlamaIndexQwenEmbedding \
    , OpenAI, OpenAIEmbedding


class VectorDBConfig(BaseModel):
    name: str
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

class AppModels(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    llm: Union[
        RunPodLLamaIndexAgentQwenLLM
        , OpenAI
    ]
    embedding_model: Union[
        RunPodLlamaIndexQwenEmbedding
        , OpenAIEmbedding
    ]


def __load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as file:
        config=yaml.safe_load(file)
    return config

def get_app_config():
    declared_config = __load_config()
    embedding_model_name = declared_config['model_selector']['embedding_model']
    llm_name = declared_config['model_selector']['llm']
    app_config = AppConfig(
        vector_db = VectorDBConfig(
            name=declared_config['vector_db']['name'].lower()
            , allowed_data_genres=[genre.lower() for genre in declared_config['vector_db']['allowed_data_genres']]
        )
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
        , retriever = RetrieverConfig(
            top_k=declared_config['retriever']['top_k']
        )
    )
    return app_config


def bootstrap_application_and_models() -> Tuple[AppConfig, AppModels]:
    app_config = get_app_config()
    return app_config, AppModels(
        llm = get_llm(
            model_id=app_config.llm.model_name
            , model_provider=app_config.llm.provider
            , inference_platform_type=app_config.llm.inference_platform
            , inference_node_supplier=app_config.llm.inference_node_supplier
            , runpod_llm_inference_id=os.environ.get("VLLM_LLM_INFERENCE_NODE_ID")
        )
        , embedding_model = get_embedder(
            model_id=app_config.embedding_model.model_name
            , model_provider=app_config.embedding_model.provider
            , inference_platform_type=app_config.embedding_model.inference_platform
            , inference_node_supplier=app_config.embedding_model.inference_node_supplier
            , runpod_enbedder_inference_id = os.environ.get('VLLM_EMBEDDING_MODEL_INFERENCE_NODE_ID')
        )
    )


if __name__ == '__main__':
    print(get_app_config())