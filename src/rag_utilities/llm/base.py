import os
from rag_utilities.llm.qwen import RunPodLlamaIndexQwenEmbedding, RunPodLLamaIndexAgentQwenLLM
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

## Implement Pydantic Check in this class
class MyLLM:
    def __new__(
        cls
        , model_id
        , model_provider:str='openai'
        , inference_node_supplier:str='runpod'
        , inference_platform_type:str='vllm'
        , **kwargs
    ):
        if model_provider.lower() == 'qwen':
            if inference_node_supplier.lower() == 'runpod':
                if inference_platform_type.lower() == 'vllm':
                    if kwargs.get('runpod_llm_inference_id', 'Not provided') == 'Not provided':
                        raise TypeError("MyLLM object is missing one keyword argument: 'runpod_llm_inference_id'")
                    elif not kwargs.get('runpod_llm_inference_id', 'Not provided'):
                        raise TypeError("'runpod_llm_inference_id' can not be None, Provide proper value for this argument.")
                    _runpod_llm_inference_id=kwargs['runpod_llm_inference_id']
                    _vllm_api_base = f"https://{_runpod_llm_inference_id}-8000.proxy.runpod.net/v1/chat/completions"
                    return RunPodLLamaIndexAgentQwenLLM(
                        api_url=_vllm_api_base
                        , overriden_model_name=model_id
                    )
                else:
                    raise NotImplementedError(f"The interface class for '{inference_platform_type}'-'{model_provider}' model running on '{inference_node_supplier}' is not implemented yet.")
            else:
                raise NotImplementedError(f"The interface class for '{inference_platform_type}'-'{model_provider}' model running on '{inference_node_supplier}' is not implemented yet.")
        elif model_provider.lower() == 'openai':
            if not os.environ.get('OPENAI_API_KEY', None):
                raise Exception(f"`OPENAI_API_KEY` must be declared as an environment variable to use any OpenAI Model..")
            return OpenAI(model=model_id)
        else:
            raise NotImplementedError(f"The interface class for '{inference_platform_type}'-'{model_provider}' model running on '{inference_node_supplier}' is not implemented yet.")


class MyEmbedder:
    def __new__(
        cls
        , model_id
        , model_provider:str='openai'
        , inference_node_supplier:str='runpod'
        , inference_platform_type:str='vllm'
        , **kwargs
    ):
        if model_provider.lower() == 'openai':
            if not os.environ.get('OPENAI_API_KEY', None):
                raise Exception(f"`OPENAI_API_KEY` must be declared as an environment variable to use any OpenAI Model..")
            return OpenAIEmbedding(model=model_id)
        else:
            if inference_node_supplier.lower() == 'runpod':
                if inference_platform_type.lower() == 'vllm':
                    if kwargs.get('runpod_enbedder_inference_id', 'Not provided') == 'Not provided':
                        raise TypeError("MyEmbedder object is missing one keyword argument: 'runpod_enbedder_inference_id'")
                    elif not kwargs.get('runpod_enbedder_inference_id', 'Not provided'):
                        raise TypeError("'runpod_enbedder_inference_id' can not be None, Provide proper value for this argument.")
                    _runpod_enbedder_inference_id=kwargs['runpod_enbedder_inference_id']
                    _vllm_api_base = f"https://{_runpod_enbedder_inference_id}-8000.proxy.runpod.net/v1/embeddings"
                    return RunPodLlamaIndexQwenEmbedding(
                        endpoint_url=_vllm_api_base
                    )
                else:
                    raise NotImplementedError(f"The interface class for '{inference_platform_type}' embedding model running on '{inference_node_supplier}' is not implemented yet.")
            else:
                raise NotImplementedError(f"The interface class for '{inference_platform_type}' embedding model running on '{inference_node_supplier}' is not implemented yet.")


def get_llm(
    model_id="gpt-4o-mini"
    , model_provider='openai'
    , inference_platform_type=''
    , inference_node_supplier=''
    , runpod_llm_inference_id = None
):
    return MyLLM(
        model_id = model_id
        , model_provider = model_provider
        , inference_platform_type = inference_platform_type
        , inference_node_supplier = inference_node_supplier
        , runpod_llm_inference_id = runpod_llm_inference_id
    )

def get_embedder(
    model_id="gpt-4o-mini"
    , model_provider='OpenAI'
    , inference_platform_type=''
    , inference_node_supplier=''
    , runpod_enbedder_inference_id = None
):
    return MyEmbedder(
        model_id=model_id
        , model_provider = model_provider
        , inference_platform_type = inference_platform_type
        , inference_node_supplier = inference_node_supplier
        , runpod_enbedder_inference_id = runpod_enbedder_inference_id
    )
