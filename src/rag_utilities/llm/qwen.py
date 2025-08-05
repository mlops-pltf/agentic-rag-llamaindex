import os, aiohttp
from typing import List, Any
from llama_index.core.llms.llm import LLM
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import ChatMessage, ChatResponse, LLMMetadata, CompletionResponse  # Sometimes required



## ChatResponse with __aiter__ and __anext__
## Note how __aiter__ is synchronous and __anext__ is asynchronous
class QwenChatResponse(ChatResponse):
    def __aiter__(self):
        return self

    async def __anext__(self):
        ## Not a very good __anext__ implementation
        ## Fix it in future
        if not hasattr(self, "_has_yielded"):
            self._has_yielded = True
            return self
        else:
            raise StopAsyncIteration


## We are implementing new class for Agent because
## In agent call .achat is expected to return a ChatResponse object
## That has to have __aiter__ and __anext__ implemented
class RunPodLLamaIndexAgentQwenLLM(LLM):
    api_url: str  # Pydantic field
    overriden_model_name: str  # Pydantic field

    def __init__(self, api_url: str, overriden_model_name: str, **kwargs):
        super().__init__(api_url=api_url, overriden_model_name=overriden_model_name, **kwargs)
        self.overriden_model_name = overriden_model_name

    # Async chat: core method for LlamaIndex RAG
    async def achat(
        self, messages: List[ChatMessage], **kwargs
    ) -> QwenChatResponse:
        # Prepare payload
        payload = {
            # "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "model": self.overriden_model_name,
            "messages": [
                {"role": m.role, "content": m.content} for m in messages
            ],
            "temperature": 0.7,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, json=payload) as resp:
                result = await resp.json()
                output_text = result["choices"][0]["message"]["content"]
                return QwenChatResponse(message=ChatMessage(role="assistant", content=output_text))

    async def astream_chat(self, *args, **kwargs) -> Any:
        # print("Called with args:", args, "kwargs:", kwargs)
        if args:
            messages = args[0]
        else:
            messages = kwargs.get("messages", None)
        if messages is None:
            raise ValueError("astream_chat requires 'messages' as first arg or keyword.")
        # print("Messages for model:", messages)
        resp = await self.achat(messages, **kwargs)
        print(f'Model response: {resp.message}')
        return resp

    async def astream_complete(self, prompt: str, **kwargs) -> Any:
        raise NotImplementedError()

    async def chat(self, messages: List[ChatMessage], **kwargs) -> str:
        raise NotImplementedError()

    def complete(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError()

    def stream_chat(self, messages: List[ChatMessage], **kwargs) -> Any:
        raise NotImplementedError()

    def stream_complete(self, prompt: str, **kwargs) -> Any:
        raise NotImplementedError()

    async def acomplete(self, prompt: str, **kwargs) -> CompletionResponse:
        payload = {
            # "model": "Qwen/Qwen2.5-Coder-7B-Instruct",   # Adjust model name as needed
            "model": self.overriden_model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.95,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, json=payload) as resp:
                result = await resp.json()
                output_text = result["choices"][0]["message"]["content"]
                return CompletionResponse(
                    text=output_text,
                    usage=result.get("usage", {}),
                    model=self.metadata.model_name
                )
    
    async def _get_query_embedding(self, query: str):
        # Simple solution: use asyncio to run async in sync context (not efficient, but unblocks you)
        import asyncio
        return asyncio.run(self._aget_query_embedding(query))

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            name="RunPodQwenLLM",
            description="RunPod Qwen LLM for chat completions",
            model_name=self.overriden_model_name,  # Adjust model name as needed
            max_input_size=4096,  # Adjust based on the model's capabilities
            max_output_tokens=1024,  # Adjust based on your needs
            context_window=2048,  # Adjust based on your needs
        )


class RunPodLlamaIndexQwenEmbedding(BaseEmbedding):
    endpoint_url: str

    def __init__(self, endpoint_url: str, **kwargs):
        super().__init__(endpoint_url=endpoint_url, **kwargs)

    async def _aget_text_embedding(self, text: str):
        async with aiohttp.ClientSession() as session:
            payload = {"input": text}
            async with session.post(self.endpoint_url, json=payload) as resp:
                result = await resp.json()
                return result["data"][0]["embedding"]

    def _get_text_embedding(self, text: str):
        raise NotImplementedError("Sync embedding not implemented.")

    async def _aget_query_embedding(self, query: str):
        return await self._aget_text_embedding(query)

    def _get_query_embedding(self, query: str):
        raise NotImplementedError("Sync embedding not implemented.")
