import os
from rag_utilities.db import get_allowed_data_genres, get_chroma_vector_store, upload_data_into_vector_db
from rag_utilities.llm import get_models
from rag_utilities.tools import get_all_tools
from llama_index.core.workflow import Context
from llama_index.core import VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent.workflow import (
    AgentWorkflow,
    ReActAgent,
)


# llm, embedding_model = get_models(
#     model_id="gpt-4o-mini"
#     , modelProvider='OpenAI'
# )
llm, embedding_model = get_models(
    model_id="Qwen/Qwen2.5-Coder-7B-Instruct"
    , modelProvider='Qwen'
    , inferencePlatformType='vLLM'
    , inferenceNodeSupplier='runpod'
    , runpod_llm_inference_id=os.environ.get("VLLM_LLM_INFERENCE_NODE_ID")
    , runpod_enbedder_inference_id = os.environ.get('VLLM_EMBEDDING_MODEL_INFERENCE_NODE_ID')
)

async def upload_data(data_genre, files):
    return await upload_data_into_vector_db(data_genre, embedding_model, files)

async def invoke_rag_agent_workflow(user_msg):
    all_tools = get_all_tools(llm, embedding_model)
    data_expert_agent = ReActAgent(
        name="Vector_Data_Query_Agent",
        description="Is able to query vector data",
        system_prompt="""
            A helpful assistant that can use the provided tools to access different genre of data from vector indexes and extract meaningful result.
            You also have access to internet via `internet_search_tool`. Use this tool to answer questions about current affairs.
            But only use `internet_search_tool` if you do not find data in the provided vector indexes.
            If you are using `internet_search_tool` to answer any question, Say so.
            If you do not get an answer from the provided tools, say so.
        """,
        tools=all_tools,
        llm=llm,
    )

    # Create agent configs
    # NOTE: we can use FunctionAgent or ReActAgent here.
    # FunctionAgent works for LLMs with a function calling API.
    # ReActAgent works for any LLM.

    # Create and run the workflow
    agent_wf = AgentWorkflow(
        agents=[data_expert_agent]
    )
    agent_ctx = Context(agent_wf)
    return await agent_wf.run(user_msg=user_msg, ctx=agent_ctx)


if __name__ == '__main__':
    from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec
    # print(help(DuckDuckGoSearchToolSpec.duckduckgo_instant_search))
    # print(DuckDuckGoSearchToolSpec().duckduckgo_full_search(query="Suggest"))