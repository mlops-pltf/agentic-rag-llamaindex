import os
from jinja2 import Template
from rag_utilities.tools import get_all_tools
from rag_utilities.utils import AppModels
from llama_index.core.workflow import Context
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent.workflow import (
    AgentWorkflow,
    ReActAgent,
)
from rag_utilities.prompt import VECTOR_DATA_QUERY_AGENT_SYSTEM_PROMPT_TEMPLATE

if __name__ == '__main__':
    os.environ["VLLM_LLM_INFERENCE_NODE_ID"] = "dummy"
    os.environ["VLLM_EMBEDDING_MODEL_INFERENCE_NODE_ID"] = "dummy"

async def invoke_rag_agent_workflow(user_msg, app_models: AppModels):
    all_tools = get_all_tools(app_models.llm, app_models.embedding_model)
    VECTOR_DATA_QUERY_AGENT_SYSTEM_PROMPT = Template(
        VECTOR_DATA_QUERY_AGENT_SYSTEM_PROMPT_TEMPLATE
    ).render(
        tools=[tool for tool in all_tools if type(tool) == QueryEngineTool]
    )
    data_expert_agent = ReActAgent(
        name="Vector_Data_Query_Agent",
        description="Is able to query vector data",
        system_prompt=VECTOR_DATA_QUERY_AGENT_SYSTEM_PROMPT,
        tools=all_tools,
        llm=app_models.llm,
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
    # from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec
    # print(help(DuckDuckGoSearchToolSpec.duckduckgo_instant_search))
    # print(DuckDuckGoSearchToolSpec().duckduckgo_full_search(query="Suggest"))
    # print(llm)
    # print(embedding_model)
    # from jinja2 import Template
    # from rag_utilities.tools import get_all_tools
    # from rag_utilities.agent.agents import llm, embedding_model
    from rag_utilities.utils import get_app_config
    from rag_utilities.llm import get_llm, get_embedder

    app_config = get_app_config()
    llm = get_llm(
        model_id=app_config.llm.model_name
        , model_provider=app_config.llm.provider
        , inference_platform_type=app_config.llm.inference_platform
        , inference_node_supplier=app_config.llm.inference_node_supplier
        , runpod_llm_inference_id='dummy'
    )
    embedding_model = get_embedder(
        model_id=app_config.embedding_model.model_name
        , model_provider=app_config.embedding_model.provider
        , inference_platform_type=app_config.embedding_model.inference_platform
        , inference_node_supplier=app_config.embedding_model.inference_node_supplier
        , runpod_enbedder_inference_id = 'dummy'
    )
    all_tools = get_all_tools(llm, embedding_model)
    # for tool in all_tools:
    #     if type(tool) == QueryEngineTool:
    #         print(tool.metadata.name)
    #         print(tool.metadata.description)
    VECTOR_DATA_QUERY_AGENT_SYSTEM_PROMPT = Template(
        VECTOR_DATA_QUERY_AGENT_SYSTEM_PROMPT_TEMPLATE
    ).render(
        tools=[tool for tool in all_tools if type(tool) == QueryEngineTool]
    )
    print(VECTOR_DATA_QUERY_AGENT_SYSTEM_PROMPT)