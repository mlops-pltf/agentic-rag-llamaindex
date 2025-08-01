from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec
from rag_utilities.db import get_allowed_data_genres, get_chroma_vector_store
from rag_utilities.llm import MyLLM, MyEmbedder
from llama_index.core import VectorStoreIndex

def get_all_tools(llm:MyLLM, embedding_model:MyEmbedder) -> list:
    all_tools = []
    ## Adding internet search tool
    all_tools.append(
            FunctionTool.from_defaults(
            DuckDuckGoSearchToolSpec().duckduckgo_instant_search
            , name=f'internet_search_tool'
            , description='A toll to search queries over the internet'
        )
    )

    for data_genre in get_allowed_data_genres():
        index = VectorStoreIndex.from_vector_store(
            vector_store=get_chroma_vector_store(data_genre)
            , embed_model=embedding_model
        )
        query_engine = index.as_query_engine(llm=llm, similarity_top_k=3) # as shown in the Components in LlamaIndex section

        query_engine_tool = QueryEngineTool.from_defaults(
            query_engine=query_engine,
            name=f'query_engine_tool_for_{data_genre}_data',
            description=f"All data related to {data_genre} data is accesible via this tool",
            return_direct=False,
        )
        all_tools.append(query_engine_tool)

    return all_tools