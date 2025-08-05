import chromadb
from rag_utilities.llm.base import MyEmbedder
from rag_utilities.utils import get_app_config
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.chroma import ChromaVectorStore

app_config = get_app_config()
# ALLOWED_DATA_GENRES = ['educational', 'story', 'cooking', 'persona']
ALLOWED_DATA_GENRES = app_config.vector_db.allowed_data_genres

def get_chroma_db():
    return chromadb.PersistentClient(path="./my_local_vector_db")

def get_allowed_data_genres():
    return ALLOWED_DATA_GENRES

def get_chroma_collection(data_genre:str):
    return get_chroma_db().get_or_create_collection(name=data_genre)

def get_chroma_collections():
    return [get_chroma_db().get_or_create_collection(name=genre) for genre in get_allowed_data_genres()]

def get_chroma_vector_store(data_genre:str):
    return ChromaVectorStore(chroma_collection=get_chroma_collection(data_genre))


async def get_embeddded_data(documents, pipeline):
    nodes = list()
    for batch in range(0, len(documents), 200):
        nodes.extend(await pipeline.arun(documents=documents[batch:batch+200]))
        print(f"Processed batch {batch // 200 + 1} => {batch}:{batch + 200}")
        import time; time.sleep(2) # <- Defensive guard against too many parallel embedding requests


async def upload_data_into_vector_db(data_genre:str, embedding_model:MyEmbedder, input_files:list):
    vector_store = get_chroma_vector_store(data_genre)
    reader = SimpleDirectoryReader(input_files=input_files)
    print('Starting to load the data into memory...')
    documents = reader.load_data()
    print(f'Documents read - {len(documents)}')
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(
                chunk_size=512
                , chunk_overlap=40
            ),
            embedding_model
        ],
        vector_store=vector_store,
    )
    print('Converting data into embedded vectors and writing to vector DB...')
    await get_embeddded_data(documents, pipeline)

