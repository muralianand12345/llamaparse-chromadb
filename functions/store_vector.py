import os
import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_parse import LlamaParse

from llama_index.vector_stores.chroma import ChromaVectorStore


def init_chroma_store(storage_path, collection_name):
    db = chromadb.PersistentClient(storage_path)
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return storage_context


def load_documents(result_type, documents_path):
    parser = LlamaParse(result_type=result_type)
    file_extractor = {".pdf": parser}
    documents = SimpleDirectoryReader(
        input_files=documents_path,
        file_extractor=file_extractor,
    ).load_data()
    return documents


def read_data_folder(folder_path):
    documents_path = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            documents_path.append(os.path.join(folder_path, file))
    return documents_path

