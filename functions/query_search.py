import chromadb
from llama_index.core import load_index_from_storage, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore


def load_index(storage_path, collection_name):
    """
    Load the index from the storage.

    Parameters:
        storage_path (str): The path to the storage.
        collection_name (str): The name of the collection.

    Returns:
        Index: The loaded index.

    Gets the ChromaDB collection and vector store from the storage path. 
    Loads the index from the storage context.
    """
    db = chromadb.PersistentClient(path=storage_path)
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir=storage_path
    )
    index = load_index_from_storage(storage_context=storage_context)
    return index
