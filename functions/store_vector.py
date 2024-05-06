import os
import glob
import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_parse import LlamaParse

from llama_index.vector_stores.chroma import ChromaVectorStore


def init_chroma_store(storage_path, collection_name):
    """
    Initialize the Chroma store.

    Parameters:
        storage_path (str): The path to the storage.
        collection_name (str): The name of the collection.

    Returns:
        StorageContext: The storage context.

    Creates a storage path and collection name for the Chroma store. If the collection does not exist, it will be created.
    ChromaVectoStore is used to store the Chroma vectors in the collection.
    """
    db = chromadb.PersistentClient(storage_path)
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return storage_context


def load_documents(result_type, documents_path):
    """
    Load documents from a given path.

    Parameters:
        result_type (str): The result type for the LlamaParse.
        documents_path (str): The path to the documents.

    Returns:
        dict: The loaded documents.

    The documents are loaded using the SimpleDirectoryReader with the LlamaParse extractor. 
    The result type is used to specify the type of the result. 
    """
    parser = LlamaParse(result_type=result_type)
    file_extractor = {".pdf": parser}
    documents = SimpleDirectoryReader(
        input_files=documents_path,
        file_extractor=file_extractor,
    ).load_data()
    return documents


def read_data_folder(folder_path):
    """
    Read data from a given folder path.

    Parameters:
        folder_path (str): The path to the folder.

    Returns:
        List[str]: The paths to the documents in the folder.

    The function reads the data from the folder path and returns the paths to the documents in the folder.
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"{folder_path} is not a valid folder path.")
    return glob.glob(os.path.join(folder_path, "*.pdf"))
