from llama_index.core import VectorStoreIndex

from functions.store_vector import init_chroma_store, load_documents


def load_db(storage_path, collection_name, result_type, documents_path):
    """
    Load the database.

    Parameters:
        storage_path (str): The path to the storage.
        collection_name (str): The name of the collection.
        result_type (str): The result type for the LlamaParse.
        documents_path (str): The path to the documents.

    Returns:
        Index: The loaded index.

    This function loads and refreshes the database with the latest data from the data folder.
    Reads the data from the data folder and initializes the ChromaDB.
    """
    try:
        storage_context = init_chroma_store(storage_path, collection_name)
        documents = load_documents(result_type, documents_path)
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )
        index.storage_context.persist()
        return index
    except Exception as e:
        if "No such file or directory" in str(e):
            raise Exception("No data found in the data folder")
        raise Exception(str(e))
