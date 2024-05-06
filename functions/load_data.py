from llama_index.core import VectorStoreIndex

from functions.store_vector import init_chroma_store, load_documents

def load_db(storage_path, collection_name, result_type, documents_path):
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
