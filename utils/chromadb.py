import chromadb
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_parse import LlamaParse
from llama_index.vector_stores.chroma import ChromaVectorStore


class StoreVector:
    def __init__(self, storage_path, collection_name, result_type, documents_path):
        self.storage_path = storage_path
        self.collection_name = collection_name
        self.result_type = result_type
        self.documents_path = documents_path

    def init_chroma_store(self):
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
        db = chromadb.PersistentClient(self.storage_path)
        chroma_collection = db.get_or_create_collection(self.collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return storage_context

    def load_documents(self):
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
        parser = LlamaParse(result_type=self.result_type)
        file_extractor = {".pdf": parser}
        documents = SimpleDirectoryReader(
            input_files=self.documents_path,
            file_extractor=file_extractor,
        ).load_data()
        return documents


class LoadData(StoreVector):
    def load_db(self):
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
            storage_context = self.init_chroma_store()
            documents = self.load_documents()
            index = VectorStoreIndex.from_documents(
                documents, storage_context=storage_context
            )
            index.storage_context.persist()
            return index
        except Exception as e:
            if "No such file or directory" in str(e):
                raise Exception("No data found in the data folder")
            raise Exception(str(e))


class QuerySearch:
    def __init__(self, storage_path, collection_name):
        self.storage_path = storage_path
        self.collection_name = collection_name

    def load_index(self):
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
        db = chromadb.PersistentClient(path=self.storage_path)
        chroma_collection = db.get_or_create_collection(self.collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=self.storage_path
        )
        index = load_index_from_storage(storage_context=storage_context)
        return index
