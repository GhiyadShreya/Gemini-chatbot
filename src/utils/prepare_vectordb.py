from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from typing import List
import google.generativeai as genai
from google.api_core import retry


class PrepareVectorDB:
    """
    A class for preparing and saving a VectorDB using Gemini embeddings.

    This class facilitates the process of loading documents, chunking them, and creating a VectorDB
    with Gemini embeddings. It provides methods to prepare and save the VectorDB.

    Parameters:
        data_directory (str or List[str]): The directory or list of directories containing the documents.
        persist_directory (str): The directory to save the VectorDB.
        embedding_model_engine (str): The engine for Gemini embeddings.
        chunk_size (int): The size of the chunks for document processing.
        chunk_overlap (int): The overlap between chunks.
    """

    def __init__(
            self,
            data_directory: str,
            persist_directory: str,
            embedding_model_engine: str,
            chunk_size: int,
            chunk_overlap: int
    ) -> None:
        """
        Initialize the PrepareVectorDB instance.

        """

        self.embedding_model_engine = embedding_model_engine
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.data_directory = data_directory
        self.persist_directory = persist_directory
        self.embedding = self.GeminiEmbeddings()


    class GeminiEmbeddings:
        """
        Custom embedding class using Gemini's embedding model.
        """
        def __init__(self):
            self.model = "models/embedding-001"
            
        @retry.Retry()
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            """Embed multiple texts using Gemini."""
            return [self.embed_query(text) for text in texts]
            
        @retry.Retry()
        def embed_query(self, text: str) -> List[float]:
            """Embed a single text using Gemini."""
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document"
            )
            return result["embedding"]

    def __load_all_documents(self) -> List:
        """
        Load all documents from the specified directory or directories.

        Returns:
            List: A list of loaded documents.
        """
        doc_counter = 0
        if isinstance(self.data_directory, list):
            print("Loading the uploaded documents...")
            docs = []
            for doc_dir in self.data_directory:
                docs.extend(PyMuPDFLoader(doc_dir).load())

                doc_counter += 1
            print("Number of loaded documents:", doc_counter)
            print("Number of pages:", len(docs), "\n\n")
        else:
            print("Loading documents manually...")
            document_list = os.listdir(self.data_directory)
            docs = []
            for doc_name in document_list:
                docs.extend(PyPDFLoader(os.path.join(
                    self.data_directory, doc_name)).load())
                doc_counter += 1
            print("Number of loaded documents:", doc_counter)
            print("Number of pages:", len(docs), "\n\n")

        return docs

    def __chunk_documents(self, docs: List) -> List:
        """
        Chunk the loaded documents using the specified text splitter.

        Parameters:
            docs (List): The list of loaded documents.

        Returns:
            List: A list of chunked documents.

        """
        print("Chunking documents...")
        chunked_documents = self.text_splitter.split_documents(docs)
        print("Number of chunks:", len(chunked_documents), "\n\n")
        return chunked_documents

    def prepare_and_save_vectordb(self):
        """
        Load, chunk, and create a VectorDB with Gemini embeddings, and save it.

        Returns:
            Chroma: The created VectorDB.
        """
        docs = self.__load_all_documents()
        chunked_documents = self.__chunk_documents(docs)
        print("Preparing vectordb...")
        vectordb = Chroma.from_documents(
            documents=chunked_documents,
            embedding=self.embedding,
            persist_directory=self.persist_directory
        )
        print("VectorDB is created and saved.")
        print("Number of vectors in vectordb:",
              vectordb._collection.count(), "\n\n")
        return vectordb
