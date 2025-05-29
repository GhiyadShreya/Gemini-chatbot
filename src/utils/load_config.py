
import google.generativeai as genai
import os
from dotenv import load_dotenv
import yaml
from pyprojroot import here
import shutil
from typing import Callable, List

load_dotenv()

class GeminiEmbeddings:
    """Embedding class compatible with ChromaDB"""
    def __init__(self):
        self.model = "models/embedding-001"
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        result = genai.embed_content(
            model=self.model,
            content=text,
            task_type="retrieval_query"
        )
        return result["embedding"]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        return [self.embed_query(text) for text in texts]

class LoadConfig:
    """
    Methods:
        load_gemini_cfg():
            Load Gemini configuration settings.
        create_directory(directory_path):
            Create a directory if it does not exist.
        remove_directory(directory_path):
            Removes the specified directory.
    """

    def __init__(self) -> None:
        with open(here("configs/app_config.yml")) as cfg:
            app_config = yaml.load(cfg, Loader=yaml.FullLoader)

        # LLM configs
        self.llm_engine = app_config["llm_config"]["engine"]
        self.llm_system_role = app_config["llm_config"]["llm_system_role"]
        self.persist_directory = str(here(
            app_config["directories"]["persist_directory"]))  # needs to be strin for summation in chromadb backend: self._settings.require("persist_directory") + "/chroma.sqlite3"
        self.custom_persist_directory = str(here(
            app_config["directories"]["custom_persist_directory"]))
        self.embedding_model = self.get_gemini_embeddings

        # Retrieval configs
        self.data_directory = app_config["directories"]["data_directory"]
        self.k = app_config["retrieval_config"]["k"]
        self.embedding_model_engine = app_config["embedding_model_config"]["engine"]
        self.chunk_size = app_config["splitter_config"]["chunk_size"]
        self.chunk_overlap = app_config["splitter_config"]["chunk_overlap"]

        # Summarizer config
        self.max_final_token = app_config["summarizer_config"]["max_final_token"]
        self.token_threshold = app_config["summarizer_config"]["token_threshold"]
        self.summarizer_llm_system_role = app_config["summarizer_config"]["summarizer_llm_system_role"]
        self.character_overlap = app_config["summarizer_config"]["character_overlap"]
        self.final_summarizer_llm_system_role = app_config[
            "summarizer_config"]["final_summarizer_llm_system_role"]
        self.temperature = app_config["llm_config"]["temperature"]

        # Memory
        self.number_of_q_a_pairs = app_config["memory"]["number_of_q_a_pairs"]

        # Load OpenAI credentials
        self.load_gemini_cfg()
        self.embedding_model = GeminiEmbeddings()

        # clean up the upload doc vectordb if it exists
        self.create_directory(self.persist_directory)
        self.remove_directory(self.custom_persist_directory)

    def load_gemini_cfg(self):
        """Configure Gemini API with the API key from environment variables."""
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    def get_gemini_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using Gemini's embedding model."""
        if isinstance(texts, str):
            texts = [texts]
        
        # Initialize the model
        model = "models/embedding-001"
        
        # Get embeddings for each text
        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model=model,
                content=text,
                task_type="retrieval_document"  # or "retrieval_query"
            )
            embeddings.append(result["embedding"])
        
        return embeddings

    def create_directory(self, directory_path: str):
        """
        Create a directory if it does not exist.

        Parameters:
            directory_path (str): The path of the directory to be created.
        """
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    def remove_directory(self, directory_path: str):
        """
        Removes the specified directory.

        Parameters:
            directory_path (str): The path of the directory to be removed.

        Raises:
            OSError: If an error occurs during the directory removal process.

        Returns:
            None
        """
        if os.path.exists(directory_path):
            try:
                shutil.rmtree(directory_path)
                print(
                    f"The directory '{directory_path}' has been successfully removed.")
            except OSError as e:
                print(f"Error: {e}")
        else:
            print(f"The directory '{directory_path}' does not exist.")
