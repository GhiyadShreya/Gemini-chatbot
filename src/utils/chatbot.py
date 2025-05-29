import gradio as gr
import time
import google.generativeai as genai
import os
from langchain.vectorstores import Chroma
from typing import List, Tuple
import re
import ast
import html
from utils.load_config import LoadConfig

APPCFG = LoadConfig()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


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

class ChatBot:
    """
    Class representing a chatbot with document retrieval and response generation capabilities.

    This class provides static methods for responding to user queries, handling feedback, and
    cleaning references from retrieved documents.
    """
    @staticmethod
    def initialize_vectordb(persist_dir: str):
        """Initialize ChromaDB with proper error handling"""
        try:
            embedding_function = GeminiEmbeddings()
            return Chroma(
                persist_directory=persist_dir,
                embedding_function=embedding_function
            )
        except Exception as e:
            print(f"Error initializing ChromaDB: {str(e)}")
            # Try to recover by recreating the vectorstore
            # if os.path.exists(persist_dir):
            #     import shutil
            #     shutil.rmtree(persist_dir)
            return None
        

    @staticmethod
    def strip_code_blocks(text: str) -> str:
        """
        Remove Markdown code blocks and leading/trailing backticks from LLM outputs.
        """
        import re
        # Remove triple backticks and any surrounding language specifier
        text = re.sub(r"```(?:[\w+-]*)\n?", "", text)
        text = text.replace("```", "")
        return text.strip()

        

    @staticmethod
    def respond(chatbot: List, message: str, data_type: str = "Preprocessed doc", temperature: float = 0.0) -> Tuple:
        model = genai.GenerativeModel('gemini-2.0-flash')

        if data_type == "Preprocessed doc":
            if os.path.exists(APPCFG.persist_directory):
                vectordb = Chroma(persist_directory=APPCFG.persist_directory,
                                embedding_function=APPCFG.embedding_model)
            else:
                chatbot.append((message, f"VectorDB does not exist. Please first execute the 'upload_data_manually.py' module."))
                return "", chatbot, None

        elif data_type == "Upload doc: Process for RAG":
            if os.path.exists(APPCFG.custom_persist_directory):
                vectordb = Chroma(persist_directory=APPCFG.custom_persist_directory,
                                embedding_function=APPCFG.embedding_model)
            else:
                chatbot.append((message, "No file was uploaded. Please first upload your files using the 'upload' button."))
                return "", chatbot, None

        docs = vectordb.similarity_search(message, k=APPCFG.k)

        question = "# User new question:\n" + message
        retrieved_content = ChatBot.clean_references(docs)

        chat_history = f"Chat history:\n {str(chatbot[-APPCFG.number_of_q_a_pairs:])}\n\n"
        #prompt = f"{chat_history}{retrieved_content}\n{question}"
        prompt = f"{chat_history}{retrieved_content}\n{question}"

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=temperature)
        )

        cleaned_text = ChatBot.strip_code_blocks(response.text)
        print(response.text)
        final_response = f"{cleaned_text}\n\n---"


        chatbot.append({"role": "user", "content": message})
        chatbot.append({"role": "assistant", "content": final_response})

        time.sleep(2)

        # 🟢 Return references as a nicely formatted string
        references = f"**References:**\n\n{retrieved_content}"
        return "", chatbot, references

   

    @staticmethod
     
    def clean_references(documents: List) -> str:
        """Format document references safely"""
        markdown = []
        for i, doc in enumerate(documents, 1):
            try:
                content = doc.page_content
                metadata = getattr(doc, 'metadata', {})
                
                # Basic cleaning
                content = re.sub(r'\s+', ' ', content).strip()
                content = html.unescape(content)
                
                # Format reference
                ref = f"### Document {i}\n{content}\n"
                if metadata:
                    ref += f"Source: {metadata.get('source', 'unknown')}\n"
                    ref += f"Page: {metadata.get('page', 'unknown')}\n"
                markdown.append(ref)
                
            except Exception as e:
                print(f"Error formatting document {i}: {str(e)}")
                continue
                
        return "\n\n".join(markdown) if markdown else "No references available"
