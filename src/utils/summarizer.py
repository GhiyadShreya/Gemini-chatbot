
from langchain.document_loaders import PyPDFLoader
from utils.utilities import count_num_tokens
from utils.load_config import LoadConfig
import google.generativeai as genai
import os

APPCFG = LoadConfig()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel(model_name="gemini-2.0-flash")


class Summarizer:
    """
    A class for summarizing PDF documents using Google's Gemini model.

    Methods:
        summarize_the_pdf: Summarizes the content of a PDF file using Gemini.
        get_llm_response: Retrieves the Gemini response for a given prompt.
    """

    @staticmethod
    def summarize_the_pdf(
        
        file_dir: str,
        max_final_token: int,
        token_threshold: int,
        gemini_model: str,
        temperature: float,
        summarizer_llm_system_role: str,
        final_summarizer_llm_system_role: str,
        character_overlap: int
    ):
        if not os.path.exists(file_dir) or os.path.getsize(file_dir) == 0:
            raise ValueError(f"The uploaded file '{file_dir}' is empty or does not exist.")

        docs = []
        docs.extend(PyPDFLoader(file_dir).load())
        print(f"Document length: {len(docs)}")
        max_summarizer_output_token = int(max_final_token / len(docs)) - token_threshold
        full_summary = ""
        counter = 1
        print("Generating the summary...")

        if len(docs) > 1:
            for i in range(len(docs)):
                if i == 0:
                    prompt = docs[i].page_content + docs[i + 1].page_content[:character_overlap]
                elif i < len(docs) - 1:
                    prompt = docs[i - 1].page_content[-character_overlap:] + \
                             docs[i].page_content + \
                             docs[i + 1].page_content[:character_overlap]
                else:
                    prompt = docs[i - 1].page_content[-character_overlap:] + docs[i].page_content

                role_prompt = summarizer_llm_system_role.format(max_summarizer_output_token)
                full_summary += Summarizer.get_llm_response(
                    role_prompt,
                    prompt,
                    temperature
                )

                print(f"Page {counter} was summarized. ", end="")
                counter += 1
        else:
            full_summary = docs[0].page_content
            print(f"Page {counter} was summarized. ", end="")
            counter += 1

        print("\nFull summary token length:", count_num_tokens(full_summary))

        final_summary = Summarizer.get_llm_response(
            final_summarizer_llm_system_role,
            full_summary,
            temperature
        )

       

        return final_summary

    @staticmethod
    def get_llm_response(system_role: str, prompt: str, temperature: float) -> str:
        """
        Generates content using Gemini for the provided prompt and system role.

        Args:
            system_role (str): The system instruction to guide the model.
            prompt (str): The input text.
            temperature (float): The temperature for generation.

        Returns:
            str: The generated response.
        """
        try:
            full_prompt = f"{system_role}\n\n{prompt}"
            response = gemini_model.generate_content(full_prompt, generation_config={"temperature": temperature})
            return response.text.strip()
        except Exception as e:
            return f"[Gemini Error] {e}"
