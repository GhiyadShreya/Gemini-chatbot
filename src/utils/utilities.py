#import google.generativeai as genai
import os
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")  # Similar to Gemini

def count_num_tokens(text: str) -> int:
        return len(tokenizer.encode(text))