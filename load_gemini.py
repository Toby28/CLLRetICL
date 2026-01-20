import os
from dotenv import load_dotenv
from google.api_core import retry
from tqdm import tqdm
import numpy as np
import google.generativeai as genai
import time

class gemini():
    def __init__(self,model_name = "gemini-1.5-flash"):
        load_dotenv(dotenv_path='api.env')
        self.GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
        self.gptmodel_name = model_name

    def create_gptmodel(self):
        generation_config = {
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 2,
            "max_output_tokens": 5,
            "response_mime_type": "text/plain",
        }

        model = genai.GenerativeModel(
            model_name=self.gptmodel_name,
            generation_config=generation_config,
            system_instruction="You are a helpful and accurate assistant. You will be provided with a sentence, and your task is to classify as label_0 or label_1(choose either the label_0 or the label_1 tag but NOT both)."
            # system_instruction="You are a helpful and accurate assistant. You will be provided with a sentence, and your task is to classify as label_0 or label_1(choose either the label_0 or the label_1 tag but NOT both)."
            # safety_settings = Adjust safety settings
            # See https://ai.google.dev/gemini-api/docs/safety-settings
        )

        return model

    def gen_ICL_results(self,prompt, max_retries=2, delay=2):

        model = self.create_gptmodel(self.gptmodel_name)
        retry_count = 0
        # print(prompt)

        while retry_count < max_retries:
            try:
                output = model.generate_content(prompt)
                return output.text
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(delay)
                retry_count += 1

    def build_prompt(self,text, df0, group0, group1):
        prompt = build_prompt(group0, 0, "", df0, n) + build_prompt(group1, 1, text, df0, n)
