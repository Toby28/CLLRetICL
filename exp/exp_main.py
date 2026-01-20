from datashader import max_n

from exp.exp_basic import Exp_Basic
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import matplotlib.pyplot as plt
import numpy as np
from encoder import embedding_encoder, embedding_encoder_onetime
from load_dataset import load_dataset,load_embedding_dataset
import time
from tqdm import tqdm
from mistralai import Mistral
from dotenv import load_dotenv
from llamaapi import LlamaAPI
import google.generativeai as genai
import random
from sklearn.cluster import KMeans
from votek import fast_votek
from evaluation import evaluation
from function import find_matchup_max_cosine_similarity, find_matchup_max_cosine_similarity2, find_matchup_max_cosine_similarity3, find_matchup_max_cosine_similarity4,find_matchup_max_cosine_similarity5,find_matchup_max_cosine_similarity6,find_matchup_max_cosine_similarity22,find_matchup_max_cosine_similarity33,find_matchup_max_cosine_similarity44
warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def load_model(self,api):
        load_dotenv(dotenv_path='api.env')

        # help='LLM, options:[gemini, llama, mistral];
        if self.args.llm == "gemini":
            self.API_KEY = os.getenv('GOOGLE_API_KEY')
            self.gptmodel_name = "gemini-1.5-flash"
            self.generation_config = {
                "temperature": 0.2,
                "top_p": 0.9,
                "top_k": 1,
                "max_output_tokens": 2,
                "response_mime_type": "text/plain",
            }
        elif self.args.llm == "llama":
            if api:
                self.API_KEY = api
            else:
                self.API_KEY = os.getenv('LLAMA_API_KEY')
            # self.gptmodel_name = "llama3.2-90b-vision"
            self.gptmodel_name = "llama3.2-11b-vision"
            self.generation_config = {
                "temperature": 0.01,
                "top_p": 0.9,
                "top_k": 1,
                "max_output_tokens": 2,
                "response_mime_type": "text/plain",
            }

        # elif self.args.llm == "llama":
        #     self.API_KEY = os.getenv('LLAMA_API_KEY')
        #     self.gptmodel_name = "llama3.2-90b-vision"
        #     self.generation_config = {
        #         "temperature": 0.01,
        #         "top_p": 0.9,
        #         "top_k": 1,
        #         "max_output_tokens": 2,
        #         "response_mime_type": "text/plain",
        #     }

        elif self.args.llm == "mistral":
            self.API_KEY = os.getenv('mistral_API_KEY')
            self.gptmodel_name = "mistral-large-latest"
            self.generation_config = {
                "temperature": 0.01,
                "top_p": 0.5,
                "top_k": 1,
                "max_output_tokens": 2,
                "response_mime_type": "text/plain",
            }

    def get_prompt(self):

        pass

    def build_prompt_sst2(self, label_data, label, text, database, N=32):
            # prompt = f"# Label_{label} Instruction:\n"
            prompt = ""
            # prompt = f"# Class {label}: Negative\n"
            if label == 0:
                prompt += f"Class{label}: Negative\n"

                for idx, num in enumerate(label_data[:N]):
                    # print(idx)
                    # print(num)
                    # print(label_data[num, 3])
                    # print(database.iloc[num, 2])
                    # prompt += "\n".join(f"{idx+1}. {database.iloc[num, 2]}")
                    # prompt += f"{idx+1}. {database.iloc[num, 1]}"
                    prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, 1]}\" -> \"Negative\""
                    prompt += "\n"

            elif label == 1:
                prompt += f"Class{label}: Positive\n"

                for idx, num in enumerate(label_data[:N]):
                    # print(idx)
                    # print(num)
                    # print(label_data[num, 3])
                    # print(database.iloc[num, 2])
                    # prompt += "\n".join(f"{idx+1}. {database.iloc[num, 2]}")
                    # prompt += f"{idx+1}. {database.iloc[num, 1]}"
                    prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, 1]}\" -> \"Positive\""
                    prompt += "\n"

            # if len(label_data)!=0:

            # prompt+= f"Label_1 represents positive in sentiment analysis.\n\n"
            # prompt+= f"Here are Label_{label} examples:\n"
            if text != "":
                prompt += f"Query: \"{text}\"\n"
                prompt += f"Prediction: "
            return prompt

    def build_prompt_sst2_knn(self, label_data, label, text, database, N=32):
            # prompt = f"# Label_{label} Instruction:\n"
            prompt = ""
            # prompt = f"# Class {label}: Negative\n"
            # if label == 0:
            #     prompt += f"Class{label}: Negative\n"

            for idx, num in enumerate(label_data[:N]):
                # print(idx)
                # print(num)
                # print(label_data[num, 3])
                # print(database.iloc[num, 2])
                # prompt += "\n".join(f"{idx+1}. {database.iloc[num, 2]}")
                # prompt += f"{idx+1}. {database.iloc[num, 1]}"
                if self.args.basemethod=="zicl":
                    if random.randint(0, 1) ==0:
                        prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, 1]}\" -> \"Bad\""
                        prompt += "\n"
                    else:
                        prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, 1]}\" -> \"Good\""
                        prompt += "\n"

                else:
                    if database.iloc[num, 0] ==0:
                        prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, 1]}\" -> \"Negative\""
                        prompt += "\n"
                    else:
                        prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, 1]}\" -> \"Positive\""
                        prompt += "\n"

            # elif label == 1:
            #     prompt += f"Class{label}: Positive\n"

                # for idx, num in enumerate(label_data[:N]):
                #     # print(idx)
                #     # print(num)
                #     # print(label_data[num, 3])
                #     # print(database.iloc[num, 2])
                #     # prompt += "\n".join(f"{idx+1}. {database.iloc[num, 2]}")
                #     # prompt += f"{idx+1}. {database.iloc[num, 1]}"
                #     prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, 1]}\" -> \"Positive\""
                #     prompt += "\n"

            # if len(label_data)!=0:

            # prompt+= f"Label_1 represents positive in sentiment analysis.\n\n"
            # prompt+= f"Here are Label_{label} examples:\n"
            if text != "":
                prompt += f"Query: \"{text}\"\n"
                prompt += f"Prediction: "
            return prompt

    def build_prompt_cola(self, label_data, label, text, database, N=32):
            # prompt = f"# Label_{label} Instruction:\n"
            prompt = ""
            # prompt = f"# Class {label}: Negative\n"
            if label == 0:
                prompt += f"Class{label}: unacceptable in Linguistic Acceptability(grammaticality)\n"

                for idx, num in enumerate(label_data[:N]):
                    # print(idx)
                    # print(num)
                    # print(label_data[num, 3])
                    # print(database.iloc[num, 2])
                    # prompt += "\n".join(f"{idx+1}. {database.iloc[num, 2]}")
                    # prompt += f"{idx+1}. {database.iloc[num, 1]}"
                    prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, self.content_col]}\" -> \"unacceptable\""
                    prompt += "\n"

            elif label == 1:
                prompt += f"Class{label}: acceptable in Linguistic Acceptability(grammaticality)\n"

                for idx, num in enumerate(label_data[:N]):
                    # print(idx)
                    # print(num)
                    # print(label_data[num, 3])
                    # print(database.iloc[num, 2])
                    # prompt += "\n".join(f"{idx+1}. {database.iloc[num, 2]}")
                    # prompt += f"{idx+1}. {database.iloc[num, 1]}"
                    prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, self.content_col]}\" -> \"acceptable\""
                    prompt += "\n"

            # if len(label_data)!=0:

            # prompt+= f"Label_1 represents positive in sentiment analysis.\n\n"
            # prompt+= f"Here are Label_{label} examples:\n"
            if text != "":
                prompt += f"Query: \"{text}\"\n"
                prompt += f"Prediction: "
            return prompt

    def build_prompt_cola_knn(self, label_data, label, text, database, N=32):
            # prompt = f"# Label_{label} Instruction:\n"
            prompt = ""
            # prompt = f"# Class {label}: Negative\n"
            # if label == 0:
            #     prompt += f"Class{label}: Negative\n"

            for idx, num in enumerate(label_data[:N]):
                # print(idx)
                # print(num)
                # print(label_data[num, 3])
                # print(database.iloc[num, 2])
                # prompt += "\n".join(f"{idx+1}. {database.iloc[num, 2]}")
                # prompt += f"{idx+1}. {database.iloc[num, 1]}"
                if self.args.basemethod=="zicl":
                    if random.randint(0, 1) ==0:
                        prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, 1]}\" -> \"Bad\""
                        prompt += "\n"
                    else:
                        prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, 1]}\" -> \"Good\""
                        prompt += "\n"

                else:
                    if database.iloc[num, self.label_col] ==0:
                        prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, self.content_col]}\" -> \"unacceptable\""
                        prompt += "\n"
                    else:
                        prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, self.content_col]}\" -> \"acceptable\""
                        prompt += "\n"

            # elif label == 1:
            #     prompt += f"Class{label}: Positive\n"

                # for idx, num in enumerate(label_data[:N]):
                #     # print(idx)
                #     # print(num)
                #     # print(label_data[num, 3])
                #     # print(database.iloc[num, 2])
                #     # prompt += "\n".join(f"{idx+1}. {database.iloc[num, 2]}")
                #     # prompt += f"{idx+1}. {database.iloc[num, 1]}"
                #     prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, 1]}\" -> \"Positive\""
                #     prompt += "\n"

            # if len(label_data)!=0:

            # prompt+= f"Label_1 represents positive in sentiment analysis.\n\n"
            # prompt+= f"Here are Label_{label} examples:\n"
            if text != "":
                prompt += f"Query: \"{text}\"\n"
                prompt += f"Prediction: "
            return prompt

    def build_prompt_emotion(self, label_data, label, text, database, N=32):
            # prompt = f"# Label_{label} Instruction:\n"
            prompt = ""
            # prompt = f"# Class {label}: Negative\n"
            if label == 0:
                prompt += f"Class{label}: sadness\n"
                for idx, num in enumerate(label_data[:N]):
                    prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, self.content_col]}\" -> \"sadness\""
                    prompt += "\n"
            elif label ==1:
                prompt += f"Class{label}: joy\n"
                for idx, num in enumerate(label_data[:N]):
                    prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, self.content_col]}\" -> \"joy\""
                    prompt += "\n"
            elif label ==2:
                prompt += f"Class{label}: love\n"
                for idx, num in enumerate(label_data[:N]):
                    prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, self.content_col]}\" -> \"love\""
                    prompt += "\n"
            elif label ==3:
                prompt += f"Class{label}: anger\n"
                for idx, num in enumerate(label_data[:N]):
                    prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, self.content_col]}\" -> \"anger\""
                    prompt += "\n"
            elif label ==4:
                prompt += f"Class{label}: fear\n"
                for idx, num in enumerate(label_data[:N]):
                    prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, self.content_col]}\" -> \"fear\""
                    prompt += "\n"
            elif label ==5:
                prompt += f"Class{label}: surprise\n"
                for idx, num in enumerate(label_data[:N]):
                    prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, self.content_col]}\" -> \"surprise\""
                    prompt += "\n"

            # prompt+= f"Label_1 represents positive in sentiment analysis.\n\n"
            # prompt+= f"Here are Label_{label} examples:\n"
            if text != "":
                prompt += f"Query: \"{text}\"\n"
                prompt += f"Prediction: "
            return prompt

    def build_prompt_emotion_knn(self, label_data, label, text, database, N=32):
            # prompt = f"# Label_{label} Instruction:\n"
            prompt = ""
            # prompt = f"# Class {label}: Negative\n"
            # if label == 0:
            #     prompt += f"Class{label}: Negative\n"
            for idx, num in enumerate(label_data[:N]):
                # print(idx)
                # print(num)
                # print(label_data[num, 3])
                # print(database.iloc[num, 2])
                # prompt += "\n".join(f"{idx+1}. {database.iloc[num, 2]}")
                # prompt += f"{idx+1}. {database.iloc[num, 1]}"
                if self.args.basemethod == "zicl":
                    if database.iloc[num, self.label_col] ==0:
                        prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, self.content_col]}\" -> \"grief\""
                        prompt += "\n"
                    elif database.iloc[num, self.label_col] ==1:
                        prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, self.content_col]}\" -> \"happiness\""
                        prompt += "\n"
                    elif database.iloc[num, self.label_col] ==2:
                        prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, self.content_col]}\" -> \"affection\""
                        prompt += "\n"
                    elif database.iloc[num, self.label_col] ==3:
                        prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, self.content_col]}\" -> \"fury\""
                        prompt += "\n"
                    elif database.iloc[num, self.label_col] ==4:
                        prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, self.content_col]}\" -> \"worry\""
                        prompt += "\n"
                    elif database.iloc[num, self.label_col] ==5:
                        prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, self.content_col]}\" -> \"amazement\""
                        prompt += "\n"
                else:
                    if database.iloc[num, self.label_col] ==0:
                        prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, self.content_col]}\" -> \"sadness\""
                        prompt += "\n"
                    elif database.iloc[num, self.label_col] ==1:
                        prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, self.content_col]}\" -> \"joy\""
                        prompt += "\n"
                    elif database.iloc[num, self.label_col] ==2:
                        prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, self.content_col]}\" -> \"love\""
                        prompt += "\n"
                    elif database.iloc[num, self.label_col] ==3:
                        prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, self.content_col]}\" -> \"anger\""
                        prompt += "\n"
                    elif database.iloc[num, self.label_col] ==4:
                        prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, self.content_col]}\" -> \"fear\""
                        prompt += "\n"
                    elif database.iloc[num, self.label_col] ==5:
                        prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, self.content_col]}\" -> \"surprise\""
                        prompt += "\n"

            # prompt+= f"Label_1 represents positive in sentiment analysis.\n\n"
            # prompt+= f"Here are Label_{label} examples:\n"
            if text != "":
                prompt += f"Query: \"{text}\"\n"
                prompt += f"Prediction: "
            return prompt

    def build_prompt_bbc(self, label_data, label, text, database, N=32):
            # prompt = f"# Label_{label} Instruction:\n"
            prompt = ""
            # 0: tech 1: business 2: sport 3: entertainment 4: politics
            if label == 0:
                prompt += f"Class{label}: tech\n"
                for idx, num in enumerate(label_data[:N]):
                    prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, self.content_col]}\" -> \"tech\""
                    prompt += "\n"
            elif label ==1:
                prompt += f"Class{label}: business\n"
                for idx, num in enumerate(label_data[:N]):
                    prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, self.content_col]}\" -> \"business\""
                    prompt += "\n"
            elif label ==2:
                prompt += f"Class{label}: sport\n"
                for idx, num in enumerate(label_data[:N]):
                    prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, self.content_col]}\" -> \"sport\""
                    prompt += "\n"
            elif label ==3:
                prompt += f"Class{label}: entertainment\n"
                for idx, num in enumerate(label_data[:N]):
                    prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, self.content_col]}\" -> \"entertainment\""
                    prompt += "\n"
            elif label ==4:
                prompt += f"Class{label}: politics\n"
                for idx, num in enumerate(label_data[:N]):
                    prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, self.content_col]}\" -> \"politics\""
                    prompt += "\n"

            if text != "":
                prompt += f"Query: \"{text}\"\n"
                prompt += f"Prediction: "
            return prompt

    def build_prompt_bbc_knn(self, label_data, label, text, database, N=32):
            # prompt = f"# Label_{label} Instruction:\n"
            prompt = ""
            # prompt = f"# Class {label}: Negative\n"
            # if label == 0:
            #     prompt += f"Class{label}: Negative\n"
            for idx, num in enumerate(label_data[:N]):
                # print(idx)
                # print(num)
                # print(label_data[num, 3])
                # print(database.iloc[num, 2])
                # prompt += "\n".join(f"{idx+1}. {database.iloc[num, 2]}")
                # prompt += f"{idx+1}. {database.iloc[num, 1]}"
                if self.args.basemethod == "zicl":
                    if database.iloc[num, self.label_col] ==0:
                        prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, self.content_col]}\" -> \"science\""
                        prompt += "\n"
                    elif database.iloc[num, self.label_col] ==1:
                        prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, self.content_col]}\" -> \"commerce\""
                        prompt += "\n"
                    elif database.iloc[num, self.label_col] ==2:
                        prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, self.content_col]}\" -> \"pastime\""
                        prompt += "\n"
                    elif database.iloc[num, self.label_col] ==3:
                        prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, self.content_col]}\" -> \"amusement\""
                        prompt += "\n"
                    elif database.iloc[num, self.label_col] ==4:
                        prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, self.content_col]}\" -> \"government\""
                        prompt += "\n"

                else:
                    if database.iloc[num, self.label_col] ==0:
                        prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, self.content_col]}\" -> \"tech\""
                        prompt += "\n"
                    elif database.iloc[num, self.label_col] ==1:
                        prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, self.content_col]}\" -> \"business\""
                        prompt += "\n"
                    elif database.iloc[num, self.label_col] ==2:
                        prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, self.content_col]}\" -> \"sport\""
                        prompt += "\n"
                    elif database.iloc[num, self.label_col] ==3:
                        prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, self.content_col]}\" -> \"entertainment\""
                        prompt += "\n"
                    elif database.iloc[num, self.label_col] ==4:
                        prompt += f"  {idx + 1}. Example {idx + 1}: \"{database.iloc[num, self.content_col]}\" -> \"politics\""
                        prompt += "\n"

            # prompt+= f"Label_1 represents positive in sentiment analysis.\n\n"
            # prompt+= f"Here are Label_{label} examples:\n"
            if text != "":
                prompt += f"Query: \"{text}\"\n"
                prompt += f"Prediction: "
            return prompt

    def gen_results_gemini(self, text, df0, df1, group0, group1,n=1, max_retries=4, delay=2):
        retry_count = 0
        genai.configure(api_key=self.API_KEY)
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-001",
            generation_config=self.generation_config,
            system_instruction=self.sysmessage
            # system_instruction="You are a helpful and accurate assistant. You will be provided with a sentence, and your task is to classify as label_0 or label_1(choose either the label_0 or the label_1 tag but NOT both)."
            # safety_settings = Adjust safety settings
            # See https://ai.google.dev/gemini-api/docs/safety-settings
        )
        # prompt = None
        # # prompt = f'You are a helpful and accurate classification assistant. You will be provided with a sentence, and your task is to choose ONLY one Tag(Label_0 and Label_1) for Given Sentence from instructions. Never output others.\n\n'
        # if self.args.dataset=="sst2":
        #     prompt = self.build_prompt_sst2(group0, 0, "", df0, n) + self.build_prompt_sst2(group1, 1, text, df1, n)
        # elif self.args.dataset=="cola":
        #     prompt = self.build_prompt_cola(group0, 0, "", df0, n) + self.build_prompt_cola(group1, 1, text, df1, n)
        # print(prompt)
        # return prompt

        while retry_count < max_retries:
            try:
                output = model.generate_content(self.prompt)
                # print(output)
                return output.text
                # print(output)
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(delay)
                retry_count += 1

    def gen_results_mistral(self, text, df0, df1, group0, group1,n=1, max_retries=4, delay=2):
        retry_count = 0
        # prompt = None
        # # prompt = f'You are a helpful and accurate classification assistant. You will be provided with a sentence, and your task is to choose ONLY one Tag(Label_0 and Label_1) for Given Sentence from instructions. Never output others.\n\n'
        # if self.args.dataset=="sst2":
        #     prompt = self.build_prompt_sst2(group0, 0, "", df0, n) + self.build_prompt_sst2(group1, 1, text, df1, n)
        # elif self.args.dataset=="cola":
        #     prompt = self.build_prompt_cola(group0, 0, "", df0, n) + self.build_prompt_cola(group1, 1, text, df1, n)
        # # print(prompt)
        # return prompt
        # You are given a task where there are multiple classes, and for each class, a few labeled examples are provided. Based on these examples, you need to classify a new unseen instance. Choose ONLY one tag and output the tag. Do Not output others.
        while retry_count < max_retries:
            try:
                api_key = self.API_KEY
                # model = "open-mistral-7b"

                client = Mistral(api_key=api_key)
                chat_response = client.chat.complete(
                    model=self.gptmodel_name,
                    temperature=self.generation_config["temperature"],
                    top_p=self.generation_config["top_p"],
                    messages=[{"role": "system",
                               "content": self.sysmessage},
                              {"role": "user", "content": self.prompt},
                              ],

                    max_tokens=self.generation_config["max_output_tokens"],
                )
                # print(output)
                return chat_response.choices[0].message.content
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(delay)
                retry_count += 1

    def gen_results_llama8b(self,text, df0, df1, group0, group1,n=1, max_retries=4, delay=2):
        retry_count = 0
        # prompt = None
        # # prompt = f'You are a helpful and accurate classification assistant. You will be provided with a sentence, and your task is to choose ONLY one Tag(Label_0 and Label_1) for Given Sentence from instructions. Never output others.\n\n'
        # if self.args.dataset=="sst2":
        #     prompt = self.build_prompt_sst2(group0, 0, "", df0, n) + self.build_prompt_sst2(group1, 1, text, df1, n)
        # elif self.args.dataset=="cola":
        #     prompt = self.build_prompt_cola(group0, 0, "", df0, n) + self.build_prompt_cola(group1, 1, text, df1, n)
        # # print(prompt)

        api_token = self.API_KEY
        # api_token = self.args.api
        llama = LlamaAPI(api_token)

        api_request_json = {
            "model": self.gptmodel_name,
            "messages": [
                {"role": "system",
                 "content": self.sysmessage},
                {"role": "user", "content": self.prompt},
            ],
            "max_tokens": self.generation_config["max_output_tokens"],
            "temperature": self.generation_config["temperature"],
            "top_k": self.generation_config["top_k"],
            "top_p": self.generation_config["top_p"],
        }

        while retry_count < max_retries:
            try:
                response = llama.run(api_request_json)
                # print(output)
                return response.json()['choices'][0]['message']['content']
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(delay)
                retry_count += 1

    def gen_results(self,text):
        if self.args.llm == "gemini":
                return self.gen_results_gemini(text, None, None, None,
                                        None, None)
        elif self.args.llm == "llama":
                return self.gen_results_llama8b(text, None, None, None,
                                         None, None)
        elif self.args.llm == "mistral":
                return self.gen_results_mistral(text, None, None, None,
                                         None, None)

    def exp_sst2(self,):
        # pre-process
        file_path = self.args.root_path + self.args.dataset
        self.df_train, self.df_test = load_dataset(self.args.dataset)
        self.content_col = 1
        self.label_col = 0
        self.addition_col = 2
        self.trainset_emb, self.testset_emb = embedding_encoder(self.df_train.iloc[:, self.content_col],
                                                                self.df_test.iloc[:, self.content_col], file_path, self.args.embeddingmethod)

        self.label0_emb = embedding_encoder_onetime("Negative",self.args.embeddingmethod)
        self.label1_emb = embedding_encoder_onetime("Positive", self.args.embeddingmethod)

        print(self.df_train.shape, self.df_test.shape, self.trainset_emb.shape, self.testset_emb.shape, self.args.embeddingmethod)
        self.df_train['index'] = range(len(self.df_train))
        self.df_test['index'] = range(len(self.df_test))

    def exp_cola(self):
        file_path = self.args.root_path  + self.args.dataset
        self.df_train, self.df_test = load_dataset(self.args.dataset)
        self.content_col = 3
        self.label_col = 1
        self.addition_col=4
        self.trainset_emb, self.testset_emb = embedding_encoder(self.df_train.iloc[:,self.content_col], self.df_test.iloc[:,self.content_col], file_path, self.args.embeddingmethod)

        self.label0_emb = embedding_encoder_onetime("unacceptable",self.args.embeddingmethod)
        self.label1_emb = embedding_encoder_onetime("acceptable",self.args.embeddingmethod)

        print(self.df_train.shape, self.df_test.shape, self.trainset_emb.shape, self.testset_emb.shape)
        print(self.label0_emb.shape, self.label1_emb.shape)
        self.df_train['index'] = range(len(self.df_train))
        self.df_test['index'] = range(len(self.df_test))

    def exp_emotion(self):
        file_path = self.args.root_path  + self.args.dataset
        self.df_train, self.df_test = load_dataset(self.args.dataset)
        self.content_col = 0
        self.label_col = 1
        self.addition_col=2
        self.n = self.args.Nshot
        self.trainset_emb, self.testset_emb = embedding_encoder(self.df_train.iloc[:, self.content_col],
                                                                self.df_test.iloc[:, self.content_col], file_path, self.args.embeddingmethod)
        print(self.trainset_emb.shape, self.testset_emb.shape)
        #
        label = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        self.label_emb = embedding_encoder_onetime(label, self.args.embeddingmethod)
        #
        print(self.label_emb.shape)
        self.df_train['index'] = range(len(self.df_train))
        self.df_test['index'] = range(len(self.df_test))

    def exp_bbc(self,):
        # pre-process
        file_path = self.args.root_path + self.args.dataset
        self.df_train, self.df_test = load_dataset(self.args.dataset)
        self.content_col = 0
        self.label_col = 1
        self.addition_col = 3
        self.trainset_emb, self.testset_emb = embedding_encoder(self.df_train.iloc[:, self.content_col],
                                                                self.df_test.iloc[:, self.content_col], file_path, self.args.embeddingmethod)

        # 0: tech 1: business 2: sport 3: entertainment 4: politics
        label = ["tech", "business", "sport", "entertainment", "politics"]
        self.label_emb = embedding_encoder_onetime(label,self.args.embeddingmethod)
        #
        print(self.label_emb.shape)

        print(self.df_train.shape, self.df_test.shape, self.trainset_emb.shape, self.testset_emb.shape, self.args.embeddingmethod)
        self.df_train['index'] = range(len(self.df_train))
        self.df_test['index'] = range(len(self.df_test))

    def run_results(self, setting):
        # self.load_model(self.args.api)
        self.load_model(api=None)

        ### loda dataset
        if self.args.dataset=="sst2":
            self.exp_sst2()

            if self.args.zeroshot==True:
                print("doing sst2 zeroshot")
                self.sysmessage="You are given a task where there are multiple classes where you need to assign a single label from the following categories: [positive, negative]. Return only the selected category and nothing else."
                res = []
                for i in tqdm(range(0, len(self.df_test))):

                    self.prompt= self.build_prompt_sst2(None, -1, self.df_test.iloc[i, self.content_col], None)
                    res.append(self.gen_results(self.prompt))

                filename=self.setting + ".txt"
                eval_res=evaluation(filename)
                eval_res.eval_sst2(self.df_test.iloc[:, self.label_col], res)

            elif self.args.NwayKshot == True:
                print("doing sst2 NwayKshot")
                self.sysmessage="You are given a task where there are multiple classes, and for each class, a few labeled examples are provided. Based on these examples, you need to classify a new unseen instance. Choose ONLY one tag and output the tag. Do Not output others."
                self.n=self.args.NwayKshot
                XX = self.testset_emb
                # XX=test_embeddings.detach().cpu().numpy()

                X0 = []
                X1 = []
                for i in range(len(self.df_train)):
                    if self.df_train.iloc[i, self.label_col] == 0:
                        X0.append(self.df_train.iloc[i, self.addition_col])
                    else:
                        X1.append(self.df_train.iloc[i, self.addition_col])

                # trainset_emb=train_embeddings.detach().cpu().numpy()

                tp_x0 = self.trainset_emb[X0]
                tp_x1 = self.trainset_emb[X1]

                # labelt_emb = label0_emb
                # labelc_emb = label1_emb

                center0 = KMeans(n_clusters=1, random_state=0, n_init="auto").fit(tp_x0)
                center1 = KMeans(n_clusters=1, random_state=0, n_init="auto").fit(tp_x1)

                df_train0 = self.df_train.iloc[X0, :]
                df_train1 = self.df_train.iloc[X1, :]

                for j in range(4,5):
                    res = []
                    # for i in tqdm(range(0,len(self.df_test))):

                    for i in range(1809,1810):
                        # i=100
                        n = self.args.Nshot
                        tp0 = None
                        tp1 = None
                        if j ==0:
                            tp0, _=find_matchup_max_cosine_similarity(np.expand_dims(XX[i,:], axis=0),tp_x0,self.args.sim)
                            tp1, _=find_matchup_max_cosine_similarity(np.expand_dims(XX[i,:], axis=0),tp_x1,self.args.sim)

                            print(tp_x0[tp0[:3]].shape)
                            _, _ = find_matchup_max_cosine_similarity(center0.cluster_centers_, tp_x0[tp0[:3]],
                                                                      self.args.sim)
                            _, _ = find_matchup_max_cosine_similarity(center1.cluster_centers_, tp_x0[tp0[:3]],
                                                                      self.args.sim)
                            _, _ = find_matchup_max_cosine_similarity(center1.cluster_centers_, tp_x1[tp1[:3]],
                                                                      self.args.sim)
                            _, _ = find_matchup_max_cosine_similarity(center0.cluster_centers_, tp_x1[tp1[:3]],
                                                                      self.args.sim)
                        elif j==1:
                            tp0, _=find_matchup_max_cosine_similarity2(np.expand_dims(XX[i,:], axis=0),tp_x0,np.expand_dims(self.label0_emb, axis=0),np.expand_dims(self.label1_emb, axis=0),self.args.sim, 1, 1,False)
                            tp1, _=find_matchup_max_cosine_similarity2(np.expand_dims(XX[i,:], axis=0),tp_x1,np.expand_dims(self.label1_emb, axis=0),np.expand_dims(self.label0_emb, axis=0),self.args.sim, 1, 1,False)
                        elif j==2:
                            tp0, _=find_matchup_max_cosine_similarity3(np.expand_dims(XX[i,:], axis=0),tp_x0,np.expand_dims(self.label0_emb, axis=0),self.args.sim, 1,False)
                            tp1, _=find_matchup_max_cosine_similarity3(np.expand_dims(XX[i,:], axis=0),tp_x1,np.expand_dims(self.label1_emb, axis=0),self.args.sim, 1,False)
                        elif j==3:
                            tp0, _ = find_matchup_max_cosine_similarity4(np.expand_dims(XX[i, :], axis=0), tp_x0,
                                                                    np.expand_dims(self.label1_emb, axis=0),self.args.sim, 1,False)
                            tp1, _ = find_matchup_max_cosine_similarity4(np.expand_dims(XX[i, :], axis=0), tp_x1,
                                                                    np.expand_dims(self.label0_emb, axis=0),self.args.sim, 1,False)
                        elif j==4:
                            tp0, _=find_matchup_max_cosine_similarity5(np.expand_dims(XX[i,:], axis=0),tp_x0,np.expand_dims(self.label0_emb, axis=0),np.expand_dims(self.label1_emb, axis=0),self.args.sim, self.args.w3,False)
                            tp1, _=find_matchup_max_cosine_similarity5(np.expand_dims(XX[i,:], axis=0),tp_x1,np.expand_dims(self.label1_emb, axis=0),np.expand_dims(self.label0_emb, axis=0),self.args.sim, self.args.w3,False)

                            print(tp_x0[tp0[:3]].shape)
                            _, _ = find_matchup_max_cosine_similarity(center0.cluster_centers_, tp_x0[tp0[:3]], self.args.sim)
                            _, _ = find_matchup_max_cosine_similarity(center1.cluster_centers_, tp_x0[tp0[:3]],
                                                                      self.args.sim)
                            _, _ = find_matchup_max_cosine_similarity(center1.cluster_centers_, tp_x1[tp1[:3]], self.args.sim)
                            _, _ = find_matchup_max_cosine_similarity(center0.cluster_centers_, tp_x1[tp1[:3]],
                                                                      self.args.sim)

                        self.prompt = self.build_prompt_sst2(tp0, 0, "", df_train0, n) + self.build_prompt_sst2(tp1, 1,
                                                                                                         self.df_test.iloc[i, self.content_col], df_train1, self.n)
                        print(df_train0.iloc[tp0[:3]])
                        # print(tp0)
                        print(df_train1.iloc[tp1[:3]])
                        # print(self.df_test.iloc[i, self.content_col])
                        # res.append(self.gen_results(self.prompt))

                    # filename = self.setting + "_" + str(j) + ".txt"
                    # eval_res = evaluation(filename)
                    # eval_res.eval_sst2(self.df_test.iloc[:, self.label_col], res)


            elif self.args.basemethod == "knn":
                print("doing sst2 knn")
                self.sysmessage = "You are given a task where there are multiple classes, and for each class, a few labeled examples are provided. Based on these examples, you need to classify a new unseen instance. Choose ONLY one tag and output the tag. Do Not output others."
                self.n = self.args.Nshot
                XX = self.testset_emb
                tp_x0 = self.trainset_emb

                res = []
                for i in tqdm(range(0, len(self.df_test))):
                # for i in tqdm(range(0, 5)):
                    tp0,_ = find_matchup_max_cosine_similarity(np.expand_dims(XX[i, :], axis=0), tp_x0)
                    self.prompt = self.build_prompt_sst2_knn(tp0, -1, self.df_test.iloc[i, self.content_col], self.df_train, N=self.n)
                    # print(self.prompt)
                    res.append(self.gen_results(self.prompt))

                filename = self.setting + ".txt"
                eval_res = evaluation(filename)
                eval_res.eval_sst2(self.df_test.iloc[:, self.label_col], res)

            elif self.args.basemethod == "zicl":
                print("doing sst2 zicl")
                self.sysmessage = "You are given a task where there are multiple classes where you need to assign a single label from the following categories: [positive, negative]. Return only the selected category and nothing else."
                self.n = self.args.Nshot
                XX = self.testset_emb
                tp_x0 = self.trainset_emb

                res = []
                for i in tqdm(range(0, len(self.df_test))):
                # for i in tqdm(range(0, 5)):
                    tp0,_ = find_matchup_max_cosine_similarity(np.expand_dims(XX[i, :], axis=0), tp_x0)

                    ### Physical Neighbour
                    # print(tp0)
                    for j in range(len(tp0)):
                        if tp0[j]+1<len(self.df_train):
                            tp0[j] = tp0[j]+1
                        else:
                            tp0[j] = tp0[j]-1
                    # print(tp0)
                    self.prompt = self.build_prompt_sst2_knn(tp0, -1, self.df_test.iloc[i, self.content_col], self.df_train, N=self.n)
                    # print(self.prompt)
                    res.append(self.gen_results(self.prompt))

                filename = self.setting + ".txt"
                eval_res = evaluation(filename)
                eval_res.eval_sst2(self.df_test.iloc[:, self.label_col], res)

            elif self.args.basemethod == "majorityvote":
                print("doing sst2 majorityvote")
                self.n = self.args.Nshot
                XX = self.testset_emb
                tp_x0 = self.trainset_emb

                res = []
                for i in tqdm(range(0, len(self.df_test))):
                    # for i in tqdm(range(0, 5)):
                    tp0,tp0_dist = find_matchup_max_cosine_similarity(np.expand_dims(XX[i, :], axis=0), tp_x0)
                    ### majorityvote
                    tp0 = tp0[:self.n]
                    tp0_dist=tp0_dist[:self.n]

                    count_label=np.zeros((2),dtype=int)
                    count_dist=np.zeros((2),dtype=float)

                    for j in range(self.n):
                        count_label[self.df_train.iloc[tp0[j],self.label_col]]+=1
                        count_dist[self.df_train.iloc[tp0[j], self.label_col]] += tp0_dist[j]

                    max_n=max(count_label)
                    max_dist=-100.0
                    ans=-1
                    for j in range(2):
                        if (count_label[j]==max_n) and (max_dist<count_dist[j]/count_label[j]):
                            ans=j
                            max_dist=count_dist[j]/count_label[j]
                    res.append(ans)
                filename = self.setting + ".txt"
                eval_res = evaluation(filename)
                eval_res.eval_sst2(self.df_test.iloc[:, self.label_col], res)

        elif self.args.dataset=="cola":
            # the label (0=unacceptable, 1=acceptable)

            self.exp_cola()

            if self.args.zeroshot == True:
                self.sysmessage = "You are given a task where there are multiple classes where you need to assign a single label from the following categories: [unacceptable, acceptable]. Return only the selected category and nothing else."
                res = []
                for i in tqdm(range(0, len(self.df_test))):
                    self.prompt = self.build_prompt_cola(None, -1, self.df_test.iloc[i, self.content_col], None)
                    res.append(self.gen_results(self.prompt))

                filename = self.setting + ".txt"
                eval_res = evaluation(filename)
                eval_res.eval_cola(self.df_test.iloc[:, self.label_col], res)

            elif self.args.NwayKshot == True:
                print("doing cola NwayKshot")
                self.sysmessage="You are given a task where there are multiple classes, and for each class, a few labeled examples are provided. Based on these examples, you need to classify a new unseen instance. Choose ONLY one tag and output the tag. Do Not output others."

                self.n=self.args.NwayKshot
                XX = self.testset_emb
                # XX=test_embeddings.detach().cpu().numpy()

                X0 = []
                X1 = []
                for i in range(len(self.df_train)):
                    if self.df_train.iloc[i, self.label_col] == 0:
                        X0.append(self.df_train.iloc[i, self.addition_col])
                    else:
                        X1.append(self.df_train.iloc[i, self.addition_col])

                # trainset_emb=train_embeddings.detach().cpu().numpy()

                tp_x0 = self.trainset_emb[X0]
                tp_x1 = self.trainset_emb[X1]
                print(tp_x0.shape)
                # labelt_emb = label0_emb
                # labelc_emb = label1_emb

                df_train0 = self.df_train.iloc[X0, :]
                df_train1 = self.df_train.iloc[X1, :]
                filename = None

                wo=True if self.args.basemethod == "iclwo" else False

                for j in range(1,5):
                    res = []
                    for i in tqdm(range(0,len(self.df_test))):
                    # for i in range(3):
                        # i=100
                        n = self.args.Nshot
                        tp0 = None
                        tp1 = None
                        if j ==0:
                            tp0, _=find_matchup_max_cosine_similarity(np.expand_dims(XX[i,:], axis=0),tp_x0,self.args.sim)
                            tp1, _=find_matchup_max_cosine_similarity(np.expand_dims(XX[i,:], axis=0),tp_x1,self.args.sim)
                            filename = self.setting + '_' + str(j) + ".txt"
                        elif j==1:
                            tp0, _=find_matchup_max_cosine_similarity2(np.expand_dims(XX[i,:], axis=0),tp_x0,np.expand_dims(self.label0_emb, axis=0),np.expand_dims(self.label1_emb, axis=0),self.args.sim, self.args.w1, self.args.w2, wo)
                            tp1, _=find_matchup_max_cosine_similarity2(np.expand_dims(XX[i,:], axis=0),tp_x1,np.expand_dims(self.label1_emb, axis=0),np.expand_dims(self.label0_emb, axis=0),self.args.sim, self.args.w1, self.args.w2, wo)
                            filename = self.setting + '_' + str(j) + '_' + str(self.args.w1 * 10) + '_' + str(
                                self.args.w2 * 10) + ".txt"
                        elif j==2:
                            tp0, _=find_matchup_max_cosine_similarity3(np.expand_dims(XX[i,:], axis=0),tp_x0,np.expand_dims(self.label0_emb, axis=0),self.args.sim, self.args.w1, wo)
                            tp1, _=find_matchup_max_cosine_similarity3(np.expand_dims(XX[i,:], axis=0),tp_x1,np.expand_dims(self.label1_emb, axis=0),self.args.sim, self.args.w1, wo)
                            filename = self.setting + '_' + str(j) + '_' + str(self.args.w1 * 10) + ".txt"
                        elif j==3:
                            tp0, _ = find_matchup_max_cosine_similarity4(np.expand_dims(XX[i, :], axis=0), tp_x0,
                                                                    np.expand_dims(self.label1_emb, axis=0),self.args.sim, self.args.w2, wo)
                            tp1, _ = find_matchup_max_cosine_similarity4(np.expand_dims(XX[i, :], axis=0), tp_x1,
                                                                    np.expand_dims(self.label0_emb, axis=0),self.args.sim, self.args.w2, wo)
                            filename = self.setting + '_' + str(j) + '_' + str(self.args.w2 * 10) + ".txt"
                        elif j==4:
                            tp0, _=find_matchup_max_cosine_similarity5(np.expand_dims(XX[i,:], axis=0),tp_x0,np.expand_dims(self.label0_emb, axis=0),np.expand_dims(self.label1_emb, axis=0),self.args.sim, self.args.w3, wo)
                            tp1, _=find_matchup_max_cosine_similarity5(np.expand_dims(XX[i,:], axis=0),tp_x1,np.expand_dims(self.label1_emb, axis=0),np.expand_dims(self.label0_emb, axis=0),self.args.sim, self.args.w3, wo)
                            filename = self.setting + '_' + str(j) + '_' + str(self.args.w3 * 10) + ".txt"
                        self.prompt = self.build_prompt_cola(tp0, 0, "", df_train0, n) + self.build_prompt_cola(tp1, 1,
                                                                                                         self.df_test.iloc[i, self.content_col], df_train1, n)

                        # print(tp0, tp1)
                        print(self.prompt)
                        # res.append(self.gen_results(self.prompt))

                    eval_res = evaluation(filename)
                    eval_res.eval_cola(self.df_test.iloc[:, self.label_col], res)

            elif self.args.basemethod == "knn":
                print("doing cola knn")
                self.sysmessage = "You are given a task where there are multiple classes, and for each class, a few labeled examples are provided. Based on these examples, you need to classify a new unseen instance. Choose ONLY one tag and output the tag. Do Not output others."
                self.n = self.args.Nshot
                XX = self.testset_emb
                tp_x0 = self.trainset_emb

                res = []
                for i in tqdm(range(0, len(self.df_test))):
                # for i in tqdm(range(0, 5)):
                    tp0,_ = find_matchup_max_cosine_similarity(np.expand_dims(XX[i, :], axis=0), tp_x0)
                    self.prompt = self.build_prompt_cola_knn(tp0, -1, self.df_test.iloc[i, self.content_col], self.df_train, N=self.n)
                    # print(self.prompt)
                    res.append(self.gen_results(self.prompt))

                filename = self.setting + ".txt"
                eval_res = evaluation(filename)
                eval_res.eval_cola(self.df_test.iloc[:, self.label_col], res)

            elif self.args.basemethod == "zicl":
                print("doing cola zicl")
                self.sysmessage = "You are given a task where there are multiple classes where you need to assign a single label from the following categories: [unacceptable, acceptable]. Return only the selected category and nothing else."
                self.n = self.args.Nshot
                XX = self.testset_emb
                tp_x0 = self.trainset_emb

                res = []
                for i in tqdm(range(0, len(self.df_test))):
                # for i in tqdm(range(0, 5)):
                    tp0,_ = find_matchup_max_cosine_similarity(np.expand_dims(XX[i, :], axis=0), tp_x0)

                    ### Physical Neighbour
                    # print(tp0)
                    for j in range(len(tp0)):
                        if tp0[j]+1<len(self.df_train):
                            tp0[j] = tp0[j]+1
                        else:
                            tp0[j] = tp0[j]-1
                    # print(tp0)
                    self.prompt = self.build_prompt_cola_knn(tp0, -1, self.df_test.iloc[i, self.content_col], self.df_train, N=self.n)
                    # print(self.prompt)
                    res.append(self.gen_results(self.prompt))

                filename = self.setting + ".txt"
                eval_res = evaluation(filename)
                eval_res.eval_cola(self.df_test.iloc[:, self.label_col], res)

            elif self.args.basemethod == "majorityvote":
                print("doing cola majorityvote")
                self.n = self.args.Nshot
                XX = self.testset_emb
                tp_x0 = self.trainset_emb

                res = []
                for i in tqdm(range(0, len(self.df_test))):
                    # for i in tqdm(range(0, 5)):
                    tp0,tp0_dist = find_matchup_max_cosine_similarity(np.expand_dims(XX[i, :], axis=0), tp_x0)
                    ### majorityvote
                    tp0 = tp0[:self.n]
                    tp0_dist=tp0_dist[:self.n]

                    count_label=np.zeros((2),dtype=int)
                    count_dist=np.zeros((2),dtype=float)

                    for j in range(self.n):
                        count_label[self.df_train.iloc[tp0[j],self.label_col]]+=1
                        count_dist[self.df_train.iloc[tp0[j], self.label_col]] += tp0_dist[j]

                    max_n=max(count_label)
                    max_dist=-100.0
                    ans=-1
                    for j in range(2):
                        if (count_label[j]==max_n) and (max_dist<count_dist[j]/count_label[j]):
                            ans=j
                            max_dist=count_dist[j]/count_label[j]
                    res.append(ans)
                filename = self.setting + ".txt"
                eval_res = evaluation(filename)
                eval_res.eval_cola(self.df_test.iloc[:, self.label_col], res)

        elif self.args.dataset=="emotion":
            self.exp_emotion()

            if self.args.zeroshot == True:
                # self.sysmessage = "You are given a task where there are multiple classes where you need to assign a single label from the following categories: [unacceptable, acceptable]. Return only the selected category and nothing else."
                res = []
                for i in tqdm(range(0, len(self.df_test))):
                    self.prompt = self.build_prompt_emotion(None, -1, self.df_test.iloc[i, self.content_col], None)
                    res.append(self.gen_results(self.prompt))

                filename = self.setting + ".txt"
                eval_res = evaluation(filename)
                eval_res.eval_emotion(self.df_test.iloc[:, self.label_col], res)

            elif self.args.NwayKshot == True:
                print("doing emotion NwayKshot")
                self.sysmessage="You are given a task where there are multiple classes, and for each class, a few labeled examples are provided. Based on these examples, you need to classify a new unseen instance. Choose ONLY one tag and output the tag. Do Not output based on."

                XX = self.testset_emb
                # XX=test_embeddings.detach().cpu().numpy()
                # sadness: 0 joy: 1 love: 2 anger: 3 fear: 4 surprise: 5
                X0 = []
                X1 = []
                X2 = []
                X3 = []
                X4 = []
                X5 = []
                for i in range(len(self.df_train)):
                    if self.df_train.iloc[i, self.label_col] == 0:
                        X0.append(self.df_train.iloc[i, self.addition_col])
                    elif self.df_train.iloc[i, self.label_col] == 1:
                        X1.append(self.df_train.iloc[i, self.addition_col])
                    elif self.df_train.iloc[i, self.label_col] == 2:
                        X2.append(self.df_train.iloc[i, self.addition_col])
                    elif self.df_train.iloc[i, self.label_col] == 3:
                        X3.append(self.df_train.iloc[i, self.addition_col])
                    elif self.df_train.iloc[i, self.label_col] == 4:
                        X4.append(self.df_train.iloc[i, self.addition_col])
                    elif self.df_train.iloc[i, self.label_col] == 5:
                        X5.append(self.df_train.iloc[i, self.addition_col])

                # trainset_emb=train_embeddings.detach().cpu().numpy()

                tp_x0 = self.trainset_emb[X0]
                tp_x1 = self.trainset_emb[X1]
                tp_x2 = self.trainset_emb[X2]
                tp_x3 = self.trainset_emb[X3]
                tp_x4 = self.trainset_emb[X4]
                tp_x5 = self.trainset_emb[X5]

                # labelt_emb = label0_emb
                # labelc_emb = label1_emb

                df_train0 = self.df_train.iloc[X0, :]
                df_train1 = self.df_train.iloc[X1, :]
                df_train2 = self.df_train.iloc[X2, :]
                df_train3 = self.df_train.iloc[X3, :]
                df_train4 = self.df_train.iloc[X4, :]
                df_train5 = self.df_train.iloc[X5, :]

                wo = True if self.args.basemethod == "iclwo" else False

                for j in range(4, 5):
                    res = []
                    filename = None
                    for i in tqdm(range(0, len(self.df_test))):
                        tp0=None
                        tp1=None
                        tp2=None
                        tp3=None
                        tp4=None
                        tp5=None

                        if j==0:
                            tp0,_ = find_matchup_max_cosine_similarity(np.expand_dims(XX[i, :], axis=0), tp_x0, self.args.sim, wo)

                            tp1,_ = find_matchup_max_cosine_similarity(np.expand_dims(XX[i, :], axis=0), tp_x1, self.args.sim, wo)

                            tp2,_ = find_matchup_max_cosine_similarity(np.expand_dims(XX[i, :], axis=0), tp_x2, self.args.sim, wo)

                            tp3,_ = find_matchup_max_cosine_similarity(np.expand_dims(XX[i, :], axis=0), tp_x3, self.args.sim, wo)

                            tp4,_ = find_matchup_max_cosine_similarity(np.expand_dims(XX[i, :], axis=0), tp_x4, self.args.sim, wo)

                            tp5, _ = find_matchup_max_cosine_similarity(np.expand_dims(XX[i, :], axis=0), tp_x5, self.args.sim, wo)
                            filename = self.setting + '_' + str(j) + ".txt"
                        elif j==1:
                            tp0,_ = find_matchup_max_cosine_similarity22(np.expand_dims(XX[i, :], axis=0), tp_x0, self.label_emb, 0, 6, self.args.sim, self.args.w1, self.args.w2, wo)

                            tp1,_ = find_matchup_max_cosine_similarity22(np.expand_dims(XX[i, :], axis=0), tp_x1, self.label_emb, 1, 6, self.args.sim, self.args.w1, self.args.w2, wo)

                            tp2,_ = find_matchup_max_cosine_similarity22(np.expand_dims(XX[i, :], axis=0), tp_x2, self.label_emb, 2, 6, self.args.sim, self.args.w1, self.args.w2, wo)

                            tp3,_ = find_matchup_max_cosine_similarity22(np.expand_dims(XX[i, :], axis=0), tp_x3, self.label_emb, 3, 6, self.args.sim, self.args.w1, self.args.w2, wo)

                            tp4,_ = find_matchup_max_cosine_similarity22(np.expand_dims(XX[i, :], axis=0), tp_x4, self.label_emb, 4, 6, self.args.sim, self.args.w1, self.args.w2, wo)

                            tp5, _ = find_matchup_max_cosine_similarity22(np.expand_dims(XX[i, :], axis=0), tp_x5,self.label_emb, 5, 6, self.args.sim, self.args.w1, self.args.w2, wo)
                            filename = self.setting + '_' + str(j) + '_' + str(self.args.w1 * 10) + '_' + str(self.args.w2 * 10) + ".txt"
                        elif j==2:
                            tp0,_ = find_matchup_max_cosine_similarity33(np.expand_dims(XX[i, :], axis=0), tp_x0, self.label_emb, 0, 6, self.args.sim, self.args.w1, wo)

                            tp1,_ = find_matchup_max_cosine_similarity33(np.expand_dims(XX[i, :], axis=0), tp_x1, self.label_emb, 1, 6, self.args.sim, self.args.w1, wo )

                            tp2,_ = find_matchup_max_cosine_similarity33(np.expand_dims(XX[i, :], axis=0), tp_x2, self.label_emb, 2, 6, self.args.sim, self.args.w1, wo)

                            tp3,_ = find_matchup_max_cosine_similarity33(np.expand_dims(XX[i, :], axis=0), tp_x3, self.label_emb, 3, 6, self.args.sim, self.args.w1, wo)

                            tp4,_ = find_matchup_max_cosine_similarity33(np.expand_dims(XX[i, :], axis=0), tp_x4, self.label_emb, 4, 6, self.args.sim, self.args.w1, wo)

                            tp5, _ = find_matchup_max_cosine_similarity33(np.expand_dims(XX[i, :], axis=0), tp_x5,
                                                                          self.label_emb, 5, 6, self.args.sim, self.args.w1, wo)
                            filename = self.setting + '_' + str(j) + '_' + str(self.args.w1 * 10) + ".txt"
                        elif j == 3:
                            tp0, _ = find_matchup_max_cosine_similarity44(np.expand_dims(XX[i, :], axis=0), tp_x0,
                                                                          self.label_emb, 0, 6, self.args.sim, self.args.w2, wo)

                            tp1, _ = find_matchup_max_cosine_similarity44(np.expand_dims(XX[i, :], axis=0), tp_x1,
                                                                          self.label_emb, 1, 6, self.args.sim, self.args.w2, wo)

                            tp2, _ = find_matchup_max_cosine_similarity44(np.expand_dims(XX[i, :], axis=0), tp_x2,
                                                                          self.label_emb, 2, 6, self.args.sim, self.args.w2, wo)

                            tp3, _ = find_matchup_max_cosine_similarity44(np.expand_dims(XX[i, :], axis=0), tp_x3,
                                                                          self.label_emb, 3, 6, self.args.sim, self.args.w2, wo)

                            tp4, _ = find_matchup_max_cosine_similarity44(np.expand_dims(XX[i, :], axis=0), tp_x4,
                                                                          self.label_emb, 4, 6, self.args.sim, self.args.w2, wo)

                            tp5, _ = find_matchup_max_cosine_similarity44(np.expand_dims(XX[i, :], axis=0), tp_x5,
                                                                          self.label_emb, 5, 6, self.args.sim, self.args.w2, wo)
                            filename = self.setting + '_' + str(j) + '_' + str(self.args.w2 * 10) + ".txt"
                        elif j == 4:
                            tp0, _ = find_matchup_max_cosine_similarity6(np.expand_dims(XX[i, :], axis=0), tp_x0,
                                                                         self.label_emb, 0, 6, self.args.sim, self.args.w3, wo)

                            tp1, _ = find_matchup_max_cosine_similarity6(np.expand_dims(XX[i, :], axis=0), tp_x1,
                                                                         self.label_emb, 1, 6, self.args.sim, self.args.w3, wo)

                            tp2, _ = find_matchup_max_cosine_similarity6(np.expand_dims(XX[i, :], axis=0), tp_x2,
                                                                         self.label_emb, 2, 6, self.args.sim, self.args.w3, wo)

                            tp3, _ = find_matchup_max_cosine_similarity6(np.expand_dims(XX[i, :], axis=0), tp_x3,
                                                                         self.label_emb, 3, 6, self.args.sim, self.args.w3, wo)

                            tp4, _ = find_matchup_max_cosine_similarity6(np.expand_dims(XX[i, :], axis=0), tp_x4,
                                                                         self.label_emb, 4, 6, self.args.sim, self.args.w3, wo)

                            tp5, _ = find_matchup_max_cosine_similarity6(np.expand_dims(XX[i, :], axis=0), tp_x5,
                                                                         self.label_emb, 5, 6, self.args.sim, self.args.w3, wo)
                            filename = self.setting + '_' + str(j) + '_' + str(self.args.w3 * 10) + ".txt"
                        # print(tp0)

                        # self.prompt= self.build_prompt_emotion(group0, 0, "", df0, self.n) + self.build_prompt_cola(group1, 1, text, df1, n)

                        self.prompt = self.build_prompt_emotion(tp0, 0, "", df_train0, self.n) + \
                                      self.build_prompt_emotion(tp1, 1, "", df_train1, self.n) + \
                                      self.build_prompt_emotion(tp2, 2, "", df_train2, self.n) + \
                                      self.build_prompt_emotion(tp3, 3, "", df_train3, self.n) + \
                                      self.build_prompt_emotion(tp4, 4, "", df_train4, self.n) + \
                                      self.build_prompt_emotion(tp5, 5, self.df_test.iloc[i, self.content_col], df_train5, self.n)

                        # print(self.prompt)

                        res.append(self.gen_results(self.prompt))

                    # filename = self.setting + '_' + str(j) + ".txt"
                    eval_res = evaluation(filename)
                    eval_res.eval_emotion(self.df_test.iloc[:, self.label_col], res)

            elif self.args.basemethod == "knn":
                print("doing emotion knn")
                self.sysmessage = "You are given a task where there are multiple classes, and for each class, a few labeled examples are provided. Based on these examples, you need to classify a new unseen instance. Choose ONLY one tag and output the tag. Do Not output others."
                self.n = self.args.Nshot
                XX = self.testset_emb
                tp_x0 = self.trainset_emb

                res = []
                for i in tqdm(range(0, len(self.df_test))):
                # for i in tqdm(range(0, 5)):
                    tp0,_ = find_matchup_max_cosine_similarity(np.expand_dims(XX[i, :], axis=0), tp_x0, self.args.sim)
                    self.prompt = self.build_prompt_emotion_knn(tp0, -1, self.df_test.iloc[i, self.content_col], self.df_train, N=self.n)
                    # print(self.prompt)
                    res.append(self.gen_results(self.prompt))

                filename = self.setting + ".txt"
                eval_res = evaluation(filename)
                eval_res.eval_emotion(self.df_test.iloc[:, self.label_col], res)

            elif self.args.basemethod == "zicl":
                print("doing emotion zicl")
                self.sysmessage = "You are given a task where there are multiple classes where you need to assign a single label from the following categories: [sadness, joy, love, anger, fear, surprise]. Return only the selected category and nothing else."
                self.n = self.args.Nshot
                XX = self.testset_emb
                tp_x0 = self.trainset_emb

                res = []
                for i in tqdm(range(0, len(self.df_test))):
                # for i in tqdm(range(0, 5)):
                    tp0,_ = find_matchup_max_cosine_similarity(np.expand_dims(XX[i, :], axis=0), tp_x0)

                    ### Physical Neighbour
                    # print(tp0)
                    for i in range(len(tp0)):
                        if tp0[i]+1<len(self.df_train):
                            tp0[i] = tp0[i]+1
                        else:
                            tp0[i] = tp0[i]-1
                    # print(tp0)
                    self.prompt = self.build_prompt_emotion_knn(tp0, -1, self.df_test.iloc[i, self.content_col], self.df_train, N=self.n)
                    # print(self.prompt)
                    res.append(self.gen_results(self.prompt))

                filename = self.setting + ".txt"
                eval_res = evaluation(filename)
                eval_res.eval_emotion(self.df_test.iloc[:, self.label_col], res)

            elif self.args.basemethod == "majorityvote":
                print("doing emotion majorityvote")
                self.n = self.args.Nshot
                XX = self.testset_emb
                tp_x0 = self.trainset_emb

                res = []
                for i in tqdm(range(0, len(self.df_test))):
                    # for i in tqdm(range(0, 5)):
                    tp0,tp0_dist = find_matchup_max_cosine_similarity(np.expand_dims(XX[i, :], axis=0), tp_x0)
                    ### majorityvote
                    tp0 = tp0[:self.n]
                    tp0_dist=tp0_dist[:self.n]

                    count_label=np.zeros((6),dtype=int)
                    count_dist=np.zeros((6),dtype=float)

                    for j in range(self.n):
                        count_label[self.df_train.iloc[tp0[j],self.label_col]]+=1
                        count_dist[self.df_train.iloc[tp0[j], self.label_col]] += tp0_dist[j]

                    max_n=max(count_label)
                    max_dist=-100.0
                    ans=-1
                    for j in range(6):
                        if (count_label[j]==max_n) and (max_dist<count_dist[j]/count_label[j]):
                            ans=j
                            max_dist=count_dist[j]/count_label[j]
                    res.append(ans)
                filename = self.setting + ".txt"
                eval_res = evaluation(filename)
                eval_res.eval_emotion(self.df_test.iloc[:, self.label_col], res)

            elif self.args.basemethod == "votek":
                print("doing emotion votek")
                self.sysmessage="You are given a task where there are multiple classes, and for each class, a few labeled examples are provided. Based on these examples, you need to classify a new unseen instance. Choose ONLY one tag and output the tag. Do Not output based on."

                XX = self.testset_emb
                # XX=test_embeddings.detach().cpu().numpy()
                # sadness: 0 joy: 1 love: 2 anger: 3 fear: 4 surprise: 5
                X0 = []
                X1 = []
                X2 = []
                X3 = []
                X4 = []
                X5 = []
                for i in range(len(self.df_train)):
                    if self.df_train.iloc[i, self.label_col] == 0:
                        X0.append(self.df_train.iloc[i, self.addition_col])
                    elif self.df_train.iloc[i, self.label_col] == 1:
                        X1.append(self.df_train.iloc[i, self.addition_col])
                    elif self.df_train.iloc[i, self.label_col] == 2:
                        X2.append(self.df_train.iloc[i, self.addition_col])
                    elif self.df_train.iloc[i, self.label_col] == 3:
                        X3.append(self.df_train.iloc[i, self.addition_col])
                    elif self.df_train.iloc[i, self.label_col] == 4:
                        X4.append(self.df_train.iloc[i, self.addition_col])
                    elif self.df_train.iloc[i, self.label_col] == 5:
                        X5.append(self.df_train.iloc[i, self.addition_col])

                # trainset_emb=train_embeddings.detach().cpu().numpy()

                tp_x0 = self.trainset_emb[X0]
                tp_x1 = self.trainset_emb[X1]
                tp_x2 = self.trainset_emb[X2]
                tp_x3 = self.trainset_emb[X3]
                tp_x4 = self.trainset_emb[X4]
                tp_x5 = self.trainset_emb[X5]

                # labelt_emb = label0_emb
                # labelc_emb = label1_emb

                df_train0 = self.df_train.iloc[X0, :]
                df_train1 = self.df_train.iloc[X1, :]
                df_train2 = self.df_train.iloc[X2, :]
                df_train3 = self.df_train.iloc[X3, :]
                df_train4 = self.df_train.iloc[X4, :]
                df_train5 = self.df_train.iloc[X5, :]

                for j in range(0, 1):
                    res = []
                    filename = None
                    tp0 = fast_votek(tp_x0, 3, 3)

                    tp1 = fast_votek(tp_x1, 3, 3)

                    tp2 = fast_votek(tp_x2, 3, 3)

                    tp3 = fast_votek(tp_x3, 3, 3)

                    tp4 = fast_votek(tp_x4, 3, 3)

                    tp5 = fast_votek(tp_x5, 3, 3)
                    for i in tqdm(range(0, len(self.df_test))):

                        filename = self.setting + '_' + str(j) + ".txt"


                        # self.prompt= self.build_prompt_emotion(group0, 0, "", df0, self.n) + self.build_prompt_cola(group1, 1, text, df1, n)

                        self.prompt = self.build_prompt_emotion(tp0, 0, "", df_train0, self.n) + \
                                      self.build_prompt_emotion(tp1, 1, "", df_train1, self.n) + \
                                      self.build_prompt_emotion(tp2, 2, "", df_train2, self.n) + \
                                      self.build_prompt_emotion(tp3, 3, "", df_train3, self.n) + \
                                      self.build_prompt_emotion(tp4, 4, "", df_train4, self.n) + \
                                      self.build_prompt_emotion(tp5, 5, self.df_test.iloc[i, self.content_col], df_train5, self.n)

                        # print(self.prompt)

                        res.append(self.gen_results(self.prompt))

                    # filename = self.setting + '_' + str(j) + ".txt"
                    eval_res = evaluation(filename)
                    eval_res.eval_emotion(self.df_test.iloc[:, self.label_col], res)

            elif self.args.basemethod == "selfprompt":

                print("doing emotion selfprompt")
                self.sysmessage = "You are given a task where there are multiple classes, and for each class, a few labeled examples are provided. Based on these examples, you need to classify a new unseen instance. Choose ONLY one tag and output the tag. Do Not output based on."

                XX = self.testset_emb
                # XX=test_embeddings.detach().cpu().numpy()
                # sadness: 0 joy: 1 love: 2 anger: 3 fear: 4 surprise: 5
                X0 = []
                X1 = []
                X2 = []
                X3 = []
                X4 = []
                X5 = []
                for i in range(len(self.df_train)):
                    if self.df_train.iloc[i, self.label_col] == 0:
                        X0.append(self.df_train.iloc[i, self.addition_col])
                    elif self.df_train.iloc[i, self.label_col] == 1:
                        X1.append(self.df_train.iloc[i, self.addition_col])
                    elif self.df_train.iloc[i, self.label_col] == 2:
                        X2.append(self.df_train.iloc[i, self.addition_col])
                    elif self.df_train.iloc[i, self.label_col] == 3:
                        X3.append(self.df_train.iloc[i, self.addition_col])
                    elif self.df_train.iloc[i, self.label_col] == 4:
                        X4.append(self.df_train.iloc[i, self.addition_col])
                    elif self.df_train.iloc[i, self.label_col] == 5:
                        X5.append(self.df_train.iloc[i, self.addition_col])

                # trainset_emb=train_embeddings.detach().cpu().numpy()

                tp_x0 = self.trainset_emb[X0]
                tp_x1 = self.trainset_emb[X1]
                tp_x2 = self.trainset_emb[X2]
                tp_x3 = self.trainset_emb[X3]
                tp_x4 = self.trainset_emb[X4]
                tp_x5 = self.trainset_emb[X5]

                # labelt_emb = label0_emb
                # labelc_emb = label1_emb

                df_train0 = self.df_train.iloc[X0, :]
                df_train1 = self.df_train.iloc[X1, :]
                df_train2 = self.df_train.iloc[X2, :]
                df_train3 = self.df_train.iloc[X3, :]
                df_train4 = self.df_train.iloc[X4, :]
                df_train5 = self.df_train.iloc[X5, :]

                center0 = KMeans(n_clusters=1, random_state=0, n_init="auto").fit(tp_x0)
                center1 = KMeans(n_clusters=1, random_state=0, n_init="auto").fit(tp_x1)
                center2 = KMeans(n_clusters=1, random_state=0, n_init="auto").fit(tp_x2)
                center3 = KMeans(n_clusters=1, random_state=0, n_init="auto").fit(tp_x3)
                center4 = KMeans(n_clusters=1, random_state=0, n_init="auto").fit(tp_x4)
                center5 = KMeans(n_clusters=1, random_state=0, n_init="auto").fit(tp_x5)

                tp0,_ = find_matchup_max_cosine_similarity(center0.cluster_centers_, tp_x0, self.args.sim)
                tp1,_ = find_matchup_max_cosine_similarity(center1.cluster_centers_, tp_x1, self.args.sim)
                tp2,_ = find_matchup_max_cosine_similarity(center2.cluster_centers_, tp_x2, self.args.sim)
                tp3,_ = find_matchup_max_cosine_similarity(center3.cluster_centers_, tp_x3, self.args.sim)
                tp4,_ = find_matchup_max_cosine_similarity(center4.cluster_centers_, tp_x4, self.args.sim)
                tp5,_ = find_matchup_max_cosine_similarity(center5.cluster_centers_, tp_x5, self.args.sim)

                n = 3
                tp0 = tp0[:n]
                tp1 = tp1[:n]
                tp2 = tp2[:n]
                tp3 = tp3[:n]
                tp4 = tp4[:n]
                tp5 = tp5[:n]



                for j in range(0, 1):
                    res = []
                    filename = None

                    for i in tqdm(range(0, len(self.df_test))):
                        filename = self.setting + '_' + str(j) + ".txt"

                        # self.prompt= self.build_prompt_emotion(group0, 0, "", df0, self.n) + self.build_prompt_cola(group1, 1, text, df1, n)

                        self.prompt = self.build_prompt_emotion(tp0, 0, "", df_train0, self.n) + \
                                      self.build_prompt_emotion(tp1, 1, "", df_train1, self.n) + \
                                      self.build_prompt_emotion(tp2, 2, "", df_train2, self.n) + \
                                      self.build_prompt_emotion(tp3, 3, "", df_train3, self.n) + \
                                      self.build_prompt_emotion(tp4, 4, "", df_train4, self.n) + \
                                      self.build_prompt_emotion(tp5, 5, self.df_test.iloc[i, self.content_col],
                                                                df_train5, self.n)

                        # print(self.prompt)

                        res.append(self.gen_results(self.prompt))

                    # filename = self.setting + '_' + str(j) + ".txt"
                    eval_res = evaluation(filename)
                    eval_res.eval_emotion(self.df_test.iloc[:, self.label_col], res)

        elif self.args.dataset == "bbc":
            self.exp_bbc()
            if self.args.zeroshot == True:
                self.sysmessage = "You are given a task where there are multiple classes where you need to assign a single label from the following categories: [tech, business, sport, entertainment, politics]. Return only the selected category and nothing else."
                res = []
                for i in tqdm(range(0, len(self.df_test))):
                    self.prompt = self.build_prompt_bbc(None, -1, self.df_test.iloc[i, self.content_col], None)
                    # print(self.prompt)
                    res.append(self.gen_results(self.prompt))

                filename = self.setting + ".txt"
                eval_res = evaluation(filename)
                eval_res.eval_bbc(self.df_test.iloc[:, self.label_col], res)


            elif self.args.NwayKshot == True:

                print("doing bbc NwayKshot")

                self.sysmessage = "You are given a task where there are multiple classes, and for each class, a few labeled examples are provided. Based on these examples, you need to classify a new unseen instance. Choose ONLY one tag and output the tag. Do Not output based on."

                XX = self.testset_emb

                # XX=test_embeddings.detach().cpu().numpy()

                # # 0: tech 1: business 2: sport 3: entertainment 4: politics

                X0 = []

                X1 = []

                X2 = []

                X3 = []

                X4 = []

                for i in range(len(self.df_train)):

                    if self.df_train.iloc[i, self.label_col] == 0:

                        X0.append(self.df_train.iloc[i, self.addition_col])

                    elif self.df_train.iloc[i, self.label_col] == 1:

                        X1.append(self.df_train.iloc[i, self.addition_col])

                    elif self.df_train.iloc[i, self.label_col] == 2:

                        X2.append(self.df_train.iloc[i, self.addition_col])

                    elif self.df_train.iloc[i, self.label_col] == 3:

                        X3.append(self.df_train.iloc[i, self.addition_col])

                    elif self.df_train.iloc[i, self.label_col] == 4:

                        X4.append(self.df_train.iloc[i, self.addition_col])


                # trainset_emb=train_embeddings.detach().cpu().numpy()

                tp_x0 = self.trainset_emb[X0]

                tp_x1 = self.trainset_emb[X1]

                tp_x2 = self.trainset_emb[X2]

                tp_x3 = self.trainset_emb[X3]

                tp_x4 = self.trainset_emb[X4]

                df_train0 = self.df_train.iloc[X0, :]

                df_train1 = self.df_train.iloc[X1, :]

                df_train2 = self.df_train.iloc[X2, :]

                df_train3 = self.df_train.iloc[X3, :]

                df_train4 = self.df_train.iloc[X4, :]

                self.n = self.args.Nshot

                wo = True if self.args.basemethod == "iclwo" else False


                for j in range(4, 5):
                    res = []

                    for i in tqdm(range(0, len(self.df_test))):
                        tp0=None
                        tp1=None
                        tp2=None
                        tp3=None
                        tp4=None

                        if j==0:
                            tp0,_ = find_matchup_max_cosine_similarity(np.expand_dims(XX[i, :], axis=0), tp_x0)

                            tp1,_ = find_matchup_max_cosine_similarity(np.expand_dims(XX[i, :], axis=0), tp_x1)

                            tp2,_ = find_matchup_max_cosine_similarity(np.expand_dims(XX[i, :], axis=0), tp_x2)

                            tp3,_ = find_matchup_max_cosine_similarity(np.expand_dims(XX[i, :], axis=0), tp_x3)

                            tp4,_ = find_matchup_max_cosine_similarity(np.expand_dims(XX[i, :], axis=0), tp_x4)

                        elif j==1:
                            tp0,_ = find_matchup_max_cosine_similarity22(np.expand_dims(XX[i, :], axis=0), tp_x0, self.label_emb, 0, 5, self.args.sim, self.args.w1, self.args.w2, wo)

                            tp1,_ = find_matchup_max_cosine_similarity22(np.expand_dims(XX[i, :], axis=0), tp_x1, self.label_emb, 1, 5, self.args.sim, self.args.w1, self.args.w2, wo)

                            tp2,_ = find_matchup_max_cosine_similarity22(np.expand_dims(XX[i, :], axis=0), tp_x2, self.label_emb, 2, 5, self.args.sim, self.args.w1, self.args.w2, wo)

                            tp3,_ = find_matchup_max_cosine_similarity22(np.expand_dims(XX[i, :], axis=0), tp_x3, self.label_emb, 3, 5, self.args.sim, self.args.w1, self.args.w2, wo)

                            tp4,_ = find_matchup_max_cosine_similarity22(np.expand_dims(XX[i, :], axis=0), tp_x4, self.label_emb, 4, 5, self.args.sim, self.args.w1, self.args.w2, wo)

                        elif j==2:
                            tp0,_ = find_matchup_max_cosine_similarity33(np.expand_dims(XX[i, :], axis=0), tp_x0, self.label_emb, 0, 5, self.args.sim, self.args.w1, wo)

                            tp1,_ = find_matchup_max_cosine_similarity33(np.expand_dims(XX[i, :], axis=0), tp_x1, self.label_emb, 1, 5, self.args.sim, self.args.w1, wo)

                            tp2,_ = find_matchup_max_cosine_similarity33(np.expand_dims(XX[i, :], axis=0), tp_x2, self.label_emb, 2, 5, self.args.sim, self.args.w1, wo)

                            tp3,_ = find_matchup_max_cosine_similarity33(np.expand_dims(XX[i, :], axis=0), tp_x3, self.label_emb, 3, 5, self.args.sim, self.args.w1, wo)

                            tp4,_ = find_matchup_max_cosine_similarity33(np.expand_dims(XX[i, :], axis=0), tp_x4, self.label_emb, 4, 5, self.args.sim, self.args.w1, wo)

                        elif j == 3:
                            tp0, _ = find_matchup_max_cosine_similarity44(np.expand_dims(XX[i, :], axis=0), tp_x0,
                                                                          self.label_emb, 0, 5, self.args.sim, self.args.w2, wo)

                            tp1, _ = find_matchup_max_cosine_similarity44(np.expand_dims(XX[i, :], axis=0), tp_x1,
                                                                          self.label_emb, 1, 5, self.args.sim, self.args.w2, wo)

                            tp2, _ = find_matchup_max_cosine_similarity44(np.expand_dims(XX[i, :], axis=0), tp_x2,
                                                                          self.label_emb, 2, 5, self.args.sim, self.args.w2, wo)

                            tp3, _ = find_matchup_max_cosine_similarity44(np.expand_dims(XX[i, :], axis=0), tp_x3,
                                                                          self.label_emb, 3, 5, self.args.sim, self.args.w2, wo)

                            tp4, _ = find_matchup_max_cosine_similarity44(np.expand_dims(XX[i, :], axis=0), tp_x4,
                                                                          self.label_emb, 4, 5, self.args.sim, self.args.w2, wo)

                        elif j == 4:
                            tp0, _ = find_matchup_max_cosine_similarity6(np.expand_dims(XX[i, :], axis=0), tp_x0,
                                                                         self.label_emb, 0, 5, self.args.sim, self.args.w3, wo)

                            tp1, _ = find_matchup_max_cosine_similarity6(np.expand_dims(XX[i, :], axis=0), tp_x1,
                                                                         self.label_emb, 1, 5, self.args.sim, self.args.w3, wo)

                            tp2, _ = find_matchup_max_cosine_similarity6(np.expand_dims(XX[i, :], axis=0), tp_x2,
                                                                         self.label_emb, 2, 5, self.args.sim, self.args.w3, wo)

                            tp3, _ = find_matchup_max_cosine_similarity6(np.expand_dims(XX[i, :], axis=0), tp_x3,
                                                                         self.label_emb, 3, 5, self.args.sim, self.args.w3, wo)

                            tp4, _ = find_matchup_max_cosine_similarity6(np.expand_dims(XX[i, :], axis=0), tp_x4,
                                                                         self.label_emb, 4, 5, self.args.sim, self.args.w3, wo)

                        # print(tp0)

                        # self.prompt= self.build_prompt_emotion(group0, 0, "", df0, self.n) + self.build_prompt_cola(group1, 1, text, df1, n)

                        self.prompt = self.build_prompt_bbc(tp0, 0, "", df_train0, self.n) + \
                                      self.build_prompt_bbc(tp1, 1, "", df_train1, self.n) + \
                                      self.build_prompt_bbc(tp2, 2, "", df_train2, self.n) + \
                                      self.build_prompt_bbc(tp3, 3, "", df_train3, self.n) + \
                                      self.build_prompt_bbc(tp4, 4, self.df_test.iloc[i, self.content_col], df_train4, self.n)

                        # print(self.prompt)

                        res.append(self.gen_results(self.prompt))

                    filename = self.setting + "_" + str(j) + ".txt"

                    eval_res = evaluation(filename)

                    eval_res.eval_bbc(self.df_test.iloc[:, self.label_col], res)

            elif self.args.basemethod == "knn":
                print("doing bbc knn")
                self.sysmessage = "You are given a task where there are multiple classes, and for each class, a few labeled examples are provided. Based on these examples, you need to classify a new unseen instance. Choose ONLY one tag and output the tag. Do Not output others."
                self.n = self.args.Nshot
                XX = self.testset_emb
                tp_x0 = self.trainset_emb

                res = []
                for i in tqdm(range(0, len(self.df_test))):
                    # for i in tqdm(range(0, 5)):
                    tp0, _ = find_matchup_max_cosine_similarity(np.expand_dims(XX[i, :], axis=0), tp_x0)
                    self.prompt = self.build_prompt_bbc_knn(tp0, -1, self.df_test.iloc[i, self.content_col],
                                                                self.df_train, N=self.n)
                    # print(self.prompt)
                    res.append(self.gen_results(self.prompt))

                filename = self.setting + ".txt"
                eval_res = evaluation(filename)
                eval_res.eval_bbc(self.df_test.iloc[:, self.label_col], res)

            elif self.args.basemethod == "zicl":
                print("doing bbc zicl")
                self.sysmessage = "You are given a task where there are multiple classes where you need to assign a single label from the following categories: [tech, business, sport, entertainment, politics]. Return only the selected category and nothing else."
                self.n = self.args.Nshot
                XX = self.testset_emb
                tp_x0 = self.trainset_emb

                res = []
                for i in tqdm(range(0, len(self.df_test))):
                # for i in tqdm(range(0, 5)):
                    tp0,_ = find_matchup_max_cosine_similarity(np.expand_dims(XX[i, :], axis=0), tp_x0)

                    ### Physical Neighbour
                    # print(tp0)
                    for j in range(len(tp0)):
                        if tp0[j]+1<len(self.df_train):
                            tp0[j] = tp0[j]+1
                        else:
                            tp0[j] = tp0[j]-1
                    # print(tp0)
                    self.prompt = self.build_prompt_bbc_knn(tp0, -1, self.df_test.iloc[i, self.content_col], self.df_train, N=self.n)
                    # print(self.prompt)
                    res.append(self.gen_results(self.prompt))

                filename = self.setting + ".txt"
                eval_res = evaluation(filename)
                eval_res.eval_bbc(self.df_test.iloc[:, self.label_col], res)

            elif self.args.basemethod == "majorityvote":
                print("doing bbc majorityvote")
                self.n = self.args.Nshot
                XX = self.testset_emb
                tp_x0 = self.trainset_emb

                res = []
                for i in tqdm(range(0, len(self.df_test))):
                    # for i in tqdm(range(0, 5)):
                    tp0, tp0_dist = find_matchup_max_cosine_similarity(np.expand_dims(XX[i, :], axis=0), tp_x0)
                    ### majorityvote
                    tp0 = tp0[:self.n]
                    tp0_dist = tp0_dist[:self.n]

                    count_label = np.zeros((5), dtype=int)
                    count_dist = np.zeros((5), dtype=float)

                    for j in range(self.n):
                        count_label[self.df_train.iloc[tp0[j], self.label_col]] += 1
                        count_dist[self.df_train.iloc[tp0[j], self.label_col]] += tp0_dist[j]

                    max_n = max(count_label)
                    max_dist = -100.0
                    ans = -1
                    for j in range(5):
                        if (count_label[j] == max_n) and (max_dist < count_dist[j] / count_label[j]):
                            ans = j
                            max_dist = count_dist[j] / count_label[j]
                    res.append(ans)
                filename = self.setting + ".txt"
                eval_res = evaluation(filename)
                eval_res.eval_bbc(self.df_test.iloc[:, self.label_col], res)

            elif self.args.basemethod == "votek":
                print("doing bbc votek")

                self.sysmessage = "You are given a task where there are multiple classes, and for each class, a few labeled examples are provided. Based on these examples, you need to classify a new unseen instance. Choose ONLY one tag and output the tag. Do Not output based on."

                XX = self.testset_emb

                # XX=test_embeddings.detach().cpu().numpy()

                # # 0: tech 1: business 2: sport 3: entertainment 4: politics

                X0 = []

                X1 = []

                X2 = []

                X3 = []

                X4 = []

                for i in range(len(self.df_train)):

                    if self.df_train.iloc[i, self.label_col] == 0:

                        X0.append(self.df_train.iloc[i, self.addition_col])

                    elif self.df_train.iloc[i, self.label_col] == 1:

                        X1.append(self.df_train.iloc[i, self.addition_col])

                    elif self.df_train.iloc[i, self.label_col] == 2:

                        X2.append(self.df_train.iloc[i, self.addition_col])

                    elif self.df_train.iloc[i, self.label_col] == 3:

                        X3.append(self.df_train.iloc[i, self.addition_col])

                    elif self.df_train.iloc[i, self.label_col] == 4:

                        X4.append(self.df_train.iloc[i, self.addition_col])

                # trainset_emb=train_embeddings.detach().cpu().numpy()

                tp_x0 = self.trainset_emb[X0]

                tp_x1 = self.trainset_emb[X1]

                tp_x2 = self.trainset_emb[X2]

                tp_x3 = self.trainset_emb[X3]

                tp_x4 = self.trainset_emb[X4]

                df_train0 = self.df_train.iloc[X0, :]

                df_train1 = self.df_train.iloc[X1, :]

                df_train2 = self.df_train.iloc[X2, :]

                df_train3 = self.df_train.iloc[X3, :]

                df_train4 = self.df_train.iloc[X4, :]

                self.n = self.args.Nshot

                tp0 = fast_votek(tp_x0, 3, 3)

                tp1 = fast_votek(tp_x1, 3, 3)

                tp2 = fast_votek(tp_x2, 3, 3)

                tp3 = fast_votek(tp_x3, 3, 3)

                tp4 = fast_votek(tp_x4, 3, 3)
                self.df_test=self.df_test.iloc[742:908,:]
                for j in range(0, 1):
                    res = []

                    for i in tqdm(range(0, len(self.df_test))):
                    # for i in tqdm(range(742, 907)):

                        # print(tp0)

                        # self.prompt= self.build_prompt_emotion(group0, 0, "", df0, self.n) + self.build_prompt_cola(group1, 1, text, df1, n)

                        self.prompt = self.build_prompt_bbc(tp0, 0, "", df_train0, self.n) + \
                                      self.build_prompt_bbc(tp1, 1, "", df_train1, self.n) + \
                                      self.build_prompt_bbc(tp2, 2, "", df_train2, self.n) + \
                                      self.build_prompt_bbc(tp3, 3, "", df_train3, self.n) + \
                                      self.build_prompt_bbc(tp4, 4, self.df_test.iloc[i, self.content_col], df_train4, self.n)

                        # print(self.prompt)

                        res.append(self.gen_results(self.prompt))
                        print(self.df_test.iloc[i, self.label_col], "-", res[i])
                    filename = self.setting + "_" + str(j) + ".txt"

                    eval_res = evaluation(filename)

                    # eval_res.eval_bbc(self.df_test.iloc[:, self.label_col], res)

            elif self.args.basemethod == "selfprompt":
                print("doing bbc selfprompt")

                self.sysmessage = "You are given a task where there are multiple classes, and for each class, a few labeled examples are provided. Based on these examples, you need to classify a new unseen instance. Choose ONLY one tag and output the tag. Do Not output based on."

                XX = self.testset_emb

                # XX=test_embeddings.detach().cpu().numpy()

                # # 0: tech 1: business 2: sport 3: entertainment 4: politics

                X0 = []

                X1 = []

                X2 = []

                X3 = []

                X4 = []

                for i in range(len(self.df_train)):

                    if self.df_train.iloc[i, self.label_col] == 0:

                        X0.append(self.df_train.iloc[i, self.addition_col])

                    elif self.df_train.iloc[i, self.label_col] == 1:

                        X1.append(self.df_train.iloc[i, self.addition_col])

                    elif self.df_train.iloc[i, self.label_col] == 2:

                        X2.append(self.df_train.iloc[i, self.addition_col])

                    elif self.df_train.iloc[i, self.label_col] == 3:

                        X3.append(self.df_train.iloc[i, self.addition_col])

                    elif self.df_train.iloc[i, self.label_col] == 4:

                        X4.append(self.df_train.iloc[i, self.addition_col])

                # trainset_emb=train_embeddings.detach().cpu().numpy()

                tp_x0 = self.trainset_emb[X0]

                tp_x1 = self.trainset_emb[X1]

                tp_x2 = self.trainset_emb[X2]

                tp_x3 = self.trainset_emb[X3]

                tp_x4 = self.trainset_emb[X4]

                df_train0 = self.df_train.iloc[X0, :]

                df_train1 = self.df_train.iloc[X1, :]

                df_train2 = self.df_train.iloc[X2, :]

                df_train3 = self.df_train.iloc[X3, :]

                df_train4 = self.df_train.iloc[X4, :]

                self.n = self.args.Nshot

                center0 = KMeans(n_clusters=1, random_state=0, n_init="auto").fit(tp_x0)
                center1 = KMeans(n_clusters=1, random_state=0, n_init="auto").fit(tp_x1)
                center2 = KMeans(n_clusters=1, random_state=0, n_init="auto").fit(tp_x2)
                center3 = KMeans(n_clusters=1, random_state=0, n_init="auto").fit(tp_x3)
                center4 = KMeans(n_clusters=1, random_state=0, n_init="auto").fit(tp_x4)

                tp0,_ = find_matchup_max_cosine_similarity(center0.cluster_centers_, tp_x0, self.args.sim)
                tp1,_ = find_matchup_max_cosine_similarity(center1.cluster_centers_, tp_x1, self.args.sim)
                tp2,_ = find_matchup_max_cosine_similarity(center2.cluster_centers_, tp_x2, self.args.sim)
                tp3,_ = find_matchup_max_cosine_similarity(center3.cluster_centers_, tp_x3, self.args.sim)
                tp4,_ = find_matchup_max_cosine_similarity(center4.cluster_centers_, tp_x4, self.args.sim)

                n = 3
                tp0 = tp0[:n]
                tp1 = tp1[:n]
                tp2 = tp2[:n]
                tp3 = tp3[:n]
                tp4 = tp4[:n]

                for j in range(0, 1):
                    res = []

                    for i in tqdm(range(0, len(self.df_test))):
                        # print(tp0)

                        # self.prompt= self.build_prompt_emotion(group0, 0, "", df0, self.n) + self.build_prompt_cola(group1, 1, text, df1, n)

                        self.prompt = self.build_prompt_bbc(tp0, 0, "", df_train0, self.n) + \
                                      self.build_prompt_bbc(tp1, 1, "", df_train1, self.n) + \
                                      self.build_prompt_bbc(tp2, 2, "", df_train2, self.n) + \
                                      self.build_prompt_bbc(tp3, 3, "", df_train3, self.n) + \
                                      self.build_prompt_bbc(tp4, 4, self.df_test.iloc[i, self.content_col], df_train4,
                                                            self.n)

                        # print(self.prompt)

                        res.append(self.gen_results(self.prompt))

                    filename = self.setting + "_" + str(j) + ".txt"

                    eval_res = evaluation(filename)

                    eval_res.eval_bbc(self.df_test.iloc[:, self.label_col], res)


        # if self.args.NwayKshot == True:
        #     self.exp_cola()
        #     XX = self.testset_emb
        #     # XX=test_embeddings.detach().cpu().numpy()
        #
        #     X0 = []
        #     X1 = []
        #     for i in range(len(self.df_train)):
        #         if self.df_train.iloc[i, self.label_col] == 0:
        #             X0.append(self.df_train.iloc[i, self.addition_col])
        #         else:
        #             X1.append(self.df_train.iloc[i, self.addition_col])
        #
        #     # trainset_emb=train_embeddings.detach().cpu().numpy()
        #
        #     tp_x0 = self.trainset_emb[X0]
        #     tp_x1 = self.trainset_emb[X1]
        #
        #     # labelt_emb = label0_emb
        #     # labelc_emb = label1_emb
        #
        #     df_train0 = self.df_train.iloc[X0, :]
        #     df_train1 = self.df_train.iloc[X1, :]
        #
        #     for j in range(0,1):
        #         res = []
        #         for i in tqdm(range(0,len(self.df_test))):
        #         # for i in range(3):
        #             # i=100
        #             n = self.args.Nshot
        #             tp0 = None
        #             tp1 = None
        #             if j ==0:
        #                 tp0=find_matchup_max_cosine_similarity(np.expand_dims(XX[i,:], axis=0),tp_x0)
        #                 tp1=find_matchup_max_cosine_similarity(np.expand_dims(XX[i,:], axis=0),tp_x1)
        #             elif j==1:
        #                 tp0=find_matchup_max_cosine_similarity2(np.expand_dims(XX[i,:], axis=0),tp_x0,np.expand_dims(self.label0_emb, axis=0),np.expand_dims(self.label1_emb, axis=0))
        #                 tp1=find_matchup_max_cosine_similarity2(np.expand_dims(XX[i,:], axis=0),tp_x1,np.expand_dims(self.label1_emb, axis=0),np.expand_dims(self.label0_emb, axis=0))
        #             elif j==2:
        #                 tp0=find_matchup_max_cosine_similarity3(np.expand_dims(XX[i,:], axis=0),tp_x0,np.expand_dims(self.label0_emb, axis=0))
        #                 tp1=find_matchup_max_cosine_similarity3(np.expand_dims(XX[i,:], axis=0),tp_x1,np.expand_dims(self.label1_emb, axis=0))
        #             elif j==3:
        #                 tp0 = find_matchup_max_cosine_similarity4(np.expand_dims(XX[i, :], axis=0), tp_x0,
        #                                                         np.expand_dims(self.label1_emb, axis=0))
        #                 tp1 = find_matchup_max_cosine_similarity4(np.expand_dims(XX[i, :], axis=0), tp_x1,
        #                                                         np.expand_dims(self.label0_emb, axis=0))
        #
        #             if self.args.llm == "gemini":
        #                 res.append(self.gen_results_gemini(self.df_test.iloc[i, self.content_col], df_train0, df_train1, tp0, tp1, n))
        #             elif self.args.llm == "llama":
        #                 res.append(self.gen_results_llama8b(self.df_test.iloc[i, self.content_col], df_train0, df_train1, tp0, tp1, n))
        #             elif self.args.llm == "mistral":
        #                 res.append(self.gen_results_mistral(self.df_test.iloc[i, self.content_col], df_train0, df_train1, tp0, tp1, n))
        #
        #         filename = self.setting + ".txt"
        #         eval_res = evaluation(filename)
        #         eval_res.eval_cola(self.df_test.iloc[:, self.label_col], res)
        #
        #
        #     '''
        #     ### fastvotek
        #     XX = self.testset_emb
        #     # XX=test_embeddings.detach().cpu().numpy()
        #
        #     X0 = []
        #     X1 = []
        #     for i in range(len(self.df_train)):
        #         if self.df_train.iloc[i, self.label_col] == 0:
        #             X0.append(self.df_train.iloc[i, self.addition_col])
        #         else:
        #             X1.append(self.df_train.iloc[i, self.addition_col])
        #
        #     # trainset_emb=train_embeddings.detach().cpu().numpy()
        #
        #     tp_x0 = self.trainset_emb[X0]
        #     tp_x1 = self.trainset_emb[X1]
        #
        #     labelt_emb = self.label0_emb
        #     labelc_emb = self.label1_emb
        #
        #     df_train0 = self.df_train.iloc[X0, :]
        #     df_train1 = self.df_train.iloc[X1, :]
        #
        #     n=self.args.Nshot
        #
        #     tp0 = fast_votek(tp_x0, n, n)
        #     tp1 = fast_votek(tp_x1, n, n)
        #
        #     for j in range(0,4):
        #         res = []
        #
        #         if j==0:
        #             pass
        #         elif j==1:
        #             ### contrastive
        #             for i in range(len(tp_x0)):
        #                 tp_x0[i,:] = tp_x0[i,:] + labelt_emb - labelc_emb
        #             for i in range(len(tp_x1)):
        #                 tp_x1[i,:] = tp_x1[i,:] + labelc_emb - labelt_emb
        #         elif j==2:
        #             ### labelaug
        #             for i in range(len(tp_x0)):
        #                 tp_x0[i,:] = tp_x0[i,:] + labelt_emb
        #             for i in range(len(tp_x1)):
        #                 tp_x1[i,:] = tp_x1[i,:] + labelc_emb
        #         elif j==3:
        #             ### penality
        #             for i in range(len(tp_x0)):
        #                 tp_x0[i, :] = tp_x0[i, :] - labelc_emb
        #             for i in range(len(tp_x1)):
        #                 tp_x1[i, :] = tp_x1[i, :] - labelt_emb
        #
        #         tp0 = fast_votek(tp_x0, n, n)
        #         tp1 = fast_votek(tp_x1, n, n)
        #
        #         # for i in tqdm(range(0, 400)):
        #         for i in tqdm(range(0,len(self.df_test))):
        #             if self.args.llm == "gemini":
        #                 res.append(
        #                     self.gen_results_gemini(self.df_test.iloc[i, self.content_col], df_train0, df_train1, tp0,
        #                                             tp1, n))
        #             elif self.args.llm == "llama":
        #                 res.append(
        #                     self.gen_results_llama8b(self.df_test.iloc[i, self.content_col], df_train0, df_train1, tp0,
        #                                              tp1, n))
        #             elif self.args.llm == "mistral":
        #                 res.append(
        #                     self.gen_results_mistral(self.df_test.iloc[i, self.content_col], df_train0, df_train1, tp0,
        #                                              tp1, n))
        #
        #         '''
        #
        #     # ### selfprompt
        #     # XX = self.testset_emb
        #     # # XX=test_embeddings.detach().cpu().numpy()
        #     #
        #     # X0 = []
        #     # X1 = []
        #     # for i in range(len(self.df_train)):
        #     #     if self.df_train.iloc[i, self.label_col] == 0:
        #     #         X0.append(self.df_train.iloc[i, self.addition_col])
        #     #     else:
        #     #         X1.append(self.df_train.iloc[i, self.addition_col])
        #     #
        #     # # trainset_emb=train_embeddings.detach().cpu().numpy()
        #     #
        #     # tp_x0 = self.trainset_emb[X0]
        #     # tp_x1 = self.trainset_emb[X1]
        #     #
        #     # labelt_emb = self.label0_emb
        #     # labelc_emb = self.label1_emb
        #     #
        #     # df_train0 = self.df_train.iloc[X0, :]
        #     # df_train1 = self.df_train.iloc[X1, :]
        #     #
        #     # n=self.args.Nshot
        #     #
        #     # for j in range(0,4):
        #     #     res = []
        #     #
        #     #     if j==0:
        #     #         pass
        #     #     elif j==1:
        #     #         ### contrastive
        #     #         for i in range(len(tp_x0)):
        #     #             tp_x0[i,:] = tp_x0[i,:] + labelt_emb - labelc_emb
        #     #         for i in range(len(tp_x1)):
        #     #             tp_x1[i,:] = tp_x1[i,:] + labelc_emb - labelt_emb
        #     #     elif j==2:
        #     #         ### labelaug
        #     #         for i in range(len(tp_x0)):
        #     #             tp_x0[i,:] = tp_x0[i,:] + labelt_emb
        #     #         for i in range(len(tp_x1)):
        #     #             tp_x1[i,:] = tp_x1[i,:] + labelc_emb
        #     #     elif j==3:
        #     #         ### penality
        #     #         for i in range(len(tp_x0)):
        #     #             tp_x0[i, :] = tp_x0[i, :] - labelc_emb
        #     #         for i in range(len(tp_x1)):
        #     #             tp_x1[i, :] = tp_x1[i, :] - labelt_emb
        #     #
        #     #     center0 = KMeans(n_clusters=1, random_state=0, n_init="auto").fit(tp_x0)
        #     #     center1 = KMeans(n_clusters=1, random_state=0, n_init="auto").fit(tp_x1)
        #     #
        #     #     tp0 = find_matchup_max_cosine_similarity(center0.cluster_centers_, tp_x0)
        #     #     tp1 = find_matchup_max_cosine_similarity(center1.cluster_centers_, tp_x1)
        #     #
        #     #     tp0 = tp0[:n]
        #     #     tp1 = tp1[:n]
        #     #
        #     #     # for i in tqdm(range(0, 400)):
        #     #     for i in tqdm(range(0,len(self.df_test))):
        #     #         if self.args.llm == "gemini":
        #     #             res.append(
        #     #                 self.gen_results_gemini(self.df_test.iloc[i, self.content_col], df_train0, df_train1, tp0,
        #     #                                         tp1, n))
        #     #         elif self.args.llm == "llama":
        #     #             res.append(
        #     #                 self.gen_results_llama8b(self.df_test.iloc[i, self.content_col], df_train0, df_train1, tp0,
        #     #                                          tp1, n))
        #     #         elif self.args.llm == "mistral":
        #     #             res.append(
        #     #                 self.gen_results_mistral(self.df_test.iloc[i, self.content_col], df_train0, df_train1, tp0,
        #     #                                          tp1, n))
        #     #
        #     #
        #     #
        #     #
        #     #
        #     #
        #     #
        #     #     count = 0
        #     #     filename=self.setting + "_" + str(j)+ ".txt"
        #     #     with open(filename, "w") as file:
        #     #         for i in range(len(res)):
        #     #             if (res[i] == "Class1" or res[i] == "acceptabl"
        #     #                 or res[i] == "acceptable" or res[i] == "\"acceptable" or res[i] == "acceptable " ) and self.df_test.iloc[i, self.label_col] == 1:
        #     #                 pass
        #     #             elif (res[i] == "Class0" or res[i] == "\"unacceptable\""
        #     #                   or res[i] == "unacceptable\n" or res[i] == "unaccept" or res[i] == "unacceptable") and self.df_test.iloc[i, self.label_col] == 0:
        #     #                 pass
        #     #             else:
        #     #                 print(i, '-', self.df_test.iloc[i, self.label_col], '-', res[i])
        #     #                 file.write("{}-{}-{}\n".format(i, self.df_test.iloc[i, self.label_col], res[i]))
        #     #                 count += 1
        #     #         file.write(f"{count}")
        #     #
        #     #     print(count)







class sst2():
    def __init__(self, args):
        self.dataset_name = args.dataset