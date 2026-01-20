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
from load_dataset import load_dataset
import time
from tqdm import tqdm
from mistralai import Mistral
from dotenv import load_dotenv
from llamaapi import LlamaAPI
import google.generativeai as genai
from sklearn.cluster import KMeans
from votek import fast_votek
from function import find_matchup_max_cosine_similarity, find_matchup_max_cosine_similarity2, find_matchup_max_cosine_similarity3, find_matchup_max_cosine_similarity4
warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def load_model(self):
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
            self.API_KEY = os.getenv('LLAMA_API_KEY')
            self.gptmodel_name = "llama3.1-8b"
            self.generation_config = {
                "temperature": 0.01,
                "top_p": 0.5,
                "top_k": 1,
                "max_output_tokens": 2,
                "response_mime_type": "text/plain",
            }

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

    def gen_results_gemini(self, text, df0, df1, group0, group1,n=1, max_retries=4, delay=2):
        retry_count = 0
        genai.configure(api_key=self.API_KEY)
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-001",
            generation_config=self.generation_config,
            system_instruction="You are given a task where there are multiple classes, and for each class, a few labeled examples are provided. Based on these examples, you need to classify a new unseen instance. Choose ONLY one tag and output the tag. Do Not output others."
            # system_instruction="You are a helpful and accurate assistant. You will be provided with a sentence, and your task is to classify as label_0 or label_1(choose either the label_0 or the label_1 tag but NOT both)."
            # safety_settings = Adjust safety settings
            # See https://ai.google.dev/gemini-api/docs/safety-settings
        )
        prompt = None
        # prompt = f'You are a helpful and accurate classification assistant. You will be provided with a sentence, and your task is to choose ONLY one Tag(Label_0 and Label_1) for Given Sentence from instructions. Never output others.\n\n'
        if self.args.dataset=="sst2":
            prompt = self.build_prompt_sst2(group0, 0, "", df0, n) + self.build_prompt_sst2(group1, 1, text, df1, n)
        elif self.args.dataset=="cola":
            prompt = self.build_prompt_cola(group0, 0, "", df0, n) + self.build_prompt_cola(group1, 1, text, df1, n)
        # print(prompt)
        # return prompt

        while retry_count < max_retries:
            try:
                output = model.generate_content(prompt)
                # print(output)
                return output.text
                # print(output)
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(delay)
                retry_count += 1

    def gen_results_mistral(self, text, df0, df1, group0, group1,n=1, max_retries=4, delay=2):
        retry_count = 0
        prompt = None
        # prompt = f'You are a helpful and accurate classification assistant. You will be provided with a sentence, and your task is to choose ONLY one Tag(Label_0 and Label_1) for Given Sentence from instructions. Never output others.\n\n'
        if self.args.dataset=="sst2":
            prompt = self.build_prompt_sst2(group0, 0, "", df0, n) + self.build_prompt_sst2(group1, 1, text, df1, n)
        elif self.args.dataset=="cola":
            prompt = self.build_prompt_cola(group0, 0, "", df0, n) + self.build_prompt_cola(group1, 1, text, df1, n)
        # print(prompt)
        # return prompt

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
                               "content": "You are given a task where there are multiple classes, and for each class, a few labeled examples are provided. Based on these examples, you need to classify a new unseen instance. Choose ONLY one tag and output the tag. Do Not output others."},
                              {"role": "user", "content": prompt},
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
        prompt = None
        # prompt = f'You are a helpful and accurate classification assistant. You will be provided with a sentence, and your task is to choose ONLY one Tag(Label_0 and Label_1) for Given Sentence from instructions. Never output others.\n\n'
        if self.args.dataset=="sst2":
            prompt = self.build_prompt_sst2(group0, 0, "", df0, n) + self.build_prompt_sst2(group1, 1, text, df1, n)
        elif self.args.dataset=="cola":
            prompt = self.build_prompt_cola(group0, 0, "", df0, n) + self.build_prompt_cola(group1, 1, text, df1, n)
        # print(prompt)

        api_token = self.API_KEY
        llama = LlamaAPI(api_token)

        api_request_json = {
            "model": self.gptmodel_name,
            "messages": [
                {"role": "system",
                 "content": "You are given a task where there are multiple classes, and for each class, a few labeled examples are provided. Based on these examples, you need to classify a new unseen instance. Choose ONLY one tag and output the tag. Never output others except the tag."},
                {"role": "user", "content": prompt},
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

    def run_results(self, setting):
        self.load_model()

        # loda dataset
        if self.args.dataset=="sst2":
            file_path = self.args.root_path + self.args.dataset
            self.df_train, self.df_test = load_dataset(self.args.dataset)
            self.content_col = 0
            self.label_col = 1
            self.addition_col = 2
            self.trainset_emb, self.testset_emb = embedding_encoder(self.df_train.iloc[:,self.content_col], self.df_test.iloc[:,self.content_col], file_path)

            self.label0_emb = embedding_encoder_onetime("Negative")
            self.label1_emb = embedding_encoder_onetime("Positive")

        elif self.args.dataset=="cola":
            # the label (0=unacceptable, 1=acceptable)
            file_path = self.args.root_path  + self.args.dataset
            self.df_train, self.df_test = load_dataset(self.args.dataset)
            self.content_col = 3
            self.label_col = 1
            self.addition_col=4
            self.trainset_emb, self.testset_emb = embedding_encoder(self.df_train.iloc[:,self.content_col], self.df_test.iloc[:,self.content_col], file_path)

            self.label0_emb = embedding_encoder_onetime("unacceptable")
            self.label1_emb = embedding_encoder_onetime("acceptable")

        print(self.df_train.shape, self.df_test.shape, self.trainset_emb.shape, self.testset_emb.shape)
        self.df_train['index'] = range(len(self.df_train))
        self.df_test['index'] = range(len(self.df_test))

        if self.args.NwayKshot == True:
            '''
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

            df_train0 = self.df_train.iloc[X0, :]
            df_train1 = self.df_train.iloc[X1, :]

            for j in range(0,4):
                res = []
                for i in tqdm(range(0,len(self.df_test))):
                # for i in range(3):
                    # i=100
                    n = self.args.Nshot
                    tp0 = None
                    tp1 = None
                    if j ==0:
                        tp0=find_matchup_max_cosine_similarity(np.expand_dims(XX[i,:], axis=0),tp_x0)
                        tp1=find_matchup_max_cosine_similarity(np.expand_dims(XX[i,:], axis=0),tp_x1)
                    elif j==1:
                        tp0=find_matchup_max_cosine_similarity2(np.expand_dims(XX[i,:], axis=0),tp_x0,np.expand_dims(self.label0_emb, axis=0),np.expand_dims(self.label1_emb, axis=0))
                        tp1=find_matchup_max_cosine_similarity2(np.expand_dims(XX[i,:], axis=0),tp_x1,np.expand_dims(self.label1_emb, axis=0),np.expand_dims(self.label0_emb, axis=0))
                    elif j==2:
                        tp0=find_matchup_max_cosine_similarity3(np.expand_dims(XX[i,:], axis=0),tp_x0,np.expand_dims(self.label0_emb, axis=0))
                        tp1=find_matchup_max_cosine_similarity3(np.expand_dims(XX[i,:], axis=0),tp_x1,np.expand_dims(self.label1_emb, axis=0))
                    elif j==3:
                        tp0 = find_matchup_max_cosine_similarity4(np.expand_dims(XX[i, :], axis=0), tp_x0,
                                                                np.expand_dims(self.label1_emb, axis=0))
                        tp1 = find_matchup_max_cosine_similarity4(np.expand_dims(XX[i, :], axis=0), tp_x1,
                                                                np.expand_dims(self.label0_emb, axis=0))

                    if self.args.llm == "gemini":
                        res.append(self.gen_results_gemini(self.df_test.iloc[i, self.content_col], df_train0, df_train1, tp0, tp1, n))
                    elif self.args.llm == "llama":
                        res.append(self.gen_results_llama8b(self.df_test.iloc[i, self.content_col], df_train0, df_train1, tp0, tp1, n))
                    elif self.args.llm == "mistral":
                        res.append(self.gen_results_mistral(self.df_test.iloc[i, self.content_col], df_train0, df_train1, tp0, tp1, n))


            '''
            '''
            ### fastvotek
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

            labelt_emb = self.label0_emb
            labelc_emb = self.label1_emb

            df_train0 = self.df_train.iloc[X0, :]
            df_train1 = self.df_train.iloc[X1, :]

            n=self.args.Nshot

            tp0 = fast_votek(tp_x0, n, n)
            tp1 = fast_votek(tp_x1, n, n)

            for j in range(0,4):
                res = []

                if j==0:
                    pass
                elif j==1:
                    ### contrastive
                    for i in range(len(tp_x0)):
                        tp_x0[i,:] = tp_x0[i,:] + labelt_emb - labelc_emb
                    for i in range(len(tp_x1)):
                        tp_x1[i,:] = tp_x1[i,:] + labelc_emb - labelt_emb
                elif j==2:
                    ### labelaug
                    for i in range(len(tp_x0)):
                        tp_x0[i,:] = tp_x0[i,:] + labelt_emb
                    for i in range(len(tp_x1)):
                        tp_x1[i,:] = tp_x1[i,:] + labelc_emb
                elif j==3:
                    ### penality
                    for i in range(len(tp_x0)):
                        tp_x0[i, :] = tp_x0[i, :] - labelc_emb
                    for i in range(len(tp_x1)):
                        tp_x1[i, :] = tp_x1[i, :] - labelt_emb

                tp0 = fast_votek(tp_x0, n, n)
                tp1 = fast_votek(tp_x1, n, n)

                # for i in tqdm(range(0, 400)):
                for i in tqdm(range(0,len(self.df_test))):
                    if self.args.llm == "gemini":
                        res.append(
                            self.gen_results_gemini(self.df_test.iloc[i, self.content_col], df_train0, df_train1, tp0,
                                                    tp1, n))
                    elif self.args.llm == "llama":
                        res.append(
                            self.gen_results_llama8b(self.df_test.iloc[i, self.content_col], df_train0, df_train1, tp0,
                                                     tp1, n))
                    elif self.args.llm == "mistral":
                        res.append(
                            self.gen_results_mistral(self.df_test.iloc[i, self.content_col], df_train0, df_train1, tp0,
                                                     tp1, n))

                '''

            ### selfprompt
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

            labelt_emb = self.label0_emb
            labelc_emb = self.label1_emb

            df_train0 = self.df_train.iloc[X0, :]
            df_train1 = self.df_train.iloc[X1, :]

            n=self.args.Nshot

            for j in range(0,4):
                res = []

                if j==0:
                    pass
                elif j==1:
                    ### contrastive
                    for i in range(len(tp_x0)):
                        tp_x0[i,:] = tp_x0[i,:] + labelt_emb - labelc_emb
                    for i in range(len(tp_x1)):
                        tp_x1[i,:] = tp_x1[i,:] + labelc_emb - labelt_emb
                elif j==2:
                    ### labelaug
                    for i in range(len(tp_x0)):
                        tp_x0[i,:] = tp_x0[i,:] + labelt_emb
                    for i in range(len(tp_x1)):
                        tp_x1[i,:] = tp_x1[i,:] + labelc_emb
                elif j==3:
                    ### penality
                    for i in range(len(tp_x0)):
                        tp_x0[i, :] = tp_x0[i, :] - labelc_emb
                    for i in range(len(tp_x1)):
                        tp_x1[i, :] = tp_x1[i, :] - labelt_emb

                center0 = KMeans(n_clusters=1, random_state=0, n_init="auto").fit(tp_x0)
                center1 = KMeans(n_clusters=1, random_state=0, n_init="auto").fit(tp_x1)

                tp0 = find_matchup_max_cosine_similarity(center0.cluster_centers_, tp_x0)
                tp1 = find_matchup_max_cosine_similarity(center1.cluster_centers_, tp_x1)

                tp0 = tp0[:n]
                tp1 = tp1[:n]

                # for i in tqdm(range(0, 400)):
                for i in tqdm(range(0,len(self.df_test))):
                    if self.args.llm == "gemini":
                        res.append(
                            self.gen_results_gemini(self.df_test.iloc[i, self.content_col], df_train0, df_train1, tp0,
                                                    tp1, n))
                    elif self.args.llm == "llama":
                        res.append(
                            self.gen_results_llama8b(self.df_test.iloc[i, self.content_col], df_train0, df_train1, tp0,
                                                     tp1, n))
                    elif self.args.llm == "mistral":
                        res.append(
                            self.gen_results_mistral(self.df_test.iloc[i, self.content_col], df_train0, df_train1, tp0,
                                                     tp1, n))







                count = 0
                filename=self.setting + "_" + str(j)+ ".txt"
                with open(filename, "w") as file:
                    for i in range(len(res)):
                        if (res[i] == "Class1" or res[i] == "acceptabl"
                            or res[i] == "acceptable" or res[i] == "\"acceptable" or res[i] == "acceptable " ) and self.df_test.iloc[i, self.label_col] == 1:
                            pass
                        elif (res[i] == "Class0" or res[i] == "\"unacceptable\""
                              or res[i] == "unacceptable\n" or res[i] == "unaccept" or res[i] == "unacceptable") and self.df_test.iloc[i, self.label_col] == 0:
                            pass
                        else:
                            print(i, '-', self.df_test.iloc[i, self.label_col], '-', res[i])
                            file.write("{}-{}-{}\n".format(i, self.df_test.iloc[i, self.label_col], res[i]))
                            count += 1
                    file.write(f"{count}")

                print(count)



class sst2():
    def __init__(self, args):
        self.dataset_name = args.dataset