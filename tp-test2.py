import time
from tqdm import tqdm
from transformers import GPTJForCausalLM, AutoTokenizer
import torch
from load_dataset import load_dataset
import numpy as np
from function import find_matchup_max_cosine_similarity
import pandas as pd
import numpy as np

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics.pairwise import euclidean_distances,cosine_similarity

import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer

def build_prompt(label_data, label, text, database,N=32):
    #prompt = f"# Label_{label} Instruction:\n"
    prompt=""
    # prompt = f"# Class {label}: Negative\n"
    if label==0:
      prompt += f"Class{label}: Negative\n"

      for idx, num in enumerate(label_data[:N]):
        #print(idx)
        #print(num)
        #print(label_data[num, 3])
        #print(database.iloc[num, 2])
        #prompt += "\n".join(f"{idx+1}. {database.iloc[num, 2]}")
        #prompt += f"{idx+1}. {database.iloc[num, 1]}"
        prompt += f"  {idx+1}. Example {idx+1}: \"{database.iloc[num, 1]}\" -> \"Negative\""
        prompt += "\n"

    elif label==1:
      prompt += f"Class{label}: Positive\n"

      for idx, num in enumerate(label_data[:N]):
        #print(idx)
        #print(num)
        #print(label_data[num, 3])
        #print(database.iloc[num, 2])
        #prompt += "\n".join(f"{idx+1}. {database.iloc[num, 2]}")
        #prompt += f"{idx+1}. {database.iloc[num, 1]}"
        prompt += f"  {idx+1}. Example {idx+1}: \"{database.iloc[num, 1]}\" -> \"Positive\""
        prompt += "\n"

    # if len(label_data)!=0:

      #prompt+= f"Label_1 represents positive in sentiment analysis.\n\n"
      # prompt+= f"Here are Label_{label} examples:\n"
    if text!="":

      prompt += f"Query: \"{text}\"\n"
      prompt += f"Prediction: "
    return prompt

def gen_results(model, tokenizer, text, df0,df1, group0, group1, max_retries=1, delay=2):
    device = "cpu"
    retry_count = 0
    prompt = f'You are given a task where there are multiple classes, and for each class, a few labeled examples are provided. Based on these examples, you need to classify a new unseen instance. Choose ONLY one tag. Never output others.\n\n'
    prompt = prompt + build_prompt(group0, 0, "", df0, n) + build_prompt(group1, 1, text, df1,n)
    # print(prompt)
    # return prompt

    while retry_count < max_retries:
      try:
          # output = model.generate_content(prompt)
          input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

          gen_tokens = model.generate(
              input_ids,
              do_sample=True,
              temperature=0.2,
              max_length=5000,
          )
          # print(output)
          return tokenizer.batch_decode(gen_tokens)[0]
      except Exception as e:
          print(f"Error: {e}")
          time.sleep(delay)
          retry_count += 1


def load_model():
    device = "cpu"
    model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

    return model, tokenizer



if __name__ == "__main__":
    df_train , df_test = load_dataset('sst2')
    print(df_train.shape)

    # train_emb = np.load('./datasets/sst2/trainemb.npy', allow_pickle=True)
    # test_emb = np.load('./datasets/sst2/testemb.npy', allow_pickle=True)
    #
    # print(train_emb.shape, test_emb.shape)
    # np.save('./datasets/sst2/trainemb.npy', trainset_emb)
    # np.save('./datasets/sst2/testemb.npy', testset_emb)

    # trainset['embedding'] = trainset_emb[:,:]
    # testset['embedding'] = testset_emb[:,:]
    # df_train['index']=range(len(df_train))
    # df_test['index']=range(len(df_test))

    lab = np.zeros((2), dtype=np.int32)

    for i in range(len(df_test)):

        lab[df_test.iloc[i,0]] += 1
    print(lab)


    ### pre-processing
    # XX = test_emb
    # # XX=test_embeddings.detach().cpu().numpy()
    # X0 = []
    # X1 = []
    # for i in range(len(df_train)):
    #     if df_train.iloc[i, 0] == 0:
    #         X0.append(df_train.iloc[i, 2])
    #     else:
    #         X1.append(df_train.iloc[i, 2])
    #
    # # trainset_emb=train_embeddings.detach().cpu().numpy()
    #
    # tp_x0 = train_emb[X0]
    # tp_x1 = train_emb[X1]
    #
    # # labelt_emb = np.expand_dims((label0_emb), axis=0)
    # # labelc_emb = np.expand_dims((label1_emb), axis=0)
    #
    # # contrastive
    # # dist=cosine_similarity(centers,X0)
    # # tp_dist00 = cosine_similarity(labelt_emb,tp_x0)
    # # tp_dist01 = cosine_similarity(labelc_emb,tp_x0)
    #
    # # tp_dist10 = cosine_similarity(labelt_emb,tp_x1)
    # # tp_dist11 = cosine_similarity(labelc_emb,tp_x1)
    #
    # # for i in range(len(tp_x0)):
    # #     tp_x0[i, :] = tp_x0[i, :] + labelt_emb[0] - labelc_emb[0]
    # # for i in range(len(tp_x1)):
    # #     tp_x1[i, :] = tp_x1[i, :] + labelc_emb[0] - labelt_emb[0]
    #
    # df_train0 = df_train.iloc[X0, :]
    # df_train1 = df_train.iloc[X1, :]
    #
    #
    #
    # ### self-prompting
    # res=[]
    # center0 = KMeans(n_clusters=1, random_state=0, n_init="auto").fit(tp_x0)
    # center1 = KMeans(n_clusters=1, random_state=0, n_init="auto").fit(tp_x1)
    #
    # tp0=find_matchup_max_cosine_similarity(center0.cluster_centers_,tp_x0)
    # tp1=find_matchup_max_cosine_similarity(center1.cluster_centers_,tp_x1)
    #
    # n=3
    # tp0=tp0[:n]
    # tp1=tp1[:n]
    #
    # model, tokenizer = load_model()
    # for i in tqdm(range(len(df_test))):
    # # for i in range(1):
    #   #i=82
    #
    #   # tp0=find_matchup_max_cosine_similarity(np.expand_dims(XX[i,:], axis=0),tp_x0)
    #   # tp1=find_matchup_max_cosine_similarity(np.expand_dims(XX[i,:], axis=0),tp_x1)
    #
    #   #print(tp_x0)
    #   #tp_x1=find_matchup_max_cosine_similarity(np.expand_dims(XX[i,:], axis=0),X1)
    #   # print(tp0)
    #   # print(tp1)
    #
    #   #print(df0.iloc[tp_x0[i],2])
    #   #tp_x0=tp_x0[8:]
    #
    #   res.append(gen_results(model, tokenizer, df_test.iloc[i, 1], df_train0, df_train1, tp0, tp1))
    #
    # count = 0
    # for i in range(len(res)):
    #     if (res[i] == 'Positive \n' or res[i] == 'Positive') and df_test.iloc[i, 0] == 1:
    #         pass
    #     elif (res[i] == 'Negative \n' or res[i] == 'Label_0 \n') and df_test.iloc[i, 0] == 0:
    #         pass
    #     else:
    #         print(i, '-', df_test.iloc[i, 0], '-', res[i])
    #         count += 1
    #
    # print(count)
