import pandas as pd
import numpy as np

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from tqdm import tqdm
from load_dataset import load_dataset,load_embedding_dataset

def find_matchup_max_cosine_similarity(centers, X0):
    dist = cosine_similarity(centers, X0)

    index = np.zeros((32), dtype=np.int16)
    maxdist = np.zeros((32), dtype=np.float64)

    maxdist[0] = 0
    index -= 1
    maxxx = -1
    tp_index = -1

    for j in range(0, len(maxdist)):
        if j == 0:
            maxxx = 0.99
            tp_index -= 1
        else:
            maxxx = maxdist[j - 1]
            tp_index = index[j - 1]

        for i in range(len(dist[0])):
            if (dist[0][i] < maxxx) and (dist[0][i] > maxdist[j]) and (tp_index != i):
                index[j] = i
                maxdist[j] = dist[0][i]

    # print("MAXDIST:",index, maxdist)

    return index, maxdist

def find_matchup_max_cosine_similarity2(centers,X0,labelt_emb,labelc_emb):
  # centers: query
  dist=cosine_similarity(centers,X0)
  tp_dist= cosine_similarity(labelt_emb,X0)
  tp_dist2= cosine_similarity(labelc_emb,X0)

  dist=dist+tp_dist-tp_dist2

  index=np.zeros((32),dtype=np.int16)
  maxdist=np.zeros((32),dtype=np.float64)

  maxdist[0]=0
  index-=1
  maxxx=-1
  tp_index=-1

  for j in range(0,len(maxdist)):
    if j==0:
      maxxx=100.0
      tp_index-=1
    else:
      maxxx=maxdist[j-1]
      tp_index=index[j-1]

    for i in range(len(dist[0])):
      if (dist[0][i]<maxxx) and (dist[0][i]>maxdist[j]) and (tp_index!=i):

        index[j]=i
        maxdist[j]=dist[0][i]

  # print("MAXDIST:",index, maxdist)

  return index


def find_matchup_max_cosine_similarity3(centers,X0,labelt_emb):
  # centers: query
  dist=cosine_similarity(centers,X0)
  tp_dist= cosine_similarity(labelt_emb,X0)

  dist=dist+tp_dist

  index=np.zeros((32),dtype=np.int16)
  maxdist=np.zeros((32),dtype=np.float64)

  maxdist[0]=0
  index-=1
  maxxx=-1
  tp_index=-1

  for j in range(0,len(maxdist)):
    if j==0:
      maxxx=100.0
      tp_index-=1
    else:
      maxxx=maxdist[j-1]
      tp_index=index[j-1]

    for i in range(len(dist[0])):
      if (dist[0][i]<maxxx) and (dist[0][i]>maxdist[j]) and (tp_index!=i):

        index[j]=i
        maxdist[j]=dist[0][i]

  # print("MAXDIST:",index, maxdist)

  return index

def find_matchup_max_cosine_similarity4(centers,X0,labelc_emb):
  # centers: query
  dist=cosine_similarity(centers,X0)
  tp_dist= cosine_similarity(labelc_emb,X0)

  dist=dist-tp_dist

  index=np.zeros((32),dtype=np.int16)
  maxdist=np.zeros((32),dtype=np.float64)

  maxdist[0]=0
  index-=1
  maxxx=-1
  tp_index=-1

  for j in range(0,len(maxdist)):
    if j==0:
      maxxx=100.0
      tp_index-=1
    else:
      maxxx=maxdist[j-1]
      tp_index=index[j-1]

    for i in range(len(dist[0])):
      if (dist[0][i]<maxxx) and (dist[0][i]>maxdist[j]) and (tp_index!=i):

        index[j]=i
        maxdist[j]=dist[0][i]

  # print("MAXDIST:",index, maxdist)

  return index




def knnvote(df_test, XX, tp_x0, tp_x1, n=1):
    res = []
    k=n
    for i in tqdm(range(len(df_test))):
        # for i in range(1):
        # i=82
        tp0, tp0_dist = find_matchup_max_cosine_similarity(np.expand_dims(XX[i, :], axis=0), tp_x0)
        tp1, tp1_dist = find_matchup_max_cosine_similarity(np.expand_dims(XX[i, :], axis=0), tp_x1)
        # print(tp_x0)
        # tp_x1=find_matchup_max_cosine_similarity(np.expand_dims(XX[i,:], axis=0),X1)
        # print(tp0)
        # print(tp1)

        # print(df0.iloc[tp_x0[i],2])
        # tp_x0=tp_x0[8:]
        tp0_dist = tp0_dist[:k]
        tp1_dist = tp1_dist[:k]

        if sum(tp0_dist) > sum(tp1_dist):
            res.append(0)
        else:
            res.append(1)
    return res



if __name__ == "__main__":
    df_train, df_test = load_dataset()
    train_emb, test_emb = load_embedding_dataset()

    df_train['index'] = range(len(df_train))
    df_test['index'] = range(len(df_test))

    X0 = []
    X1 = []
    for i in range(len(df_train)):
        if df_train.iloc[i, 0] == 0:
            X0.append(df_train.iloc[i, 2])
        else:
            X1.append(df_train.iloc[i, 2])

    res = knnvote(df_test, test_emb, train_emb[X0], train_emb[X1], n=3)

    count = 0
    for i in range(len(res)):
        if res[i] == 1 and df_test.iloc[i, 0] == 1:
            pass
        elif res[i] == 0 and df_test.iloc[i, 0] == 0:
            pass
        else:
            print(i, '-', df_test.iloc[i, 0], '-', res[i])
            count += 1

    print(count)

