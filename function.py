import pandas as pd
import numpy as np

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics.pairwise import euclidean_distances,cosine_similarity

import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer

# def find_matchup_max_cosine_similarity(centers,X0, sim):
#
#   dist=None
#   if sim=='cosine':
#     dist=cosine_similarity(centers,X0)
#   elif sim=='euclidean':
#     dist=euclidean_distances(centers,X0)
#
#
#   index=np.zeros((32),dtype=np.int16)
#   maxdist=np.zeros((32),dtype=np.float64)
#
#   maxdist[0]=0
#   index-=1
#   maxxx=-1
#   tp_index=-1
#
#   for j in range(0,len(maxdist)):
#     if j==0:
#       maxxx=0.99
#       tp_index-=1
#     else:
#       maxxx=maxdist[j-1]
#       tp_index=index[j-1]
#
#     for i in range(len(dist[0])):
#       if (dist[0][i]<maxxx) and (dist[0][i]>maxdist[j]) and (tp_index!=i):
#
#         index[j]=i
#         maxdist[j]=dist[0][i]
#
#   # print("MAXDIST:",index, maxdist)
#
#   if sim=='euclidean':
#     print("euclidean:", index, maxdist)
#
#   return index,maxdist


def find_matchup_max_cosine_similarity(centers,X0, sim):

  dist=None
  if sim=='cosine':
    dist=cosine_similarity(centers,X0)
  elif sim=='euclidean':
    dist=euclidean_distances(centers,X0)
  # print(dist.shape)

  index=np.zeros((3),dtype=np.int16)
  maxdist=np.zeros((3),dtype=np.float64)

  mp={}
  for i in range(len(dist[0])):
    mp[i]=dist[0][i]

  tp=sorted(mp.items(),key=lambda item: item[1],reverse=True)

  for i in range(len(maxdist)):
    maxdist[i]=tp[i][1]
    index[i]=tp[i][0]

  print(index, maxdist)
  # if sim=='euclidean':
  #   print("euclidean:", index, maxdist)

  return index,maxdist




def find_matchup_max_cosine_similarity2(centers,X0,labelt_emb,labelc_emb, sim, w1, w2, wo):
  # centers: query

  dist=None
  tp_dist=None
  tp_dist2=None
  dist_copy = None
  if sim=='cosine':
    dist=cosine_similarity(centers,X0)
    dist_copy=dist
    tp_dist= cosine_similarity(labelt_emb,X0)
    tp_dist2= cosine_similarity(labelc_emb,X0)
  elif sim=='euclidean':
    dist=euclidean_distances(centers,X0)
    tp_dist= euclidean_distances(labelt_emb,X0)
    tp_dist2= euclidean_distances(labelc_emb,X0)

  # if not wo:
  #   dist=dist+ w1 * tp_dist - w2 * tp_dist2
  # else:
  #   dist = w1 * tp_dist - w2 * tp_dist2

  index=np.zeros((32),dtype=np.int16)
  maxdist=np.zeros((32),dtype=np.float64)

  mp={}
  for i in range(len(dist[0])):
    mp[i]=dist[0][i]

  tp=sorted(mp.items(),key=lambda item: item[1],reverse=True)

  for i in range(len(maxdist)):
    maxdist[i]=tp[i][1]
    index[i]=tp[i][0]

  for i in range(3):
    print(index[i])
    print(dist_copy[0,index[i]])
    print(tp_dist[0,index[i]])
    print(tp_dist2[0, index[i]])
  # print("MAXDIST:",index, maxdist)
  # if sim=='euclidean':
  #   print("euclidean:", index, maxdist)

  return index,maxdist

def find_matchup_max_cosine_similarity22(centers,X0,label_emb,label_index,n, sim, w1, w2, wo):
  # centers: query
  dist = None
  tp_dist = None

  if sim == 'cosine':
    dist=cosine_similarity(centers,X0)
    tp_dist= cosine_similarity(label_emb,X0)

  elif sim=='euclidean':
    dist=euclidean_distances(centers,X0)
    tp_dist= euclidean_distances(label_emb,X0)


  tp_dist2 = np.zeros((dist.shape),dtype=float)

  for i in range(n):
    if i!=label_index:
      tp_dist2=tp_dist2+tp_dist[i]

  tp_dist2 = tp_dist2 /(n-1)

  if not wo:
    dist=dist+ w1 * tp_dist[label_index,:]- w2 * tp_dist2
  else:
    dist = w1 * tp_dist[label_index,:]- w2 * tp_dist2

  index=np.zeros((32),dtype=np.int16)
  maxdist=np.zeros((32),dtype=np.float64)

  mp={}
  for i in range(len(dist[0])):
    mp[i]=dist[0][i]

  tp=sorted(mp.items(),key=lambda item: item[1],reverse=True)

  for i in range(len(maxdist)):
    maxdist[i]=tp[i][1]
    index[i]=tp[i][0]

  # print("MAXDIST:",index, maxdist)
  # if sim=='euclidean':
  #   print("euclidean:", index, maxdist)

  return index, maxdist


def find_matchup_max_cosine_similarity3(centers,X0,labelt_emb, sim, w1, wo):
  # centers: query

  dist = None
  tp_dist = None
  tp_dist2 = None

  if sim == 'cosine':
    dist=cosine_similarity(centers,X0)
    tp_dist= cosine_similarity(labelt_emb,X0)

  elif sim=='euclidean':
    dist=euclidean_distances(centers,X0)
    tp_dist= euclidean_distances(labelt_emb,X0)

  if not wo:
    dist=dist+ w1 * tp_dist
  else:
    dist = w1 * tp_dist

  index=np.zeros((32),dtype=np.int16)
  maxdist=np.zeros((32),dtype=np.float64)

  mp={}
  for i in range(len(dist[0])):
    mp[i]=dist[0][i]

  tp=sorted(mp.items(),key=lambda item: item[1],reverse=True)

  for i in range(len(maxdist)):
    maxdist[i]=tp[i][1]
    index[i]=tp[i][0]

  # print("MAXDIST:",index, maxdist)
  # if sim=='euclidean':
  #   print("euclidean:", index, maxdist)

  return index,maxdist

def find_matchup_max_cosine_similarity33(centers,X0,label_emb,label_index,n, sim, w1, wo):
  # centers: query

  dist = None
  tp_dist = None

  if sim == 'cosine':

    dist=cosine_similarity(centers,X0)
    tp_dist= cosine_similarity(label_emb,X0)

  elif sim=='euclidean':
    dist=euclidean_distances(centers,X0)
    tp_dist= euclidean_distances(label_emb,X0)

  tp_dist2 = np.zeros((dist.shape),dtype=float)

  if not wo:
    dist=dist+w1 * tp_dist[label_index,:]
  else:
    dist =tp_dist2+ w1 * tp_dist[label_index,:]

  index=np.zeros((32),dtype=np.int16)
  maxdist=np.zeros((32),dtype=np.float64)

  mp={}
  for i in range(len(dist[0])):
    mp[i]=dist[0][i]

  tp=sorted(mp.items(),key=lambda item: item[1],reverse=True)

  for i in range(len(maxdist)):
    maxdist[i]=tp[i][1]
    index[i]=tp[i][0]

  # print("MAXDIST:",index, maxdist)
  # if sim=='euclidean':
  #   print("euclidean:", index, maxdist)
  return index,maxdist


def find_matchup_max_cosine_similarity4(centers,X0,labelc_emb, sim, w2, wo):
  # centers: query

  dist = None
  tp_dist = None

  if sim == 'cosine':
    dist=cosine_similarity(centers,X0)
    tp_dist= cosine_similarity(labelc_emb,X0)
  elif sim == 'euclidean':
    dist = euclidean_distances(centers,X0)
    tp_dist = euclidean_distances(labelc_emb,X0)

  if not wo:
    dist=dist-w2 * tp_dist
  else:
    dist = -w2 * tp_dist

  index=np.zeros((32),dtype=np.int16)
  maxdist=np.zeros((32),dtype=np.float64)

  mp={}
  for i in range(len(dist[0])):
    mp[i]=dist[0][i]

  tp=sorted(mp.items(),key=lambda item: item[1],reverse=True)

  for i in range(len(maxdist)):
    maxdist[i]=tp[i][1]
    index[i]=tp[i][0]

  # print("MAXDIST:",index, maxdist)
  # if sim=='euclidean':
  #   print("euclidean:", index, maxdist)
  return index,maxdist


def find_matchup_max_cosine_similarity44(centers,X0,label_emb,label_index,n, sim, w2, wo):
  # centers: query

  dist = None
  tp_dist = None

  if sim == 'cosine':
    dist=cosine_similarity(centers,X0)
    tp_dist= cosine_similarity(label_emb,X0)

  elif sim == 'euclidean':
    dist=euclidean_distances(centers,X0)
    tp_dist= euclidean_distances(label_emb,X0)


  tp_dist2 = np.zeros((dist.shape),dtype=float)

  for i in range(n):
    if i!=label_index:
      tp_dist2=tp_dist2+tp_dist[i]

  tp_dist2 = tp_dist2 /(n-1)

  if not wo:
    dist=dist-w2 * tp_dist2
  else:
    dist = -w2 * tp_dist2

  index=np.zeros((32),dtype=np.int16)
  maxdist=np.zeros((32),dtype=np.float64)

  mp={}
  for i in range(len(dist[0])):
    mp[i]=dist[0][i]

  tp=sorted(mp.items(),key=lambda item: item[1],reverse=True)

  for i in range(len(maxdist)):
    maxdist[i]=tp[i][1]
    index[i]=tp[i][0]

  # print("MAXDIST:",index, maxdist)
  # if sim=='euclidean':
  #   print("euclidean:", index, maxdist)
  return index,maxdist

def find_matchup_max_cosine_similarity5(centers,X0,labelt_emb,labelc_emb, sim, w3, wo):

  # centers: query

  dist = None
  tp_dist = None
  tp_dist2 = None
  dist_copy=None
  tp_dist_copy = None
  tp_dist2_copy = None
  if sim == 'cosine':

    dist=cosine_similarity(centers,X0)
    dist_copy=dist
    tp_dist= np.exp(cosine_similarity(labelt_emb,X0))
    tp_dist2= np.exp(cosine_similarity(labelc_emb,X0))
    tp_dist_copy = cosine_similarity(labelt_emb,X0)
    tp_dist2_copy = cosine_similarity(labelc_emb,X0)

  elif sim == 'euclidean':
    dist=euclidean_distances(centers,X0)
    tp_dist= np.exp(euclidean_distances(labelt_emb,X0))
    tp_dist2= np.exp(euclidean_distances(labelc_emb,X0))

  if not wo:
    dist=dist + w3 * np.log(tp_dist/tp_dist2)
  else:
    dist = w3 * np.log(tp_dist/tp_dist2)

  index=np.zeros((32),dtype=np.int16)
  maxdist=np.zeros((32),dtype=np.float64)

  mp={}
  for i in range(len(dist[0])):
    mp[i]=dist[0][i]

  tp=sorted(mp.items(),key=lambda item: item[1],reverse=True)

  for i in range(len(maxdist)):
    maxdist[i]=tp[i][1]
    index[i]=tp[i][0]

  for i in range(3):
    print(index[i])
    print(dist_copy[0,index[i]])
    print(tp_dist_copy[0,index[i]])
    print(tp_dist2_copy[0,index[i]])

  # print("euclidean:", index, maxdist)
  # print("MAXDIST:",index, maxdist)
  # if sim=='euclidean':
  #   print("euclidean:", index, maxdist)
  return index,maxdist


def find_matchup_max_cosine_similarity6(centers,X0,label_emb,label_index,n,sim, w3, wo):
  # centers: query

  dist = None
  tp_dist = None
  tp_dist2 = None

  if sim == 'cosine':
    dist=cosine_similarity(centers,X0)
    tp_dist= np.exp(cosine_similarity(label_emb,X0))

  elif sim == 'euclidean':
    dist=euclidean_distances(centers,X0)
    tp_dist= np.exp(euclidean_distances(label_emb,X0))

  tp_dist2 = np.zeros((dist.shape),dtype=float)

  for i in range(n):
    if i!=label_index:
      tp_dist2=tp_dist2+tp_dist[i]

  tp_dist2 = tp_dist2 /(n-1)

  if not wo:
    dist=dist+ w3 * np.log(tp_dist / tp_dist2)
  else:
    dist = w3 * np.log(tp_dist/tp_dist2)

  index=np.zeros((32),dtype=np.int16)
  maxdist=np.zeros((32),dtype=np.float64)

  mp={}
  for i in range(len(dist[0])):
    mp[i]=dist[0][i]

  tp=sorted(mp.items(),key=lambda item: item[1],reverse=True)

  for i in range(len(maxdist)):
    maxdist[i]=tp[i][1]
    index[i]=tp[i][0]

  # print("MAXDIST:",index, maxdist)
  # if sim=='euclidean':
  #   print("euclidean:", index, maxdist)
  return index,maxdist