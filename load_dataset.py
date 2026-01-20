import pandas as pd
import numpy as np
import datasets

from encoder import embedding_encoder
def load_dataset(data_filename):
    df_train = None
    df_test = None
    if data_filename=="sst2":
        ### 0 negative; 1 positive
        df_train = pd.read_csv("./datasets/sst2/train.tsv", header=None, delimiter="\t")
        df_test = pd.read_csv("./datasets/sst2/test.tsv", header=None, delimiter="\t")

    elif data_filename=="cola":
        # the label (0=unacceptable, 1=acceptable)
        df_train = pd.read_csv('./datasets/cola/in_domain_train.tsv', header=None, delimiter="\t")
        df_test = pd.read_csv('./datasets/cola/in_domain_dev.tsv', header=None, delimiter="\t")
        df_test2 = pd.read_csv('./datasets/cola/out_of_domain_dev.tsv', header=None, delimiter="\t")
        print(df_train.iloc[0,:])
        print(df_test.shape, df_test2.shape)
        df_test = pd.concat([df_test, df_test2], ignore_index=True)
        print(df_test.shape)
        print(sum(df_test.iloc[:,1]))
    elif data_filename == "emotion":

        ds = datasets.load_dataset("dair-ai/emotion", "split")
        df_train = pd.DataFrame(ds['train'])
        df_test = pd.DataFrame(ds['test'])
        print(df_train.iloc[0, :])
        print(df_train.shape)

    elif data_filename == "bbc":

        ds_train = datasets.load_dataset("SetFit/bbc-news", split='train')
        ds_test = datasets.load_dataset("SetFit/bbc-news", split='test')
        df_train = pd.DataFrame(ds_train)
        df_test = pd.DataFrame(ds_test)
        print(df_train.iloc[0, :])
        print(df_train.shape)

    return df_train,df_test

def load_embedding_dataset(data_filename):
    train_emb = None
    test_emb = None
    if data_filename=="sst2":
        train_emb = np.load('./datasets/sst2/trainemb.npy', allow_pickle=True)
        test_emb = np.load('./datasets/sst2/testemb.npy', allow_pickle=True)

    elif data_filename == "emotion":
        train_emb = np.load('./datasets/emotion/trainemb.npy', allow_pickle=True)
        test_emb = np.load('./datasets/emotion/testemb.npy', allow_pickle=True)

    return train_emb,test_emb


if __name__ == "__main__":
    trainset , testset = load_dataset("sst2")
    print(trainset.shape, testset.shape)

    trainset , testset = load_dataset("cola")
    print(trainset.shape, testset.shape)

    trainset , testset = load_dataset("emotion")
    print(trainset.shape, testset.shape)

    trainset , testset = load_dataset("bbc")
    print(trainset.shape, testset.shape)
    #
    # train_emb = np.load('./datasets/sst2/trainemb.npy', allow_pickle=True)
    # test_emb = np.load('./datasets/sst2/testemb.npy', allow_pickle=True)
    #
    # print(train_emb.shape, test_emb.shape)
    # # np.save('./datasets/sst2/trainemb.npy', trainset_emb)
    # # np.save('./datasets/sst2/testemb.npy', testset_emb)
    #
    # # trainset['embedding'] = trainset_emb[:,:]
    # # testset['embedding'] = testset_emb[:,:]
    # trainset['index']=range(len(trainset))
    # testset['index']=range(len(testset))

    # for i in range(len(testset)):
    # testset.'' = i

    # print(testset.iloc[10,:])


