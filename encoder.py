from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

def embedding_encoder(train_inputs,test_inputs,root_path,embedding_model):
    train_emb = None
    test_emb = None

    if embedding_model=="all-MiniLM-L6-v2":
        # 1. Load a pretrained Sentence Transformer model
        model = SentenceTransformer("all-MiniLM-L6-v2")
        train_file = Path(root_path + "/trainemb.npy")
        test_file = Path(root_path + "/testemb.npy")

        if train_file.is_file():
            train_emb = np.load(train_file, allow_pickle=True)
        else:
            # The sentences to encode

            # 2. Calculate embeddings by calling model.encode()
            train_emb = model.encode(train_inputs)
            print("train_embedding_shape:", train_emb.shape)

            np.save(train_file, train_emb)

        if test_file.is_file():
            test_emb = np.load(test_file, allow_pickle=True)
        else:
            # The sentences to encode

            # 2. Calculate embeddings by calling model.encode()
            test_emb = model.encode(test_inputs)
            print("test_embedding_shape:", test_emb.shape)

            np.save(test_file, test_emb)
    elif embedding_model=="mpnet":
        # 1. Load a pretrained Sentence Transformer model
        train_file = Path(root_path + "/trainemb_mpnet.npy")
        test_file = Path(root_path + "/testemb_mpnet.npy")

        model = SentenceTransformer("all-mpnet-base-v2")

        if train_file.is_file():
            train_emb = np.load(train_file, allow_pickle=True)
        else:
            # The sentences to encode

            # 2. Calculate embeddings by calling model.encode()
            train_emb = model.encode(train_inputs)
            print("train_embedding_shape:", train_emb.shape)

            np.save(train_file, train_emb)

        if test_file.is_file():
            test_emb = np.load(test_file, allow_pickle=True)
        else:
            # The sentences to encode

            # 2. Calculate embeddings by calling model.encode()
            test_emb = model.encode(test_inputs)
            print("test_embedding_shape:", test_emb.shape)

            np.save(test_file, test_emb)
    elif embedding_model=="bert":
        # 1. Load a pretrained Sentence Transformer model
        train_file = Path(root_path + "/trainemb_bert.npy")
        test_file = Path(root_path + "/testemb_bert.npy")

        # Import our models. The package will take care of downloading the models automatically
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained("bert-base-uncased")


        if train_file.is_file():
            train_emb = np.load(train_file, allow_pickle=True)
        else:
            # The sentences to encode
            # Tokenize input texts

            train_inputs_tp = tokenizer(list(train_inputs), padding=True, truncation=True, return_tensors="pt")

            # Get the embeddings
            with torch.no_grad():
                train_embeddings = model(**train_inputs_tp, output_hidden_states=True, return_dict=True).pooler_output

            train_emb=train_embeddings.numpy()

            print("train_embedding_shape:",train_emb.shape)

            np.save(train_file,train_emb)

        if test_file.is_file():
            test_emb = np.load(test_file, allow_pickle=True)
        else:
            # The sentences to encode
            test_inputs_tp = tokenizer(list(test_inputs), padding=True, truncation=True, return_tensors="pt")
            # 2. Calculate embeddings by calling model.encode()
            with torch.no_grad():
                test_embeddings = model(**test_inputs_tp, output_hidden_states=True, return_dict=True).pooler_output
            test_emb = test_embeddings.numpy()
            print("test_embedding_shape:",test_emb.shape)

            np.save(test_file,test_emb)
    return train_emb, test_emb

def embedding_encoder_onetime(inputs,embedding_model):
    if embedding_model=="all-MiniLM-L6-v2":
        # 1. Load a pretrained Sentence Transformer model
        model = SentenceTransformer("all-MiniLM-L6-v2")

        # The sentences to encode
        # 2. Calculate embeddings by calling model.encode()
        embeddings = model.encode(inputs)
        print("embedding_onetime", embeddings.shape)
        return embeddings

    elif embedding_model=="mpnet":
        # 1. Load a pretrained Sentence Transformer model
        model = SentenceTransformer("all-mpnet-base-v2")

        # The sentences to encode
        # 2. Calculate embeddings by calling model.encode()
        embeddings = model.encode(inputs)
        print("embedding_onetime", embeddings.shape)
        return embeddings

    elif embedding_model == "bert":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained("bert-base-uncased")

        texts = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            embeddings = model(**texts, output_hidden_states=True, return_dict=True).pooler_output

        if embeddings.shape[0]==1:
            embeddings=embeddings[0]
        print("embedding_onetime", embeddings.shape)
    return embeddings.numpy()

if __name__ == "__main__":
    pass