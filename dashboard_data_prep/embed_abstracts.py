import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification
from src.embedding import generate_embeddings
import torch
from tqdm import tqdm
import time

def embed_via_bert(abstracts, checkpoint, save_embeddings_to):
    ### BERT
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("running on device: {}".format(device))

    local_checkpoint = False

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint)
    if local_checkpoint:
        model = AutoModelForTokenClassification.from_pretrained(checkpoint)
    mat = np.empty([len(abstracts), 768])
    abstract_batch = abstracts
    # Start timing
    start_time = time.time()

    # Process abstracts in batches and track progress with tqdm
    for i, abst in enumerate(tqdm(abstract_batch, desc="Generating Embeddings")):
        if local_checkpoint:
            _, mat[i], _ = generate_embeddings(abst, tokenizer, model, device, classification_model=True)
        else:
            _, mat[i], _ = generate_embeddings(abst, tokenizer, model, device)
        last_iter = np.array([i])
        np.save('./data/variables/last_iter_batch_1', last_iter)

    # save embedding
    np.save(save_embeddings_to, mat)

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time for generating embeddings: {elapsed_time:.2f} seconds")

if __name__ == '__main__':
    # "dmis-lab/biobert-v1.1"
    # "dmis-lab/biobert-v1.1"
    # "allenai/scibert_scivocab_uncased"
    # "bert-base-uncased"
    # "dmis-lab/biobert-v1.1"
    # "allenai/scibert_scivocab_uncased"
    # "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    # "dmis-lab/biobert-base-cased-v1.2"
    # "bert-base-uncased"
    # "allenai/scibert_scivocab_uncased"
    # "bert-base-uncased"
    
    df = pd.read_csv('data/input/ddg2p_db.csv')
    df = df.drop_duplicates(subset='pmid', keep='first')
    
    column_to_embed = "abstract"
    
    df[column_to_embed] = df[column_to_embed].fillna(df["title"])
    abstracts = df[column_to_embed].tolist()
    print("Number of abstracts: ", len(abstracts))
    
    checkpoint = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    model_name = checkpoint.replace("/","_")
    save_embeddings_to = f'data/embeddings/embeddings_{model_name}_{len(abstracts)}'
    embed_via_bert(abstracts, checkpoint, save_embeddings_to)



