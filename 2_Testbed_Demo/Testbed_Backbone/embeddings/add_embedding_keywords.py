'''
This function is to add keywords to the embedding model, so as to facilitate the embedding of keywords in the embedding model
This function is implemented by modifying the tokenizer of the embedding model
This function is only effective for the model corresponding to the EMBEDDING_MODEL parameter, and the output model is stored in the original model
Thanks to @CharlesJu1 and @charlesyju for their contributions in coming up with ideas and the most basic PR

The saved model is located in the same directory as the original embedded model with the name of the original model +Merge_Keywords_ timestamp
'''
import sys
sys.path.append("..")
from datetime import datetime
from configs import (
    MODEL_PATH,
    EMBEDDING_MODEL,
    EMBEDDING_KEYWORD_FILE,
)
import os
import torch
from safetensors.torch import save_model
from sentence_transformers import SentenceTransformer


def get_keyword_embedding(bert_model, tokenizer, key_words):
    tokenizer_output = tokenizer(key_words, return_tensors="pt", padding=True, truncation=True)

    # No need to manually convert to tensor as we've set return_tensors="pt"
    input_ids = tokenizer_output['input_ids']

    # Remove the first and last token for each sequence in the batch
    input_ids = input_ids[:, 1:-1]

    keyword_embedding = bert_model.embeddings.word_embeddings(input_ids)
    keyword_embedding = torch.mean(keyword_embedding, 1)

    return keyword_embedding


def add_keyword_to_model(model_name=EMBEDDING_MODEL, keyword_file: str = "", output_model_path: str = None):
    key_words = []
    with open(keyword_file, "r") as f:
        for line in f:
            key_words.append(line.strip())

    st_model = SentenceTransformer(model_name)
    key_words_len = len(key_words)
    word_embedding_model = st_model._first_module()
    bert_model = word_embedding_model.auto_model
    tokenizer = word_embedding_model.tokenizer
    key_words_embedding = get_keyword_embedding(bert_model, tokenizer, key_words)
    # key_words_embedding = st_model.encode(key_words)

    embedding_weight = bert_model.embeddings.word_embeddings.weight
    embedding_weight_len = len(embedding_weight)
    tokenizer.add_tokens(key_words)
    bert_model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=32)

    # key_words_embedding_tensor = torch.from_numpy(key_words_embedding)
    embedding_weight = bert_model.embeddings.word_embeddings.weight
    with torch.no_grad():
        embedding_weight[embedding_weight_len:embedding_weight_len + key_words_len, :] = key_words_embedding

    if output_model_path:
        os.makedirs(output_model_path, exist_ok=True)
        word_embedding_model.save(output_model_path)
        safetensors_file = os.path.join(output_model_path, "model.safetensors")
        metadata = {'format': 'pt'}
        save_model(bert_model, safetensors_file, metadata)
        print("save model to {}".format(output_model_path))


def add_keyword_to_embedding_model(path: str = EMBEDDING_KEYWORD_FILE):
    keyword_file = os.path.join(path)
    model_name = MODEL_PATH["embed_model"][EMBEDDING_MODEL]
    model_parent_directory = os.path.dirname(model_name)
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_model_name = "{}_Merge_Keywords_{}".format(EMBEDDING_MODEL, current_time)
    output_model_path = os.path.join(model_parent_directory, output_model_name)
    add_keyword_to_model(model_name, keyword_file, output_model_path)


if __name__ == '__main__':
    add_keyword_to_embedding_model(EMBEDDING_KEYWORD_FILE)

    # input_model_name = ""
    # output_model_path = ""
    # # Below is a comparison of tokenizer test cases before and after adding keywords
    # def print_token_ids(output, tokenizer, sentences):
    #     for idx, ids in enumerate(output['input_ids']):
    #         print(f'sentence={sentences[idx]}')
    #         print(f'ids={ids}')
    #         for id in ids:
    #             decoded_id = tokenizer.decode(id)
    #             print(f'    {decoded_id}->{id}')
    #
    # sentences = [
    #     'Data science and Big Data technology',
    #     'Langchain-Chatchat'
    # ]
    #
    # st_no_keywords = SentenceTransformer(input_model_name)
    # tokenizer_without_keywords = st_no_keywords.tokenizer
    # print("===== tokenizer with no keywords added =====")
    # output = tokenizer_without_keywords(sentences)
    # print_token_ids(output, tokenizer_without_keywords, sentences)
    # print(f'-------- embedding with no keywords added -----')
    # embeddings = st_no_keywords.encode(sentences)
    # print(embeddings)
    #
    # print("--------------------------------------------")
    # print("--------------------------------------------")
    # print("--------------------------------------------")
    #
    # st_with_keywords = SentenceTransformer(output_model_path)
    # tokenizer_with_keywords = st_with_keywords.tokenizer
    # print("===== tokenizer with keyword added =====")
    # output = tokenizer_with_keywords(sentences)
    # print_token_ids(output, tokenizer_with_keywords, sentences)
    #
    # print(f'-------- embedding with keywords added -----')
    # embeddings = st_with_keywords.encode(sentences)
    # print(embeddings)