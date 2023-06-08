import pandas as pd
import numpy as np

import joblib
import torch

from sklearn import preprocessing
from sklearn import model_selection

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import config
import dataset
from model import EntityModel


def process_data():
    df = pd.read_csv('ner_dataset.csv', encoding="latin-1")
    df.loc[:, "Sentence #"] = df["Sentence #"].fillna(method="ffill")

    enc_pos = preprocessing.LabelEncoder()
    enc_tag = preprocessing.LabelEncoder()

    df.loc[:, "POS"] = enc_pos.fit_transform(df["POS"])
    df.loc[:, "Tag"] = enc_tag.fit_transform(df["Tag"])

    sentences = df.groupby("Sentence #")["Word"].apply(list).values
    pos = df.groupby("Sentence #")["POS"].apply(list).values
    tag = df.groupby("Sentence #")["Tag"].apply(list).values
    return sentences, pos, tag, enc_pos, enc_tag


def find_named_entities(text):
    meta_data = joblib.load("meta.bin")
    enc_pos = meta_data["enc_pos"]
    enc_tag = meta_data["enc_tag"]

    num_pos = len(list(enc_pos.classes_))
    num_tag = len(list(enc_tag.classes_))

    text = text.split('.')
    result = dict()
    for i in range(len(text)):
        dict_of_words = {}
        for word in text[i].split():
            li = config.TOKENIZER.encode(word)
            for ww in li[1:len(li) - 1]:
                dict_of_words[ww] = word
        dict_of_words[101] = 'beg'
        dict_of_words[102] = 'end'
        tokenized_sentence = config.TOKENIZER.encode(text[i])
        text[i] = text[i].split()

        test_dataset = dataset.EntityDataset(
            texts=[text[i]], 
            pos=[[0] * len(text[i])], 
            tags=[[0] * len(text[i])]
        )

        device = torch.device("cpu")
        model = EntityModel(num_tag=num_tag, num_pos=num_pos)
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=torch.device('cpu')))
        model.to(device)
        with torch.no_grad():
            data = test_dataset[0]
            for k, v in data.items():
                data[k] = v.to(device).unsqueeze(0)
            tag, pos, _ = model(**data)

            li = []
            for j in range(len(tokenized_sentence)):
                result[dict_of_words[tokenized_sentence[j]]] = enc_tag.inverse_transform(
                        tag.argmax(2).cpu().numpy().reshape(-1)
                    )[j:j+1][0]
                
    return result


def evaluate1(sentence):

    tokenized_sentence = config.TOKENIZER.encode(sentence)
    tokens = config.TOKENIZER.convert_ids_to_tokens(tokenized_sentence)
    sentence = sentence.split()
    
    test_dataset = dataset.EntityDataset(
        texts=[sentence], 
        pos=[[0] * len(sentence)], 
        tags=[[0] * len(sentence)]
    )

    with torch.no_grad():
        data = test_dataset[0]
        for k, v in data.items():
            data[k] = v.to(device).unsqueeze(0)
        tag, pos, _ = model(**data)

        return tokens, enc_tag.inverse_transform(
                tag.argmax(2).cpu().numpy().reshape(-1)
            )[:len(tokenized_sentence)]

def merge_subwords(token_list, tag_list):
    merged_tokens = ['^']
    merged_tags = []

    for token, tag in zip(token_list, tag_list):
        if merged_tokens[-1] == "'":
            merged_tokens[-1] = merged_tokens[-1] + token
        else:
            if token not in ['[CLS]', '[SEP]'] and (token.startswith("##") is False):
                merged_tokens.append(token)
                merged_tags.append(tag)
            elif token.startswith("##"):
                merged_tokens[-1] = merged_tokens[-1] + token[2:]

    return merged_tokens[1::], merged_tags

def tokenize(sentence, correct_tags):
    splitted_sentence = sentence.split('=')
    padded_tags = []
    for i in range(len(splitted_sentence)):
        tokenized_word = config.TOKENIZER.encode(splitted_sentence[i])
        if len(tokenized_word) > 3:
            padded_tags.extend([correct_tags[i]] * (len(tokenized_word) - 2))
        else:
            padded_tags.append(correct_tags[i])
    return padded_tags