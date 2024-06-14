import torch
from transformers import BertModel
import gluonnlp as nlp

MAX_LEN = 64
BATCH_SIZE = 32


def get_kobert_model(model_path, vocab_file, device):
    bert_model = BertModel.from_pretrained(model_path)
    bert_model.to(device)
    bert_model.eval()
    vocab_b_obj = nlp.vocab.BERTVocab.from_sentencepiece(vocab_file, padding_token='[PAD]')
    return bert_model, vocab_b_obj
