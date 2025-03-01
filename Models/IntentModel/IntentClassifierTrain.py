import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
np.bool = np.bool_
import gluonnlp as nlp
from gluonnlp.data import SentencepieceTokenizer
from tqdm import tqdm
import pandas as pd
from utils.KoBERT_utils import get_kobert_model
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import BertModel
from kobert_tokenizer import KoBERTTokenizer
from collections import Counter
from sklearn.model_selection import train_test_split

import IntentClassifierDataset as ID

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("[GET] tokenizer, bertmodel, vocab")
tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")
bertmodel, vocab = get_kobert_model('skt/kobert-base-v1', tokenizer.vocab_file, device)
print("Done")

# get dataset
print("[GET] dataset")
# datasetURL = "https://raw.githubusercontent.com/IDIOcoder/Chat-bot/main/dataset/intent_dataset.csv"
data_path = '../../dataset/intent_dataset.csv'
data = pd.read_csv(data_path, encoding='utf-8')
print("Done")

data_list = []
for ques, label in zip(data['sentence'], data['intent']):
  data = []
  data.append(ques)
  data.append(str(label))

  data_list.append(data)

# 데이터셋의 라벨을 모두 모아 클래스 빈도 계산
labels = [int(label) for _, label in data_list]
class_counts = Counter(labels)
num_samples = len(labels)

# 클래스 가중치 계산 (클래스 빈도의 역수)
class_weights = {cls: num_samples / count for cls, count in class_counts.items()}
class_weights_tensor = torch.tensor([class_weights[i] for i in sorted(class_weights.keys())], dtype=torch.float).to(device)

#Setting Parameters
max_len = 64
batch_size = 32
warmup_ratio = 0.1
num_epochs = 10
max_grad_norm = 1
log_interval = 200
learning_rate = 5e-5

#train & test 데이터로 나누기
print("[INFO] split train, test dataset")
dataset_train, dataset_test = train_test_split(data_list, test_size=0.2, shuffle=True, random_state=34)

tok=tokenizer.tokenize
data_train = ID.IntentClassifierDataset(dataset_train, 0, 1, tok, vocab, max_len, True, False)
data_test = ID.IntentClassifierDataset(dataset_test,0, 1, tok, vocab,  max_len, True, False)

train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=2)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=2)

#BERT 모델 불러오기
class IntentClassifierModel(nn.Module):
  def __init__(self,
               bert,
               hidden_size=768,
               num_classes=3,
               dr_rate=None,
               params=None):
    super(IntentClassifierModel, self).__init__()

    self.bert = bert
    self.dr_rate = dr_rate

    self.classifier = nn.Linear(hidden_size, num_classes)
    if dr_rate:
      self.dropout = nn.Dropout(p=dr_rate)

  def gen_attention_mask(self, token_ids, valid_length):
    attention_mask = torch.zeros_like(token_ids)
    for i, v in enumerate(valid_length):
      attention_mask[i][:v] = 1
    return attention_mask.float()

  def forward(self, token_ids, valid_length, segment_ids):
    attention_mask = self.gen_attention_mask(token_ids, valid_length)

    _, pooler = self.bert(input_ids=token_ids,
                          token_type_ids=segment_ids.long(),
                          attention_mask=attention_mask.float().to(token_ids.device),
                          return_dict=False)
    if self.dr_rate:
      out = self.dropout(pooler)
    return self.classifier(out)


model = IntentClassifierModel(bertmodel, dr_rate=0.5).to(device)


# optimizer와 schedule설정
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params':[p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params':[p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor) #다중분류를 위한 대표적인 loss func

t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

#정확도 측정을 위한 함수 정의
def calc_accuracy(X, Y):
  max_vals, max_indices = torch.max(X, 1)
  train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
  return train_acc

train_dataloader


##### Training #####
print("[INFO] training")
train_history=[]
test_history=[]
loss_history=[]
for e in range(num_epochs):
    train_acc = 0.0
    test_acc = 0.0
    model.train()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length = valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)

        #print(label.shape,out.shape)
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        train_acc += calc_accuracy(out, label)
        if batch_id % log_interval == 0:
            print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
            train_history.append(train_acc / (batch_id+1))
            loss_history.append(loss.data.cpu().numpy())
    print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
    #train_history.append(train_acc / (batch_id+1))

    model.eval()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        test_acc += calc_accuracy(out, label)
    print("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))
    test_history.append(test_acc / (batch_id+1))



##### Test #####
def predict(predict_sentence):

    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = ID.IntentClassifierDataset(dataset_another, 0, 1, tok, vocab, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)

    model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length= valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)


        test_eval=[]
        for i in out:
            logits=i
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0:
                test_eval.append("대화의")
            elif np.argmax(logits) == 1:
                test_eval.append("레시피 요청의")
            elif np.argmax(logits) == 2:
                test_eval.append("긍정의")
            elif np.argmax(logits) == 3:
                test_eval.append("부정의")

        print(">> 입력하신 내용에서 " + test_eval[0] + " 의도가 느껴집니다.")

end = 1
while end == 1 :
    sentence = input("하고싶은 말을 입력해주세요 : ")
    if sentence == "0" :
        break
    predict(sentence)
    print("\n")

##### 모델 다운로드 #####
if input("save model? (y/n): ") == "y":
    torch.save(model.state_dict(), '../../weights/intent_weights.pth')

if input("save tokenizer? (y/n): ") == "y":
    tokenizer.save_pretrained('./saved_tokenizer')
