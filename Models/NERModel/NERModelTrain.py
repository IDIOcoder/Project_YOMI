import wget
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ElectraTokenizerFast
from transformers import ElectraForTokenClassification, AdamW
from tqdm import tqdm

# download dataset
train_data = wget.download('https://raw.githubusercontent.com/IDIOcoder/Chat-bot/main/dataset/ner_dataset.txt')

# configure tag, download tokenizer
tokenizer = ElectraTokenizerFast.from_pretrained('monologg/koelectra-base-v3-discriminator')
tag2idx = {"O": 0, "B-FOOD": 1, "I-FOOD": 2}
idx2tag = {v: k for k, v in tag2idx.items()}

# define Dataset Class
class NERDataset(Dataset):
    def __init__(self, file_path):
        self.sentences = []
        self.labels = []

        with open(file_path, "r", encoding='utf-8') as f:
            sentence = []
            label = []
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) == 2:
                        word, tag = parts
                        sentence.append(word)
                        label.append(tag)
                    else:
                        print(f"Warning: Line with incorrect format found: {line.strip()}")
                else:
                    if sentence:
                        self.sentences.append(sentence)
                        self.labels.append(label)
                        sentence = []
                        label = []

        if sentence:
            self.sentences.append(sentence)
            self.labels.append(label)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        words = self.sentences[idx]
        labels = self.labels[idx]

        encoding = tokenizer(words, is_split_into_words=True, return_offsets_mapping=True, padding='max_length', truncation=True, max_length=128)
        labels = [tag2idx[label] for label in labels]

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        token_type_ids = encoding['token_type_ids']
        offsets = encoding['offset_mapping']

        # Create a new list of labels for each sub-token
        new_labels = []
        current_word_idx = -1
        for offset, input_id in zip(offsets, input_ids):
            if input_id == tokenizer.cls_token_id or input_id == tokenizer.sep_token_id:
                new_labels.append(tag2idx["O"]) # CLS와 SEP 토큰에는 O로 라벨링
            elif input_id == tokenizer.pad_token_id:
                new_labels.append(tag2idx["O"]) # PAD 토큰에 대해 라벨링
            else:
                if offset[0] == 0:
                    current_word_idx += 1
                new_labels.append(labels[current_word_idx])

        # Padding labels to max_length
        max_length = 128
        padded_labels = new_labels[:max_length]
        padded_labels = padded_labels + [tag2idx["O"]] * (max_length - len(padded_labels))

        item = {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'token_type_ids': torch.tensor(token_type_ids),
            'labels': torch.tensor(padded_labels)
        }

        return item

# define collate_func
def collate_fn(batch):
    max_len = max([len(item['input_ids']) for item in batch])

    input_ids = torch.stack([torch.cat([item['input_ids'], torch.zeros(max_len - len(item['input_ids']))]) for item in batch])
    attention_mask = torch.stack([torch.cat([item['attention_mask'], torch.zeros(max_len - len(item['attention_mask']))]) for item in batch])
    token_type_ids = torch.stack([torch.cat([item['token_type_ids'], torch.zeros(max_len - len(item['token_type_ids']))]) for item in batch])
    labels = torch.stack([torch.cat([item['labels'], torch.zeros(max_len - len(item['labels']))]) for item in batch])

    return {
        'input_ids': input_ids.long(),
        'attention_mask': attention_mask.long(),
        'token_type_ids': token_type_ids.long(),
        'labels': labels.long()
    }

# 데이터 로드 및 DataLoader 생성
train_dataset = NERDataset(train_data)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

# define model, set optimizer
model = ElectraForTokenClassification.from_pretrained('monologg/koelectra-base-v3-discriminator', num_labels=len(tag2idx))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

# Train
model.train()
for epoch in range(5):  # Epoch 수를 조정하세요
    total_loss = 0
    for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_dataloader)}")

def predict(model, sentence):
    model.eval()
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)

    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).squeeze().tolist()
        print(predictions)

    tokenized_input = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze().tolist())
    print(tokenized_input)

    result = []
    for token, pred in zip(tokenized_input, predictions):
        if token.startswith('##'):
            result[-1][0] += token[2:]
        else:
            result.append([token, idx2tag[pred]])

    return result


# test
while True:
    sentence = input("Enter a sentence: ")
    if sentence == 'quit' or sentence == 'exit':
        break
    result = predict(model, sentence)

    # 결과 출력
    for word, tag in result:
        print(f"{word}: {tag}")

if input("save model? (y/n):")=="y":
    torch.save(model, 'NERModel.pth')