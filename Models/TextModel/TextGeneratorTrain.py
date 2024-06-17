import pandas as pd
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# 데이터셋 준비
data_url = "https://raw.githubusercontent.com/IDIOcoder/Chat-bot/main/dataset/answer_dataset.csv"
df = pd.read_csv(data_url, encoding='utf-8')
df.dropna(subset=["Q"], inplace=True)
df.dropna(subset=["A"], inplace=True)

# 질문과 대답을 리스트로 변환
questions = df['Q'].tolist()
answers = df['A'].tolist()
turns = df['T'].tolist()

# 데이터 전처리 함수
def preprocess_data(questions, answers, turns):
    data = []
    for q, a, turn in zip(questions, answers, turns):
        if turn.startswith("multi"):
            # 멀티턴 데이터의 경우
            data.append(f"{q} 대답: {a}")
        else:
            # 싱글턴 데이터의 경우
            data.append(f"질문: {q} 대답: {a}")
    return data

# 전처리된 데이터
data = preprocess_data(questions, answers, turns)

# 데이터셋 분할 (학습 0.8, 검증 0.2)
train_size = int(0.8 * len(data))
train_data = data[:train_size]
val_data = data[train_size:]

# 하이퍼 파라메터 설정
Q_TKN = "<usr>"
A_TKN = "<sys>"
BOS = "</s>"
EOS = "</s>"
MASK = "<unused0>"
SENT = "<unused1>"
PAD = "<pad>"

# KoGPT-2 토크나이저 및 모델 로드
print("loading tokenizer, model")
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                    bos_token=BOS,
                                                    eos_token=EOS,
                                                    unk_token='<unk>',
                                                    pad_token=PAD,
                                                    mask_token=MASK)
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

# 데이터셋 클래스 정의
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.input_ids=[]
        self.attention_masks = []

        for text in texts:
            encoded_dict = tokenizer(
                text,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            self.input_ids.append(encoded_dict['input_ids'])
            self.attention_masks.append(encoded_dict['attention_mask'])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx].squeeze(),
            'attention_mask': self.attention_masks[idx].squeeze()
        }

# 데이터셋 및 데이터로더 생성
print("generate dataset")
train_dataset = TextDataset(train_data, tokenizer)
val_dataset = TextDataset(val_data, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# 장치설정 및 모델 옮기기
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
scaler = GradScaler()
epochs = 5

# 학습 루프
for epoch in range(epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}')

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        model.zero_grad()

        with autocast():
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=total_loss/len(train_dataloader))

    print(f"Epoch {epoch+1} : Avg_loss={total_loss/len(train_dataloader)}")

# 모델 평가 모드로 설정
model.eval()

# 대화 생성 함수
def generate_response(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=128, num_return_sequences=1, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

# 대화 테스트
while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = generate_response(f"질문: {user_input} 대답:")
    print(f"Bot: {response}")

if input("save model? (y/n): ") == 'y':
    torch.save(model, 'textGeneratorModel.pth')