# 라이브러리 불러오기
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core import LightningModule
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import re, os
from tqdm import tqdm


print("[Step 1] Set Hyper Parameters")
Q_TKN = "<usr>"
A_TKN = "<sys>"
BOS = "</s>"
EOS = "</s>"
MASK = "<unused0>"
SENT = "<unused1>"
PAD = "<pad>"
EPOCHS = 15
Sneg = -1e18
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 3e-5

print("[Step 2] Get tokenizer & model & dataset")
koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2",
    bos_token=BOS,
    eos_token=EOS,
    unk_token='<unk>',
    pad_token=PAD,
    mask_token=MASK
)


model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")

data_path = '../../dataset/answer_dataset.csv'
df = pd.read_csv(data_path, encoding='utf-8')
df.dropna(subset=["Q"], inplace=True)
df.dropna(subset=["A"], inplace=True)


# CheckPoint
print(df.head())
if input("[Check Point] Proceed? (y/n): ") != "y":
    exit()


print("[Step 3] Define find_max_len func & Dataset Class & collate_batch func & dataLoader")
def find_max_len(dataset, tokenizer):
    max_len = 0
    for idx, row in dataset.iterrows():
        q = row['Q'].strip()
        a = row['A'].strip()

        # 질문과 답변을 토큰화
        q_toked = tokenizer.tokenize(q)
        a_toked = tokenizer.tokenize(a)

        # 토큰화된 질문과 답변의 길이 합산
        total_len = len(q_toked) + len(a_toked)

        # 최대 길이 갱신
        if total_len > max_len:
            max_len = total_len

    return max_len

max_len = find_max_len(df, koGPT2_TOKENIZER)
print(f"[DEBUG] Max token length in the dataset: {max_len}")


# 데이터셋 정의
class ChatbotDataset(Dataset):
    def __init__(self, chats, max_len=40):
        self._data = chats
        self.max_len = max_len
        self.q_token = Q_TKN
        self.a_token = A_TKN
        self.sent_token = SENT
        self.eos = EOS
        self.mask = MASK
        self.tokenizer = koGPT2_TOKENIZER

    def __len__(self):  # ChatbotDataset의 길이를 리턴.
        return len(self._data)

    def __getitem__(self, idx):  # 로드한 ChatbotDataset을 차례차례 DataLoader로 넘겨주는 메서드
        turn = self._data.iloc[idx]
        q = turn["Q"].strip()  # 질문을 가져온다.
        a = turn["A"].strip()  # 답변을 가져온다.

        q_toked = self.tokenizer.tokenize(self.q_token + q + self.sent_token)
        q_len = len(q_toked)

        a_toked = self.tokenizer.tokenize(self.a_token + a + self.eos)
        a_len = len(a_toked)

        # 질문의 길이가 최대길이보다 크면
        if q_len > self.max_len:
            q_toked = q_toked[-(self.max_len // 2):]  # 질문길이를 최대길이의 반으로
            q_len = len(q_toked)

        # 질문의 길이 + 답변의 길이가 최대길이보다 크면
        if q_len + a_len > self.max_len:
            a_len = self.max_len - q_len
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)

        # 답변 labels = [mask, mask, ..., mask, ..., <bos>, ...답변..., <eos>, <pad> ...]
        labels = [self.mask, ] * q_len + a_toked[1:]

        # mask = 질문길이 0 + 답변길이 1 + 나머지 0
        mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len)
        # 답변 labels를 index로 만든다.
        labels_ids = self.tokenizer.convert_tokens_to_ids(labels)
        labels_ids += [self.tokenizer.pad_token_id] * (self.max_len - len(labels_ids))

        # 질문 + 답변을 index로 만든다.
        token_ids = self.tokenizer.convert_tokens_to_ids(q_toked + a_toked)
        token_ids += [self.tokenizer.pad_token_id] * (self.max_len - len(token_ids))

        # attention mask 생성
        attention_mask = [1] * (q_len + a_len) + [0] * (self.max_len - q_len - a_len)

        # 질문 + 답변, 마스크, 답변, attention mask
        return (token_ids, np.array(mask), labels_ids, attention_mask)


def collate_batch(batch):
    data = [item[0] for item in batch]
    mask = [item[1] for item in batch]
    label = [item[2] for item in batch]
    attention_mask = [item[3] for item in batch]
    return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label), torch.LongTensor(attention_mask)


train_set = ChatbotDataset(df, max_len=max_len)
train_dataloader = DataLoader(
    train_set,
    batch_size=32,  # 배치 크기 줄이기
    num_workers=2,  # 윈도우->0 / 리눅스->2
    shuffle=True,
    collate_fn=collate_batch,
)

model.to(device)


print("[Step 4] Train Model")
criterion = torch.nn.CrossEntropyLoss(reduction="none")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

scaler = torch.cuda.amp.GradScaler()  # Mixed precision을 위한 GradScaler

for epoch in range(EPOCHS):
    dataloader = tqdm(train_dataloader, desc=f"Epoch {epoch}")
    model.train()
    for batch_idx, samples in enumerate(dataloader):
        # Gradient 0 으로 초기화
        optimizer.zero_grad()
        token_ids, mask, label, attention_mask = samples

        # 데이터 타입 확인 및 GPU로 이동
        token_ids, mask, label, attention_mask = token_ids.to(device), mask.to(device), label.to(device), attention_mask.to(device)

        with torch.cuda.amp.autocast():  # Mixed precision 사용
            out = model(token_ids, attention_mask=attention_mask)
            out = out.logits
            # 마스크 3D로 확장 및 출력 필터링
            mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
            mask_out = torch.where(mask_3d == 1, out, Sneg*torch.ones_like(out))

            # 라벨 값이 클래스 개수 범위 내에 있는지 확인
            assert label.max().item() < model.config.vocab_size, "라벨 값이 클래스 개수보다 큽니다."

            loss = criterion(mask_out.transpose(2, 1), label)
            avg_loss = loss.sum() / mask.sum()

        scaler.scale(avg_loss).backward()  # Mixed precision을 위한 gradient scaling
        scaler.step(optimizer)
        scaler.update()

        # GPU 메모리 정리
        del token_ids, mask, label, attention_mask, out, mask_3d, mask_out, loss, avg_loss
        torch.cuda.empty_cache()


print("[Step 5] Test Trained-Model")
model.eval()
# 답변 생성 함수
def sample_sequence(model, tokenizer, q, max_len=40, temperature=1.0, top_k=50, top_p=0.9):
    with torch.no_grad():
        a = ""
        for _ in range(max_len):
            input_ids = torch.LongTensor(
                tokenizer.encode(Q_TKN + q + SENT + A_TKN + a)
            ).unsqueeze(dim=0).to(device)
            pred = model(input_ids)
            logits = pred.logits[:, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            probabilities = torch.nn.functional.softmax(filtered_logits, dim=-1)
            gen_id = torch.multinomial(probabilities, num_samples=1).item()
            if gen_id >= tokenizer.vocab_size:
                print(f"Invalid token ID: {gen_id}")
                break
            gen = tokenizer.convert_ids_to_tokens([gen_id])[0]
            if gen == EOS:
                break
            a += gen.replace("▁", " ")
        return a.strip()

# logits를 필터링 하여 top-k 및 top-p 샘플링을 적용
# top_k : 확률이 높은 상위 k개의 토큰 중에서 샘플링.
# top_p : 누적 확률이 p 이하가 되는 상위 토큰들 중에서 샘플링
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    # top_k 필터링
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    # top_p 필터링
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        if indices_to_remove.shape != logits.shape:
                # print(f"indices_to_remove shape mismatch: {indices_to_remove.shape} vs {logits.shape}")
                return logits
        logits[indices_to_remove] = filter_value
    return logits

# 테스트
with torch.no_grad():
    while True:
        q = input("user > ").strip()
        if q == "quit" or q == "exit":
            break
        response = sample_sequence(model, koGPT2_TOKENIZER, q, max_len=max_len, temperature=0.5, top_k=20, top_p=0.9) # temperature : 샘플링 온도로, 낮은 값은 덜 무작위 적이고, 높은 값은 더 무작위적인 결과를 만듬.
        print(f"Chatbot > {response}")

if input("Save Model? (y/n): ") == "y":
    torch.save(model.state_dict(), '../../weights/text_weights.pth')

if input("Save Tokenizer? (y/n): ") == "y":
    koGPT2_TOKENIZER.save_pretrained('./saved_tokenizer')