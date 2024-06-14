import torch
from transformers import PreTrainedTokenizerFast

Q_TKN = "<usr>"
A_TKN = "<sys>"
BOS = "</s>"
EOS = "</s>"
MASK = "<unused0>"
SENT = "<unused1>"
PAD = "<pad>"
Sneg = -1e18
max_len = 121  # 모델 학습시 확인하고 입력할 것.

TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                                 bos_token=BOS,
                                                                 eos_token=EOS,
                                                                 unk_token='<unk>',
                                                                 pad_token=PAD,
                                                                 mask_token=MASK)


# 모델의 창의적 답변을 위한 필터링 함수 정의
# logins 를 필터링 하여 top-k 및 top-p 샘플링을 적용
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
            return logits
        logits[indices_to_remove] = filter_value
    return logits
