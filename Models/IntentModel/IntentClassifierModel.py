import torch
from torch import nn
import utils.logger_utils as log

class IntentClassifierModel(nn.Module):
  def __init__(self,
               bert,
               hidden_size=768,
               num_classes=3,
               dr_rate=None,
               params=None):
    super(IntentClassifierModel, self).__init__()

    # 로그
    logger = log.get_logger('default')
    logger.info('__init__')

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