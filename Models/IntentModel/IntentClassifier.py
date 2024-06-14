from torch.utils.data import DataLoader
from Models.IntentModel.IntentClassifierModel import IntentClassifierModel
from Models.IntentModel.IntentClassifierDataset import IntentClassifierDataset
import numpy as np
np.bool = np.bool_
from kobert_tokenizer import KoBERTTokenizer
from utils.KoBERT_utils import *
import utils.logger_utils as log


class IntentClassifierPredict:
    def __init__(self, weight_path, device):
        # 로그
        self.logger = log.get_logger('default')
        self.logger.info('Initializing Intent-Classifier ...')

        self.device = device
        self.weight_path = weight_path
        self.logger.debug(f'Device: {self.device} (1/4)')

        self.logger.debug('Loading KoBERT-Tokenizer (2/4)')
        self.tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1', last_hidden_state=True)

        self.logger.debug("Loading KoBERT-Model (3/4)")
        self.bertmodel, self.vocab = get_kobert_model('skt/kobert-base-v1', self.tokenizer.vocab_file, self.device)
        self.model = IntentClassifierModel(self.bertmodel, dr_rate=0.5).to(self.device)

        self.logger.debug('Applying weights (4/4)')
        self.model.load_state_dict(torch.load(self.weight_path, map_location=self.device))
        self.model.eval()

    def predict(self, usr_input):
        self.logger.debug('Predicting Intent ...')
        data = [usr_input, '0']
        dataset_another = [data]

        another_test = IntentClassifierDataset(dataset_another,
                                               sent_idx=0,
                                               label_idx=1,
                                               bert_tokenizer=self.tokenizer.tokenize,
                                               vocab=self.vocab,
                                               max_len=MAX_LEN,
                                               pad=True,
                                               pair=False)
        test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=BATCH_SIZE, num_workers=5)

        self.model.eval()

        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
            token_ids = token_ids.long().to(self.device)
            segment_ids = segment_ids.long().to(self.device)

            valid_length = valid_length
            label = label.long().to(self.device)

            out = self.model(token_ids, valid_length, segment_ids)

            test_eval = []
            intent = None
            for i in out:
                logits = i
                logits = logits.detach().cpu().numpy()
                intent = np.argmax(logits)

                if intent == 0:
                    test_eval.append("대화")
                elif intent == 1:
                    test_eval.append("레시피 요청")

            return intent
