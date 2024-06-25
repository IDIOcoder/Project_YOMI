from torch.utils.data import DataLoader
from Models.EmotionModel.EmotionClassifierModel import EmotionClassifierModel
from Models.EmotionModel.EmotionClassifierDataset import EmotionClassifierDataset
import numpy as np
np.bool = np.bool_
from kobert_tokenizer import KoBERTTokenizer
from utils.KoBERT_utils import *
import utils.logger_utils as log


class EmotionClassifier:
    def __init__(self, weight_path, device):
        # logger
        self.logger = log.get_logger('default')
        self.logger.info('Initializing EmotionClassifier ...')

        self.device = device
        self.weight_path = weight_path
        self.logger.debug(f'Device: {self.device} (1/4)')

        self.logger.debug('Loading tokenizer (2/4)')
        self.tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1', last_hidden_state=True)

        self.logger.debug("Loading KoBERT-Model (3/4)")
        self.bertmodel, self.vocab = get_kobert_model('skt/kobert-base-v1', self.tokenizer.vocab_file, self.device)
        self.model = EmotionClassifierModel(self.bertmodel, dr_rate=0.5).to(self.device)

        self.logger.debug("Applying weights (4/4)")
        self.model.load_state_dict(torch.load(self.weight_path, map_location=self.device))
        self.model.eval()

    def predict(self, predict_sentence):
        self.logger.debug('Predicting Emotion ...')
        data = [predict_sentence, '0']
        dataset_another = [data]

        another_test = EmotionClassifierDataset(dataset_another,
                                                sent_idx=0,
                                                label_idx=1,
                                                bert_tokenizer=self.tokenizer.tokenize,
                                                vocab=self.vocab,
                                                max_len=64,
                                                pad=True,
                                                pair=False)
        test_dataloader = DataLoader(another_test, batch_size=1, num_workers=0)

        # model.eval()

        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
            token_ids = token_ids.long().to(self.device)
            segment_ids = segment_ids.long().to(self.device)
            valid_length = valid_length
            label = label.long().to(self.device)

            with torch.no_grad():
                out = self.model(token_ids, valid_length, segment_ids)

            # test_eval = []
            emotion = None
            for i in out:
                logits = i
                logits = logits.detach().cpu().numpy()
                emotion = np.argmax(logits)

                # if emotion == 0:    # 행복
                #     test_eval.append("행복")
                # elif emotion == 1:
                #     test_eval.append("중립")
                # elif emotion == 2:
                #     test_eval.append("슬픔")
                # elif emotion == 3:
                #     test_eval.append("분노")
                # elif emotion == 4:
                #     test_eval.append("불안")
                # elif emotion == 5:
                #     test_eval.append("놀람")
                # elif emotion == 6:
                #     test_eval.append("피곤")
                # elif emotion == 7:
                #     test_eval.append("후회")

            return emotion
