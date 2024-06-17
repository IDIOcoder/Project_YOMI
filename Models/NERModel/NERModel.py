import torch
from transformers import ElectraTokenizerFast
from transformers import ElectraForTokenClassification
import utils.logger_utils as log


class NERModel:
    def __init__(self, weight_path, device):
        self.logger = log.get_logger('default')
        self.logger.info("Initializing NER model ... ")

        self.tag2idx = {"O": 0, "B-FOOD": 1, "I-FOOD": 2}
        self.idx2tag = {v: k for k, v in self.tag2idx.items()}
        self.device = device
        self.weight_path = weight_path
        self.logger.debug(f"Device = {self.device} (1/4)")

        self.logger.debug("Loading KoELECTRA-Tokenizer (2/4)")
        self.tokenizer = ElectraTokenizerFast.from_pretrained('monologg/koelectra-base-v3-discriminator')

        self.logger.debug("Loading KoELECTRA-Model (3/4)")
        self.model = ElectraForTokenClassification.from_pretrained('monologg/koelectra-base-v3-discriminator', num_labels=len(self.tag2idx))
        self.model.to(self.device)

        self.logger.debug("Applying weights> (4/4)")
        self.model.load_state_dict(torch.load(self.weight_path, map_location=self.device))

    def predict(self, sentence):
        self.logger.debug("Predicting Word...")
        self.model.eval()
        inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)

        with torch.no_grad():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).squeeze().tolist()

        tokenized_input = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze().tolist())

        result = []
        for token, pred in zip(tokenized_input, predictions):
            if token.startswith('##'):
                result[-1][0] += token[2:]
            else:
                result.append([token, self.idx2tag[pred]])

        result_word = ""
        for word, token in result:
            if token == "B-FOOD" or token == "I-FOOD":
                result_word += word

        return result_word
