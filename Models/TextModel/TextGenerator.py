from utils.KoGPT2_utils import *
import torch
from transformers import GPT2LMHeadModel
import utils.logger_utils as log


class TextGeneratorPredict:
    def __init__(self, weight_path, device):
        # logger
        self.logger = log.get_logger('default')
        self.logger.info("Initializing Text-Generator (0/4)")

        self.device = device
        self.weight_path = weight_path
        self.logger.debug(f"Device = {self.device} (1/4)")

        self.logger.debug("Loading KoGPT2-Tokenizer (2/4)")
        self.tokenizer = TOKENIZER

        self.logger.debug("Loading KoGPT2-Model (3/4)")
        self.model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
        self.model.to(self.device)

        self.logger.debug("Applying weights (4/4)")
        self.model.load_state_dict(torch.load(self.weight_path, map_location=self.device))
        self.model.eval()

    def sample_sequence(self, q, max_len=40, temperature=1.0, top_k=50, top_p=0.9):
        with torch.no_grad():
            a = ""
            for _ in range(max_len):
                input_ids = torch.LongTensor(
                    self.tokenizer.encode(Q_TKN + q + SENT + A_TKN + a)
                ).unsqueeze(dim=0).to(self.device)
                pred = self.model(input_ids)
                logits = pred.logits[:, -1, :] / temperature
                filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                probabilities = torch.nn.functional.softmax(filtered_logits, dim=-1)
                gen_id = torch.multinomial(probabilities, num_samples=1).item()
                if gen_id >= self.tokenizer.vocab_size:
                    print(f"Invalid token ID: {gen_id}")
                    break
                gen = self.tokenizer.convert_ids_to_tokens([gen_id])[0]
                if gen == EOS:
                    break
                a += gen.replace("▁", " ")
            return a.strip()

    def generate_answer(self, usr_input):
        self.logger.debug("Generating answer ...")
        answer = self.sample_sequence(usr_input, max_len=max_len, temperature=0.5, top_k=20, top_p=0.9)
        return answer
