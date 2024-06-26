##########################################
# 전체모델 Yomi의 동작 파일입니다.             #
# Local환경과 Server환경 모두 동작하도록 작성.  #
##########################################
import markdown
import torch
import os
import dotenv
import json
import utils.logger_utils as log
import utils.recipe_utils as DB
from Models.IntentModel import IntentClassifier as IM
from Models.EmotionModel import EmotionClassifier as EM
from Models.TextModel import TextGenerator as TM
from Models.NERModel import NERModel as NM
from fuzzywuzzy import process

dotenv.load_dotenv()

# 병렬 처리 경고가 출력되지 않도록 합니다.
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# INFO OF user.json
current_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(current_dir, 'user')
default_content = {
    "state": 0,
    "last_input": "",     # 음식이름 추출이 실패했던 문장
    "last_response": "",
    "emotion": [0, 0, 0, 0, 0, 0, 0, 0, 0]     # 마지막 인덱스의 값은 마지막으로 추론된 감정의 인덱스
}


# 유저에 대한 정보를 전달하는 함수.
# 유저파일이 없다면 생성.
def check_usr_state(usr_id):
    if not os.path.exists(folder_path):
        log.DEFAULT.warning("User folder not exists. Creating new folder...")
        os.makedirs(folder_path)
    file_name = f'{usr_id}.json'
    json_path = os.path.join(folder_path, file_name)
    flag = 0
    if not os.path.exists(json_path):
        log.DEFAULT.debug("Create New User's file")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(default_content, f, ensure_ascii=False, indent=4)
    else:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        flag = data['state']
    return flag


# NER에서 음식이름을 특정하지 못한 경우 상태 없데이트를 위한 함수.
def update_usr_state(usr_id, *usr_input):
    json_path = os.path.join(folder_path, f'{usr_id}.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if data['state'] == 0:
        data['state'] = 1
        if usr_input:
            save_last_input(usr_id, usr_input)
    else:
        data['state'] = 0
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def save_last_input(usr_id, usr_input):
    json_path = os.path.join(folder_path, f'{usr_id}.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data['last_input'] = usr_input
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def get_last_input(usr_id):
    json_path = os.path.join(folder_path, f'{usr_id}.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['last_input']


def save_last_response(usr_id, bot_response):
    json_path = os.path.join(folder_path, f'{usr_id}.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data['last_response'] = bot_response
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def get_last_response(usr_id):
    json_path = os.path.join(folder_path, f'{usr_id}.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['last_response']


# 유저 정보 파일에서 대화에서 추론된 감정을 업데이트 합니다.
def update_emotion(usr_id, emotion):
    emotion = int(emotion)
    json_path = os.path.join(folder_path, f'{usr_id}.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data['emotion'][emotion] += 1
    data['emotion'][8] = emotion
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


class ChatBot:
    def __init__(self):
        # 장치 설정 & 모델 초기화
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.intent_model = IM.IntentClassifierPredict(os.getenv("INTENT"),self.device)
        self.emotion_model = EM.EmotionClassifier(os.getenv("EMOTION"), self.device)
        self.text_model = TM.TextGeneratorPredict(os.getenv("TEXT"), self.device)
        self.ner_model = NM.NERModel(os.getenv("NER"), self.device)

        # 레시피 DB에 저장된 요리 이름들을 불러옵니다.
        self.dish_names = DB.get_dish_names()

    # 최종 답변을 생성하는 메인 함수입니다.
    def get_response(self, usr_input, usr_id):
        try:
            flag = check_usr_state(usr_id)
            response = ""
            if flag == 0:   # 기본 동작
                intent = self.intent_model.predict(usr_input)
                if intent == 0:     # 대화
                    response = self.answer_func(usr_input, usr_id)
                elif intent == 1:       # 레시피 요청
                    response = self.recipe_func(usr_input, usr_id)
                elif intent == 2:   # 이전 대화에서 레시피가 필요하냐는 질문에 긍정한 경우
                    response = self.recipe_from_last_context(usr_id)
            elif flag == 1:     # 레시피 요청에 실패 했었을 경우
                response = self.re_search_func(usr_input, usr_id)
            save_last_input(usr_id, usr_input)
            return {"message": markdown.markdown(response)}
        except Exception as e:
            log.DEFAULT.error(e)
            response = "제 AI에 문제가 생긴 것 같습니다. 개발자에게 문의해 주세요."
            save_last_input(usr_id, usr_input)
            save_last_response(usr_id, response)
            return {"message": response}

    # <대화>의 의도로 분류된 경우 답변을 생성하는 함수입니다.
    def answer_func(self, usr_input, usr_id):
        # 감정 처리
        emotion = self.emotion_model.predict(usr_input)
        if emotion != 1:
            update_emotion(usr_id, emotion)
        response = self.text_model.generate_answer(usr_input)
        save_last_response(usr_id, response)
        # 로깅
        log.CHATBOT.debug(
            f"- Queue: {usr_input}\n"
            f"- Answer: {response}\n"
            f"- Emotion: {emotion}\n"
            f"- Dominate: None")
        return response

    # <레시피 요청>의 의도로 분류된 경우 추출된 요리에 대한 레시피를 출력하는 함수입니다.
    # 요리 이름 특정에 성공한 경우, 특정된 요리이름과 유사도가 가장 높은 요리의 레시피를 반환합니다.
    # 요리 이름 특정에 실패한 경우, 사용자에게 음식이름을 재입력 할 것을 요청합니다.
    def recipe_func(self, usr_input, usr_id):
        dish_name = self.ner_model.predict(usr_input)

        if dish_name:
            if dish_name != "":
                match_name = process.extractOne(dish_name, self.dish_names)
                response = DB.get_recipe(match_name[0])
                save_last_response(usr_id, f'{match_name[0]}의 레시피')

                # 로깅 후 리턴
                log.CHATBOT.debug(f"- Queue: {usr_input}\n"
                                  f"- DISH: {match_name[0]}")
            elif dish_name == "":
                self.recipe_from_last_context(usr_id)
        else:
            update_usr_state(usr_id, usr_input)
            response = "요청하신 내용을 인지하지 못했어요. 레시피가 필요한 요리의 이름을 입력해주세요."
            save_last_response(usr_id, response)
        return response

    # NER모델을 통해 음식이름을 특정해내지 못한경우 작동하게 되는 함수입니다.
    # 음식 사전중 사용자의 입력과 유사도가 50% 이상인 경우 해당 요리의 레시피를 출력합니다.
    # 요리 이름을 재 입력 받아도 존재하지 않다면, 존재하지 않음을 알립니다.
    # 사용자가 새로운 입력을 한 경우 답변생성 시퀀스로 넘어갑니다.
    def re_search_func(self, usr_input, usr_id):
        failed_input = get_last_input(usr_id)
        match_name = process.extractOne(usr_input, self.dish_names)
        if match_name[1] >= 85:
            # 로깅 후 리턴
            log.CHATBOT.debug(f"- Queue(Failed): {failed_input}\n"
                              f"- DISH: {match_name[0]}")
            result = DB.get_recipe(match_name[0])
            save_last_response(usr_id, f'{result}의 레시피')
        elif usr_input in failed_input:
            result = "제가 잘 모르는 요리인 것 같아요. 빠른 시일내로 업데이트 할게요!"
            save_last_response(usr_id, result)
        else:
            result = self.answer_func(usr_input)
        update_usr_state(usr_id)  # 기본 값으로 돌아갑니다.
        return result

    # 대화중 레시피 필요 여부 질문에서 긍정인 경우 실행되는 함수 입니다.
    # 먼저 챗봇의 이전 답변에서 음식 이름이 언급되었는지 확인하고
    # 언급되지 않았다면, 이전 사용자의 입력에서 음식 이름이 언급되었는지 확인합니다.
    # 음식 이름이 확인된다면, 해당 음식의 레시피를 제공합니다.
    def recipe_from_last_context(self, usr_id):
        dish_name = self.ner_model.predict(get_last_response(usr_id))
        if dish_name == "":
            dish_name = self.ner_model.predict(get_last_input(usr_id))
        match_name = process.extractOne(dish_name, self.dish_names)
        response = DB.get_recipe(match_name[0])
        save_last_response(usr_id, f'{match_name[0]}의 레시피')

        # 로깅 후 리턴
        log.CHATBOT.debug(f"{match_name[0]}의 레시피를 제공했습니다.")
        return response



