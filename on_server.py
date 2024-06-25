import json
import os
import time
import traceback
import markdown
# 버전 정보
from dotenv import load_dotenv
from flask import Flask, request, jsonify

import Yomi_ai
import utils.logger_utils as log
import utils.recipe_utils as DB

load_dotenv()
VERSION = os.getenv('ONLINE_VERSION')

"""""""""""""""""""""""""""""""""""""""""""""""""""
* < YOMI-AI >
* Work in Progress
* Run On Flask Server
! Do not forget to run Ngrok on your local machine
"""""""""""""""""""""""""""""""""""""""""""""""""""
# Flask 애플리케이션 생성
app = Flask(__name__)

today_recommand = DB.get_rand_dish()

@app.route('/', methods=['POST'])
def hello():
    text = (DB.get_recipe('히야앗코'))
    response = {"message": markdown.markdown(text)}
    time.sleep(5)
    return jsonify(response)

# chatting
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_id, user_input = data.get('user_id'), data.get('input')

    # 입력이 없는 경우
    if not user_input:
        log.DEFAULT.error('400 - No input provided')
        return jsonify({"error": "No input provided"}), 400

    # 입력이 있는 경우
    log.CHATBOT.debug(f'200 - User:{user_id} input provided')
    response = chatbot.get_response(user_input, user_id)
    return jsonify(response)


# provide recipe
@app.route('/recipe', methods=['POST'])
def return_recipe():
    data = request.get_json()
    recipe_name = data.get('name')

    recipe = DB.get_recipe(recipe_name)
    response = {"recipe": markdown.markdown(recipe)}
    return jsonify(response)


# reset user_emotion when login
@app.route('/login', methods=['POST'])
def reset_usr_file():
    data = request.get_json()
    user_id = data.get('user_id')

    with open(f'./user/{user_id}.json', 'w') as f:
        json.dump(Yomi_ai.default_content, f, ensure_ascii=False, indent=4)

    return jsonify({'status': 200})

# provide usr-emotion info
@app.route('/emotion', methods=['POST'])
def return_emotion():
    data = request.get_json()
    user_id = data.get('user_id')

    with open(f'./user/{user_id}.json', 'r') as f:
        e_data = json.load(f)
    user_emotion = e_data['emotion'][:7]
    user_emotion[e_data['emotion'][8]] += 3
    dominate = user_emotion.index(max(user_emotion)) if max(user_emotion) != 0 else 1
    emotion_name = ["행복", "중립", "슬픔", "분노", "불안", "놀람", "피곤", "후회"]
    writing = {
        0: "기분 좋은 하루에 이런 요리 어떠세요?",
        1: "이런 요리들은 어떠세요?",
        2: "기분을 전환 시켜줄 요리들이에요.",
        3: "마음을 가라앉혀 볼까요?",
        4: "이 요리들이 불안을 덜어주길 바래요.",
        5: "맛있는 요리 드시고 마음을 진정시켜 볼까요?",
        6: "지친 마음을 다독여줄 따뜻한 요리들을 추천해드릴게요.",
        7: "힘든 시기에 작은 행복을 전해줄 요리들을 추천할게요."
    }
    response = {"emotion": emotion_name[dominate],
                "writing": writing[dominate]}
    return jsonify(response)



if __name__ == '__main__':
    log.DEFAULT.info(f"///// Yomi AI | {VERSION} ... /////")

    os.environ['FLASK_ENV'] = 'development'
    os.environ['FLASK_DEBUG'] = '0'

    # check logger
    log.setup_logging()

    # ChatBot 인스턴스 생성
    chatbot = Yomi_ai.ChatBot()

    try:
        app.run(host='0.0.0.0', port=5000)
    except:
        log.DEFAULT.error(traceback.format_exc())
