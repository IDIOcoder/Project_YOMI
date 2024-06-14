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


@app.route('/', methods=['POST'])
def hello():
    text = (DB.get_recipe('히야앗코'))
    response = {"message": markdown.markdown(text)}
    time.sleep(5)
    return jsonify(response)


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


if __name__ == '__main__':
    log.DEFAULT.info(f"///// Yomi AI | {VERSION} ... /////")
    # ChatBot 인스턴스 생성
    chatbot = Yomi_ai.ChatBot()

    try:
        app.run(host='0.0.0.0', port=5050)
    except:
        log.DEFAULT.error(traceback.format_exc())
