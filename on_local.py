import os
import traceback
import Yomi_ai
import utils.logger_utils as log
import utils.recipe_utils as DB

# 병렬 처리 경고 출력 X
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def run():
    while True:
        usr_input = input("U S E R || ")
        # quit 또는 exit입력시 종료합니다.
        if usr_input == 'quit' or usr_input == 'exit':
            main_logger.info('Shutting down...\n')
            break

        response = chatbot.get_response(usr_input, 'local')
        message = response['message']
        print(f"CHATBOT || {message}")


if __name__ == "__main__":
    main_logger = log.get_logger('default')
    chatbot_logger = log.get_logger('chatbot')
    main_logger.info('Starting Yomi-AI / Version: 0.0.1-alpha')

    chatbot = Yomi_ai.ChatBot()
    try:
        run()
    except:
        main_logger.error(traceback.format_exc())
