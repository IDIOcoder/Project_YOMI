import os
import json
import logging
import logging.config

DEFAULT = logging.getLogger('default')
CHATBOT = logging.getLogger('chatbot')


def setup_logging():
    print("[DEBUG] setting up logger")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    config_path = os.path.join(current_dir, 'logger.json')


    # log dir이 있는지 확인하고 없다면 생성
    logs_dir = os.path.join(project_dir, 'logs')
    print(f"[DEBUG] log_path: {logs_dir}")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # 로그 파일들 유무 체크 및 생성
    log_files = ['chatbot.log', 'debug.log', 'error.log']
    for log_file in log_files:
        log_file_path = os.path.join(logs_dir, log_file)
        if not os.path.exists(log_file_path):
            with open(log_file_path, 'w') as f:
                pass    # 파일생성

    with open(config_path, 'r') as f:
        config = json.load(f)
        logging.config.dictConfig(config)


def get_logger(name):
    return logging.getLogger(name)
