import json
import os
import logger_utils as log

# USER FOLDER PATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
folder_path = os.path.join(project_dir, 'user')


def check_folder():
    if not os.path.exists(folder_path):
        log.DEFAULT.warning("User folder doesn't exist. Creating new folder...")
        os.makedirs(folder_path)
    else:
        log.DEFAULT.info("User folder exists!")


def check_usr_state(usr_id):
    check_folder()
    file_name = f'{usr_id}.json'


