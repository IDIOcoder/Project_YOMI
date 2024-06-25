import pymysql
import os
from dotenv import load_dotenv
import utils.logger_utils as log

# MySQL 패스워드를 가져옵니다.
load_dotenv()
SQL_KEY = os.environ.get('MYSQL_KEY')

# Logger를 설정합니다.
logger = log.get_logger("default")


# 로컬 DB로 부터 레시피 정보를 받아옵니다.
def search_recipe(dish_name: str):
    connection = pymysql.connect(
        host='172.25.0.96',
        port=3306,
        user='newon',
        password=SQL_KEY,
        db='YOMI',
        charset='utf8'
    )
    try:
        with connection.cursor() as cur:
            query = f"SELECT * FROM recipe WHERE dish=%s"
            cur.execute(query, (dish_name,))
            search_result = cur.fetchall()
            return search_result
    except Exception as e:
        logger.error('Error while getting recipe from MySQL', e)
    finally:
        connection.close()


# 레시피를 프론트 출력형식에 맞추어 반환합니다.
def get_recipe(dish_name: str):
    recipe_data = search_recipe(dish_name)
    dish = f"<h1>{recipe_data[0][0]}</h1>"
    ingredient = f"### 재료\t<ul>{li_form(recipe_data[0][1])}</ul>"
    cook_process = f"### 조리과정\t<ol>{li_form(recipe_data[0][2])}<ol>"
    result = dish + ingredient + cook_process
    return result


# #으로 구분된 문자열을 li형식의 문장들로 변환
def li_form(data: str):
    lines = data.split('#')
    # 공백인 줄 제거
    lines = [line for line in lines if line != ""]
    result = ""
    for line in lines:
        result += f"<li>{line}</li>"
    return result


# 유사도 비교를 위해 DB에 등록된 요리들의 이름을 가진 파일을 업데이트 합니다.
def get_dish_names():
    connection = pymysql.connect(
        host='172.25.0.96',
        port=3306,
        user='newon',
        password=SQL_KEY,
        db='YOMI',
        charset='utf8'
    )
    try:
        with connection.cursor() as cur:
            query = f"SELECT dish FROM recipe"
            cur.execute(query)
            search_result = cur.fetchall()
            names = []
            for name in search_result:
                names.append(name[0])
            return names
    except Exception as e:
        logger.error('Error while update dish_name from MySQL', e)
    finally:
        connection.close()


def get_rand_dish():
    connection = pymysql.connect(
        host='172.25.0.96',
        port=3306,
        user='newon',
        password=SQL_KEY,
        db='YOMI',
        charset='utf8'
    )
    try:
        with connection.cursor() as cur:
            query = "SELECT dish FROM recipe ORDER BY RAND() LIMIT 5"
            cur.execute(query)
            search_result = cur.fetchall()
            names = []
            for name in search_result:
                names.append(name[0])
            return names
    except Exception as e:
        logger.error('Error while get random dish from MySQL', e)
    finally:
        connection.close()