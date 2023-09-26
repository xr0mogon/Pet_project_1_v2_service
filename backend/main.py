"""
Программа: Модели для предсказаний пола и возраста по данным с соревнования mts ml cup
Версия 1.0
"""

import warnings
import uvicorn

from fastapi import FastAPI

from src.pipeline.pipeline import training_all_models, random_predict
from src.transform.transform import first_preprocessing_data

warnings.filterwarnings("ignore")

app = FastAPI()
# CONFIG_PATH = "../config/params.yml"


@app.get("/hello")
def welcome():
    """
    Hello
    """
    return {'message': 'Hello Data Scientist!'}


@app.post("/first_prep_data")
def first_prep_data():
    """
    Использовать только если нужна первичная предобработка данных с сайта.
    Применяется полный цикл предобработки и сохранения данных.
    Требуется очень много ОЗУ и времени! У меня выполнялось это на 32гб озу и +60гб виртуальной памяти!!
    """
    print('start first_prep_data')
    first_preprocessing_data()
    print('end first_prep_data')


@app.post("/training_models")
def training_models():
    """
    * Использовать first_prep_data только если нужна первичная предобработка данных.
    Применяется полный цикл предобработки и сохранения данных.
    Требуется много ОЗУ и времени! У меня выполнялось это на 32гб озу и +70гб виртуальной памяти!!

    * Обучение и предсказания бейслайн(1), с подбором параметров(2) и на всей выборке(3) двух
    моделей по предсказанию возраста и пола.
    """
    # print('start first_prep_data')
    # first_preprocessing_data()
    # print('end first_prep_data')

    print("start training_all_models")
    training_all_models()
    print('done training_all_models')


@app.post("/rand_pred")
def rand_pred():
    """
    Т.к. ручником сложно ввести тысячи параметров,
    ф-ция делает предсказание по случайным данным.
    """

    return random_predict()


if __name__ == "__main__":
    # Запустите сервер, используя заданный хост и порт
    uvicorn.run(app, host="127.0.0.1", port=80)
