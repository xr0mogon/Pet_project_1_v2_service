"""
Программа: Frontend часть проекта
Версия: 1.0
"""

import yaml
import pandas as pd
import requests
import streamlit as st
# import sklearn

from src.train.show_metrics import show_score
# from fastapi import requests

from src.plottings.charts import cat_ages_bars, gender_bars, url_count_gender_bars, price_gender_bars, \
    price_gender_boxplot, manufacturer_gender_bars, manufacturer_age_bars, os_age_bars, os_gender_bars
from src.data.get_split_data import get_dataset

st.set_option('deprecation.showPyplotGlobalUse', False)

config_path = '../config/params.yml'


def main_page():
    """
    Страница с описанием проекта
    """
    st.image(
        "https://static.tildacdn.com/stor6431-3963-4235-b639-633734343536/67821931.png",
        width=600,
    )

    st.title("MLOps project:  MTS ML CUP")
    st.markdown("""https://ods.ai/competitions/mtsmlcup""")
    st.write(
        """
        Описание проекта: Соревнование от МТС Digital Big Data по определению **пола/возраста** владельца cookie.
        Определение пола и возраста владельца HTTP cookie по истории активности пользователя 
        в интернете на основе синтетических данных.
        """
    )
    # name of the columns
    st.markdown(
        """ 
        ### Описание данных:
        - Материалы (1 415 MB).
        - public_train.pqt (3 MB).
        - competition_data_final_pqt.zip (1 405 MB). 
        - Context_Baseline_Public.ipynb (1 MB). 
        - submit_2.pqt в этом файле id, по которым нужно предсказать пол и возраст (2 MB).
        - sample_submission.csv  По возрасту классы: Класс 1 —19-25, Класс 2 —26-35, Класс 3 —36-45, Класс 4 —46-55, 
            Класс 5 —56-65, Класс 6— 66+ (4 MB) 
        """
    )
    # name of the columns
    st.markdown(
        """
        ### Описание колонок файла с исходными данными: 
            - 'region_name' – Регион
            - 'city_name' – Населенный пункт
            - 'cpe_manufacturer_name' – Производитель устройства
            - 'cpe_model_name' – Модель устройства
            - 'url_host' – Домен, с которого пришел рекламный запрос
            - 'cpe_type_cd' – Тип устройства (смартфон или что-то другое)
            - 'Cpe_model_os_type' – Операционная система на устройстве
            - 'price' – Оценка цены устройства
            - 'date' – Дата
            - 'part_of_day' – Время дня (утро, вечер, итд)
            - 'request_cnt' – Число запросов одного пользователя за время дня (поле part_of_day)
            - 'user_id' – ID пользователя
            
        ### Описание колонок файла с таргетами:
            - 'age' – Возраст пользователя
            - 'Is_male' – Признак пользователя : мужчина (1-Да, 0-Нет)
            - 'user_id' – ID пользователя
    """
    )

    st.markdown(
        """
        ### Проверка решений:
        - Решения проверялись автоматически на сайте соревновании. \
        Запуск происходил на полностью закрытых тестовых данных, которые не передаются участникам.
        - Рейтинг участников рассчитывался по подвыборке ответов из тестовых данных.
        - Метрика соревнования: **ROC-AUC** – для определения **пола**, **f1 weighted** – для определения **возраста**. 
        - Все решения рассчитываются по формуле -  **2 * f1_weighted + gini** по полу.
        """

    )


def preprocessing_data():
    """
    Преобразование данных.
    """
    st.title("1. Преобразование данных")

    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        endpoint = config["endpoints"]["preprocessing_data"]

    st.markdown(
        """
        ### Для дальнейшей работы EDA, Train, Predict требуются преобразованные данные соревнований:
        - **ВНИМАНИЕ**: требуется большой памяти (Рекомендованные параметры 32гб ОЗУ + 60гб виртуальной).
        - Можно использовать уже подготовленные данные.
        """)

    if st.button("Start preprocessing data"):
        with st.spinner("Преобразование данных.."):
            # requests.post("http://localhost:8000/first_prep_data")
            requests.post(endpoint, timeout=8000)
        st.success("Success!")

    # st.markdown("**Или**")
    st.markdown("**Для работы следующих страниц преобразованные данные обязательны!**")


def exploratory():
    """
    Exploratory data analysis.
    """

    st.markdown("# 2. Exploratory data analysis parts")

    # write dataset
    # st.markdown(f"Размер объединённого датасета {get_dataset('all_prep_data.parquet').shape}")
    st.markdown("Пример целевых параметров из датасета с таргетами")
    st.write(get_dataset('prep_data_with_2_targets.parquet')[["user_id", "is_male", "age_target"]].head(6))

    # plotting with checkbox
    cat_ages_bars_ = st.sidebar.checkbox("Возрастные группы")
    gender_bars_ = st.sidebar.checkbox("Баланс по половому признаку.")
    url_count_gender_bars_ = st.sidebar.checkbox("Пол или возраст влияет на кол-во переходов по ссылкам.")
    price_gender_bars_ = st.sidebar.checkbox("Стоимость ус-ва влияет на пол или возраст.")
    # price_gender_boxplot_ = st.sidebar.checkbox("Стоимость ус-ва влияет на пол или возраст.")
    manufacturer_gender_bars_ = st.sidebar.checkbox("Производитель ус-ва влияет на пол.")
    manufacturer_age_bars_ = st.sidebar.checkbox("Производитель ус-ва влияет на возраст.")
    os_age_bars_ = st.sidebar.checkbox("Операционная система влияет на возраст.")
    os_gender_bars_ = st.sidebar.checkbox("Операционная система влияет на пол.")

    if cat_ages_bars_:
        st.markdown("##### 2.1.1. Целевой признак по возрасту.")
        st.pyplot(cat_ages_bars())
        st.markdown("- **Выводы**: Наблюдается **дисбаланс классов** целевой переменной **target_age**")
    if gender_bars_:
        st.markdown("##### 2.1.2. Целевой признак по полу.")
        st.pyplot(gender_bars())
        st.markdown("- **Выводы**: Класс **is_male** довольно **неплохо сбалансирован**!")
    if url_count_gender_bars_:
        st.markdown("##### 2.2. Исследование не целевых признаков и гипотезы по ним.")
        st.markdown("##### 2.2.1. Гипотеза: гендер или возраст влияет на кол-во переходов по ссылкам.")
        st.pyplot(url_count_gender_bars())
        st.markdown("- **Выводы**: Видна зависимость - чем **старше пользователь**, тем **меньше переходов по "
                    "ссылкам** совершает пользователь **вне зависимости от пола**.")
    if price_gender_bars_:
        st.markdown("##### 2.2.2. Гипотеза: стоимость ус-ва влияет на пол или возраст.")
        st.pyplot(price_gender_bars())
        st.markdown("- **Выводы**: Заметно, что у **молодых** пользователей более дорогие ус-ва, чаще у **женского "
                    "пола**, **с возрастом** тенденция меняется **наоборот**.")
    ## Не отрисовывает боксплот
    # if price_gender_boxplot_:
    #     st.markdown("##### ")
    #     st.markdown("- **Выводы**: Заметно, что у **молодых** пользователей более дорогие ус-ва, чаще у **женского "
    #                     "пола**, **с возрастом** тенденция меняется **наоборот**.")
    if manufacturer_gender_bars_:
        st.markdown("##### 2.2.3. Гипотеза: производитель ус-ва влияет на пол")
        st.pyplot(manufacturer_gender_bars())
        st.markdown("- **Вывод**: 1) Можно заметить, как у пользователей **apple** и **samsung** **женщин** **немного "
                    "больше**, чем мужчин, а у **'китайских'** устройств пользователей **мужчин больше**.")
    if manufacturer_age_bars_:
        st.markdown("##### 2.2.4. Гипотеза: производитель ус-ва влияет на возраст.")
        st.pyplot(manufacturer_age_bars())
        st.markdown("- **Вывод**: 2) Видно, как у **всех возрастных групп** *примерно одинаковая* картина "
                    "распределения относительно производителей телефонов, но больше всего у **apple**. Сильно "
                    "выделяется **apple** у которых **больше всего молодых пользователей 0 и 1** класса, "
                    "когда у **samsung больше** пользователей **2 и 3** класса.")
    if os_age_bars_:
        st.markdown("##### 2.2.5. Гипотеза: Операционная система влияет на возраст")
        st.pyplot(os_age_bars())
        st.markdown("- **Выводы**: у **android** пользователей сильнее всего **выделяются** возрастные группы **1, "
                    "2 и 3 класса**")
    if os_gender_bars_:
        st.markdown("##### 2.2.6. Гипотеза: Операционная система влияет на пол.")
        st.pyplot(os_gender_bars())
        st.markdown("- **Выводы**: В наших данных у владельцев **android вдвое больше пользователей** мужчин и женщин "
                    "чем у **apple**. Так же у владельцы **android** чаще всего **мужчины**.")


def training():
    """
    Тренировка модели
    """
    st.title("3. Обучение моделей")

    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        endpoint = config["endpoints"]["train"]

    st.markdown(
        """
        ### Будет выполнено последовательное обучение моделей по предсказанию пола и возраста:
        - CatBoost baseline предсказания возраста
        - CatBoost с подбором параметров с помощью randomized grid search для предсказания возраста
        - CatBoost с подобранными параметрами и обученный на всех данных для предсказания возраста требуемых сабмитов
        - CatBoost baseline предсказания пола
        - CatBoost с подбором параметров с помощью  randomized grid search для предсказания пола
        - CatBoost с подобранными параметрами и обученный на всех данных для предсказания пола требуемых сабмитов
        """)
    st.markdown("### Внимание! Для обучения требуется 32гб озу + не менее 60гб выделенной виртуальной памяти")
    if st.button("Start training models"):
        with st.spinner("Преобразование данных.."):
            # requests.post("http://localhost:8000/training_models")
            # requests.post('http://fastapi:8000/training_models')
            requests.post(endpoint)

        st.success("Success!")

        # st.markdown("Посмотреть, как изменяется дельта переобучения")
        # st.write((get_dataset('df_age_overfitting.parquet')).reset_index(drop=True))
        # st.write((get_dataset('df_age_overfitting.parquet')).reset_index(drop=True))

        # Метрики и Score
        st.markdown("Score высчитывается по формуле: **2 * f1_weighted + gini**.")
        show_score(key1=0, key2=1, show='y')

        # Метрики и Score по всем данным
        st.markdown("##### Ниже посмотрим на метрики, полученные на моделях, которые обучены на всех данных")
        show_score(key1=1, key2=2)
        st.markdown(
            "**Метрики заметно улучшаются, но, как я понял из условий, скор засчитывается только по тестовым"
            "данным**")


def rand_pred():
    """
    Т.к. данных тысячи и вручную ввести их невозможно, делаем предсказания по случайным данным.
    """
    st.title("4. Предсказание по случайно сгенерированным данным")

    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        endpoint = config["endpoints"]["rand_pred"]

    if st.button("Random predict"):
        with st.spinner("Предсказание по случайным данным.."):
            # output = requests.post(endpoint)
            requests.post(endpoint)

        st.success("Success!")

        df = pd.read_parquet(f"{config['PREP_DATA']}/random_pred_df.parquet")

        st.write(df[['cpe_manufacturer_name', 'cpe_model_os_type', 'is_male', 'age']])
        # st.write(pd.DataFrame(output.json()))


def main():
    """
    Сборка пайплайна в одном блоке
    """
    page_names_to_funcs = {
        "Описание проекта": main_page,
        "1. Preprocessing data": preprocessing_data,
        "2. Exploratory data analysis": exploratory,
        "3. Training model": training,
        "4. Random predict": rand_pred
    }
    selected_page = st.sidebar.selectbox("Выберите пункт", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()


if __name__ == "__main__":
    main()
