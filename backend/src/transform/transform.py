"""
Программа: Функции для предобработки данных
Версия: 1.0
"""

import os
import pandas as pd
import numpy as np
import bisect
import gc
import yaml

from typing import Text

import warnings

warnings.filterwarnings("ignore")

config_path = '../config/params.yml'


def age_bucket(x):
    """
    Разбиваем возраст в виде новых классов
    "Класс 1: 19-25,..."
    """
    return bisect.bisect_left([25, 35, 45, 55, 65], x)


def prep_targets():
    """
    Функция предобработки таргета
    """
    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # load
    targets = pd.read_parquet(f"{config['LOCAL_DATA_PATH_mts']}/{config['TARGET_FILE']}").reset_index(drop=True)
    # targets = pd.read_parquet(f'{LOCAL_DATA_PATH_mts}/{TARGET_FILE}').reset_index(drop=True)

    # по условию отбрасываем всех, кто младше 19
    targets = targets.drop(targets.query("age < 19").index)

    # убираем пропуски
    targets.dropna(inplace=True)

    # добавляем новый признак с "классификацией" возрастов
    targets['age_target'] = targets['age'].map(age_bucket)

    # targets_age
    targets_age = targets[['user_id', 'age_target']].reset_index(drop=True)
    targets_age[['age_target']] = targets_age[['age_target']].astype('int8')

    # targets_is_male
    targets_is_male = targets[['user_id', 'is_male']]
    targets_is_male = targets_is_male.drop(
        targets_is_male.query(
            "is_male == 'NA'").index).reset_index(drop=True)

    targets_is_male['is_male'] = targets_is_male['is_male'].astype('int8')

    # saving
    targets_age.to_parquet(f"{config['PREP_DATA']}/targets_age_prep.parquet")
    targets_is_male.to_parquet(f"{config['PREP_DATA']}/targets_is_male_prep.parquet")
    # targets_age.to_parquet(f"{LOCAL_DATA_PATH}/{PREP_DATA}/targets_age_prep.parquet")
    # targets_is_male.to_parquet(f"{LOCAL_DATA_PATH}/{PREP_DATA}/targets_is_male_prep.parquet")

    del targets, targets_age, targets_is_male
    gc.collect()


# 2. Функции по предобработке данных.
# Получаем список с предобработанными данными
# part_list = os.listdir(path=f"./{config['LOCAL_DATA_PATH_mts']}/competition_data_final_pqt")


def unite_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ф-ция unite_categories
    объединяет повторяющиеся имена производителей смартфонов,
    полные названия фирм и названия операционных систем

    Parameters
    ----------
    df - датасет с колонками
    'user_id', 'cpe_manufacturer_name', 'cpe_model_os_type'

    Returns
    -------
    Предобработанный датасет c колонками
    'user_id', 'cpe_manufacturer_name', 'cpe_model_os_type'

    return pd.frame.DataFrame
    """

    df = df[['user_id', 'cpe_manufacturer_name', 'cpe_model_os_type']]

    df["cpe_manufacturer_name"].replace("Huawei Device Company Limited",
                                        "Huawei", inplace=True)
    df["cpe_manufacturer_name"].replace("Realme Chongqing Mobile Telecommunications Corp Ltd", "Realme", inplace=True),
    df["cpe_manufacturer_name"].replace("Realme Mobile Telecommunications (Shenzhen) Co Ltd", "Realme", inplace=True),
    df["cpe_manufacturer_name"].replace("Sony Mobile Communications Inc.", "Sony", inplace=True),
    df["cpe_manufacturer_name"].replace("Honor Device Company Limited", 'Honor', inplace=True)

    df["cpe_model_os_type"].replace("Apple iOS", "iOS", inplace=True)

    df = df.drop_duplicates(subset=['user_id'])

    gc.collect()
    print('unite_categories done')

    return df


def preparation_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ф-ция preparation_numbers
    заполняет пропуски стоимости ус-ва,
    создаёт новые признаки частей дня
    и подсчитывает общее количество переходов по ссылкам
    для новых признаков по каждому user_id

    Parameters
    ----------
     df - датасет с колонками
     'price','part_of_day','request_cnt','user_id'

    Returns
    -------
    Предобработанный датасет c нов. колонками
    'price', 'user_id', 'day', 'evening', 'morning', 'night'

    return pd.core.frame.DataFrame
    """

    # отбор нужных признаков
    df = df[['price', 'part_of_day', 'request_cnt', 'user_id']]

    # Заполнение пропусков стоимости медианой
    df["price"] = df["price"].fillna(np.nanmedian(df["price"].values))

    # Создание новых признаков с общим количеством запросов URL в определённые части дня
    df_part_of_day = df.pivot_table(index='user_id',
                                    columns='part_of_day',
                                    values='request_cnt',
                                    aggfunc='count'
                                    ).reset_index().fillna(0)

    df = df.merge(df_part_of_day, how="inner", on=["user_id"])

    df = df.drop_duplicates(subset=['user_id'])
    df = df.drop(columns=['part_of_day', 'request_cnt'], axis=1)

    gc.collect()
    print('preparation_numbers done')

    return df


def add_day_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ф-ция add_day_names
    создает новые признаки и превращает даты в дни недели,
    подсчитывает количество переходов по ссылкам
    в день недели по каждому user_id

    Parameters
    ----------
    df - датасет с колонками
    'date','request_cnt','user_id'

    Returns
    -------
    Предобработанный датасет c нов. колонками
    'user_id' 'fri' 'mon' 'sat' 'sun' 'thu' 'tue' 'wed'

    return pd.core.frame.DataFrame
    """

    df = df[['date', 'request_cnt', 'user_id']]

    df['day_name'] = df['date'].apply(lambda x: x.strftime('%a').lower())

    df = df.pivot_table(index='user_id',
                        columns='day_name',
                        values='request_cnt',  # ?
                        aggfunc='count').reset_index().fillna(0)

    gc.collect()
    print('add_day_names done')

    return df


def top_url_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ф-ция top_url_filter
    1) слежение за id_to_submit в текущем df
    что бы не потерять нужные submit_id
    2) Подсчёт кол-ва переходов по url в текущем df
    3) отбор самых запрашиваемых URL
    где кол-во переходов по url > n раз.
    4) оставляем в рабочем датасете только топовые URL
    5) создаём датасет, где 1 строка это 1 id и все топ URL в виде новых признаков
    6) проверка получившегося датасета на пропуски submit_id по 1ому шагу

    Parameters
    ----------
    submit_size - количество id_to_submit в текущем df
    filtering_df - df с урлами и общим кол-вом переходов по ним
    top_urls - df c отфильтрованными url
    df_new - df где сначала оставляем только топовые url, а потом
    делаем из них новые признаки
    фильтрация незначительных URL

    Returns
    -------
    Предобработанный датасет c нов. колонками
    'user_id' и много url в виде признаков

    return pd.DataFrame
    """

    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # выполняем топупл с сэмплиингом или без
    key_smpl = config['key_smpl']
    # фильтр топ url(чем >, тем точнее)
    n = config['key_n']

    print('top_url_filter start')
    id_to_submit = pd.read_parquet(f"{config['LOCAL_DATA_PATH_mts']}/{config['SUBMIT_FILE']}").reset_index(drop=True)

    n_filter = int(n)

    # 1)
    submit_size = df.merge(id_to_submit,
                           how="inner",
                           on=["user_id"]).drop_duplicates(subset=['user_id']).shape[0]
    print(f"-submits in curr dataset {submit_size}")

    # 2)
    filtering_df = df['url_host'].value_counts().reset_index(name='url_count')
    filtering_df.rename(columns={'index': 'url_host'}, inplace=True)

    # 3)
    top_urls = filtering_df[filtering_df['url_count'] > n_filter].drop(columns=['url_count'], axis=1)

    top_urls = top_urls[top_urls["url_host"].str.contains("--") == False]

    # 4)
    df_new = df.merge(top_urls, how='inner', on=["url_host"])

    # 5)
    df_new = df_new.pivot_table(index='user_id',
                                columns='url_host',
                                values='request_cnt',
                                aggfunc='count').reset_index().fillna(0)

    print(f"-df_new with top_url {df_new.shape}")

    # 6) проверяем получившийся df на пропуски submit_id, работает только без семплинга
    if key_smpl == 'N':
        new_submit = df_new.merge(id_to_submit, how="inner", on=["user_id"]).shape[0]
        print(f"-new_submit.shape {new_submit}")

        # если вдруг потеряется submit_id, то уменьшаем n_filter
        # и submit_id может появиться
        while submit_size != new_submit:
            n /= 1.3
            top_url_filter(df)

        gc.collect()
        print(f"top_url_filter done\n")

    return df_new


def pipeline_preprocess():
    """
    Ф-ция предобработки и сохранения основных данных.
    Цикл получения, предобработки и сохранения данных.
    :return: None
    """

    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    key_smpl = config['key_smpl']
    pct_smpl = config['percent_of_sample']

    print('top_url_filter start')
    id_to_submit = pd.read_parquet(f"{config['LOCAL_DATA_PATH_mts']}/{config['SUBMIT_FILE']}").reset_index(drop=True)
    part_list = os.listdir(path=f"./{config['LOCAL_DATA_PATH_mts']}/competition_data_final_pqt")
    prep_df_with_submit = 0  # счётчик

    # Цикл предобработки и сохранения всех частей данных.
    for i in range(len(part_list)):
        print(f'Data preparation part {i + 1}/10')
        # чтение данных
        temp_df = pd.read_parquet(f"{config['LOCAL_DATA_PATH_mts']}/"
                                  f"{config['DATA_FILE']}/{part_list[i]}").sample(frac=pct_smpl)
        print(f"loaded df.shape {temp_df.shape}\n")

        # 1) Подготовка категориальных признаков и частей дня
        prep_cat = unite_categories(temp_df)

        # 2) Подготовка числовых признаков
        prep_nums = preparation_numbers(temp_df)

        # 3) Подготовка признаков от даты ко дню недели
        prep_days = add_day_names(temp_df)

        # 4) Отбор самых запрашиваемых URL top_url_filter(df: pd.DataFrame, n=100, key_smpl='N')
        prep_url = top_url_filter(temp_df)

        # Объединение подготовленных данных по id
        merge1 = prep_cat.merge(prep_nums, how="inner", on=["user_id"])  # 1+2
        merge2 = merge1.merge(prep_days, how="inner", on=["user_id"])  # (1+2)+3
        temp_df = prep_url.merge(merge2, how="inner", on=["user_id"])  # (1+2+3)+4
        print("merge done")
        print(f"df.isna {temp_df.isna().sum().sum()}\n")

        # Запись подготовленного датасета в файл
        temp_df.to_parquet(f"{config['PREP_DATA']}/preprocessed_part_{i}.parquet")
        # temp_df.to_parquet(f"{LOCAL_DATA_PATH}/{PREP_DATA}//preprocessed_part_{i}.parquet")

        # отслеживание за отсутствием потерей сабмитов после отбора признаков, но не с сэмплами
        if key_smpl == 'N':
            prep_df_with_submit += temp_df.merge(id_to_submit,
                                                 how="inner",
                                                 on=["user_id"]
                                                 ).drop_duplicates(subset=["user_id"]).shape[0]

            print(f"preprocessed_data_part_{i} shape {temp_df.shape}\nsave on disc.")
            print("-------------\n")

        del temp_df, prep_cat, prep_nums, prep_days, prep_url, merge1, merge2

    # if id_to_submit.shape[0] == prep_df_with_submit:
    #     print("Number of submissions match the submissions in the prepared data?")
    #     print(f"{id_to_submit.shape[0]} == {prep_df_with_submit}")
    #     print(f"{id_to_submit.shape[0] == prep_df_with_submit}")
    #     print("Data preparation completed successfully")
    # else:
    #     print("Data preparation is not satisfactory, try changing the preparation settings")

    del prep_df_with_submit, id_to_submit

    gc.collect()


def merge_parts():
    """
    Требуется много оперативной памяти (32гб + 60гб виртуальной) на данных без семплинга
    можно использовать уже объединенные данные
    """

    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # load
    list_data_dir = os.listdir(path=f"./{config['LOCAL_DATA_PATH']}/preprocessed_data")
    # list_data_dir = os.listdir(path=f"./{LOCAL_DATA_PATH}/preprocessed_data")

    # список с предобработанными данными
    list_data_prep = []

    for i in range(len(list_data_dir)):
        if 'preprocessed_part' in list_data_dir[i]:
            list_data_prep.append(list_data_dir[i])

    # Счётчики
    all_prep_parts = []
    #     size = 0

    for i in range(len(list_data_prep)):
        print(f'Loaded prep part {i}')
        # Чтение
        temp_df = pd.read_parquet(f"{config['LOCAL_DATA_PATH']}/{config['PREP_DATA']}/{list_data_prep[i]}")

        # Уменьшение памяти
        temp_df = temp_df.astype(np.int32(0), errors='ignore')

        # массив со всеми данными для объединения
        all_prep_parts.append(temp_df)

    del list_data_prep, temp_df, list_data_dir

    # Соединение всех частей
    all_df = pd.concat(all_prep_parts, ignore_index=True)
    print(f'\nconcat {all_df.shape}\nall_df.isna() = {all_df.isna().sum().sum()}')
    del all_prep_parts

    # работа с пропусками и размерами данных
    all_df = all_df.fillna(value=np.int32(0))
    print(f'\nfillna\nall_df.isna() = {all_df.isna().sum().sum()}')

    all_df = all_df.astype(np.int32, errors='ignore')
    all_df[['cpe_manufacturer_name',
            'cpe_model_os_type']] = all_df[['cpe_manufacturer_name',
                                            'cpe_model_os_type']].astype('category')
    print(f"\nastype(np.int32) + .astype('category')")

    # Запись подготовленного общего датасета в файл
    all_df.to_parquet(f"{config['LOCAL_DATA_PATH']}/{config['PREP_DATA']}/all_prep_data.parquet")
    print('all_prep_data was saved to file successfully')

    del all_df
    gc.collect()


def merge_submits():
    """
    Датасет для предсказаний по id_to_submit.
    id_to_submit - id по которым нужно сделать предсказания
    df_submit - all_prep_data
    """
    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # load
    id_to_submit = pd.read_parquet(f"{config['LOCAL_DATA_PATH_mts']}/{config['SUBMIT_FILE']}").reset_index(drop=True)
    df_submit = pd.read_parquet(f"{config['PREP_DATA']}/all_prep_data.parquet")

    # merge id_to_submit и all_prep_data по "user_id"
    # Соединяю все полученные данные с искомыми id из сабмита для требуемых предсказаний
    df_submit = df_submit.merge(id_to_submit, how="inner", on=["user_id"])

    # save
    df_submit.to_parquet(f"{config['PREP_DATA']}/df_submit.parquet")
    print('df_submit was saved to file successfully')

    del id_to_submit, df_submit
    gc.collect()


def merge_target(target: Text = "Target_n"):
    """
    Ф-ция создает датасет для обучения с целевой переменной.
    :param target: Text с названием файла.
    :return: None
    """
    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # load
    all_df = pd.read_parquet(f"{config['PREP_DATA']}/all_prep_data.parquet")
    targets = pd.read_parquet(f"{config['PREP_DATA']}/{target}")

    # объединение targets и all_prep_data
    # для первичного обучения моделей
    df = all_df.merge(targets, how="inner", on=["user_id"])

    # save
    if target == "targets_age_prep.parquet":
        # Запись prep. Данных с учителем после общего объединения
        df.to_parquet(f"{config['PREP_DATA']}/prep_data_with_targets_age.parquet")
        print('prep_data_with_targets_age was merge and saved to file successfully')

    elif target == "targets_is_male_prep.parquet":
        df.to_parquet(f"{config['PREP_DATA']}/prep_data_with_targets_is_male.parquet")
        print('prep_data_with_targets_is_male was merge and saved to file successfully')

    else:
        print("wrong target key")

    del all_df, targets, df
    gc.collect()


def merge_both_target():
    """
    Для удобства исследования создаю общий датасет с полом и возрастом.
    """
    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Объединение таргетов
    targets1 = pd.read_parquet(f"{config['PREP_DATA']}/targets_age_prep.parquet")
    targets2 = pd.read_parquet(f"{config['PREP_DATA']}/targets_is_male_prep.parquet")
    targets_df = targets1.merge(targets2, how="inner", on=["user_id"])

    # Объединение таргетов с предобработанным датасетом
    all_df = pd.read_parquet(f"{config['PREP_DATA']}/all_prep_data.parquet")
    df = all_df.merge(targets_df, how="inner", on=["user_id"])

    # Запись
    df.to_parquet(f"{config['PREP_DATA']}/prep_data_with_2_targets.parquet")

    del targets1, targets2, targets_df, all_df, df
    gc.collect()
    print('df_with_targets was saved to file successfully')


def first_preprocessing_data():
    """
    Использовать только если нужна первичная предобработка данных с сайта.
    Применяется полный цикл предобработки и сохранения данных.
    Требуется очень много ОЗУ! У меня выполнялось это на 32гб озу и +60гб виртуальной памяти!!
    """
    pipeline_preprocess()
    print('pipeline_preprocess')
    merge_parts()
    print('merge_parts')
    merge_submits()
    print('merge_submits')
    prep_targets()
    print('prep_targets')
    merge_target("targets_age_prep.parquet")
    merge_target("targets_is_male_prep.parquet")
    merge_both_target()

    gc.collect()
