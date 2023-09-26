from typing import Tuple, Any

import pandas as pd
import numpy as np
import joblib
import yaml

from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, roc_auc_score
from random import randint
from ..data.get_split_data import get_dataset

config_path = '../config/params.yml'


def get_metrics_multiclass(y_test_bin, y_test, y_pred, y_prob, name, type_multi):
    """
    Функция для получения метриков
    y_test_bin - бинаризованные тестовые метки класса
    y_test - метки класса без бинаризации
    y_prob - предсказанные вероятности классов
    name - название модели/подхода
    type_multi - тип многоклассовой классификации для ROC-AUC (ovo/ovr)
    """

    df_metrics = pd.DataFrame()

    df_metrics['model'] = [name]
    df_metrics['F1_weighted'] = float('{:.4f}'.format(f1_score(y_test, y_pred, average="weighted")))
    # df_metrics['F1_weighted'] = f1_score(y_test, y_pred, average="weighted")

    return df_metrics


def get_f1_score_overfitting(model, X_train, y_train, X_test, y_test, name):
    """
    Ф-ция для проверки на переобучение
    """
    df_metrics = pd.DataFrame()

    df_metrics['model'] = [name]

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    value_train = f1_score(y_train, y_pred_train, average="weighted")
    value_test = f1_score(y_test, y_pred_test, average="weighted")

    df_metrics["F1_weighted_train"] = float('{:.4f}'.format(value_train))
    df_metrics["F1_weighted_test"] = float('{:.4f}'.format(value_test))
    df_metrics["delta"] = f'{(abs(value_train - value_test) / value_test * 100):.1f} %'

    return df_metrics


def func_rgs_age(X_train, X_test, y_train, y_test, eval_set, cat_features):
    """
    Ф-ция выполняет подбор параметров для катбуста по randomized grid search.
    :return: dict
    """
    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    grid = {
        "n_estimators": [1500],
        "learning_rate": [0.11958499997854231],
        "boosting_type": ['Ordered', 'Plain'],  #
        "bootstrap_type": ["Bayesian", "Bernoulli", "MVS"],
        "grow_policy": ["SymmetricTree", "Depthwise", "Lossguide"],
        "custom_metric": ['F1'],
        "max_depth": list(range(6, 10)),
        "l2_leaf_reg": [*np.arange(1, 10)],  #
        "random_state": config['RAND']
    }

    model = CatBoostClassifier(silent=True,
                               cat_features=cat_features,  # ?
                               task_type=config['task_type'],
                               early_stopping_rounds=100
                               )

    grid_search_result = model.randomized_search(grid,
                                                 X=X_train,
                                                 y=y_train,
                                                 cv=5,
                                                 n_iter=50,
                                                 refit=True,
                                                 shuffle=True,
                                                 stratified=True,
                                                 calc_cv_statistics=True,
                                                 search_by_train_test_split=True,
                                                 verbose=False,
                                                 plot=False)  # True

    joblib.dump(grid_search_result['params'], f"{config['PREP_DATA']}/grid_search_result_age.pkl")
    print("rgs age done")
    return grid_search_result['params']


def func_rgs_gender(X_train, X_test, y_train, y_test, eval_set, cat_features):
    """
    Ф-ция выполняет подбор параметров для катбуста по randomized grid search.
    :return: dict
    """
    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # если требуется подбор параметров
    grid = {
        "n_estimators": [1500],
        "learning_rate": [0.03863900154829025],
        "boosting_type": ['Plain', 'Ordered'],
        "bootstrap_type": ["Bayesian", "Bernoulli"],  # ,"MVS" прерывает работу rgs
        "grow_policy": ["SymmetricTree", "Depthwise", "Lossguide"],
        "l2_leaf_reg": np.arange(0.1, 1, 0.05),  # , "None"
        "random_strength": [1, 2, 5, 10, 20, 50, 100, "None"],
        "random_state": config['RAND']
    }

    model = CatBoostClassifier(silent=True,
                               cat_features=cat_features,
                               task_type=config['task_type'],
                               early_stopping_rounds=100
                               )

    grid_search_result = model.randomized_search(grid,
                                                 X=X_train,
                                                 y=y_train,
                                                 cv=5,  #
                                                 n_iter=50,  #
                                                 refit=True,  #
                                                 shuffle=True,  #
                                                 stratified=True,  #
                                                 calc_cv_statistics=True,  #
                                                 search_by_train_test_split=True,  #
                                                 verbose=False,
                                                 plot=False)  # True

    joblib.dump(grid_search_result['params'], f"{config['PREP_DATA']}/grid_search_result_gender.pkl")
    print("rgs gender done")

    return grid_search_result['params']


def get_metrics_gini(y_test, y_pred, name):
    """
    Функция для получения метрики Gini по формуле (2* roc_auc -1)
    :param y_pred: y_pred
    :param y_test: y_test
    :param name: имя модели
    :return: DataFrame
    """
    df_metrics = pd.DataFrame()
    df_metrics['model'] = [name]
    df_metrics['Gini'] = float('{:.4f}'.format(2 * roc_auc_score(y_test, y_pred) - 1))
    return df_metrics


def check_gini_overfitting(model, X_train, y_train, X_test, y_test, metric_fun, name):
    """
    Проверка на overfitting
    Функция добавляет метрики в датасет для отслеживания переобучения модели.
    :param model: модель
    :param X_train: X_train
    :param y_train: y_train
    :param X_test: X_test
    :param y_test: y_test
    :param metric_fun:
    :param name: roc_auc_score
    :return: pd.dataframe
    """

    df_metrics = pd.DataFrame()

    df_metrics['model'] = [name]

    y_pred_train = model.predict_proba(X_train)[:, 1]
    y_pred_test = model.predict_proba(X_test)[:, 1]
    value_train = 2 * metric_fun(y_train, y_pred_train) - 1
    value_test = 2 * metric_fun(y_test, y_pred_test) - 1

    df_metrics["Gini_train"] = float('{:.4f}'.format(value_train))
    df_metrics["Gini_test"] = float('{:.4f}'.format(value_test))
    df_metrics["delta"] = f'{(abs(value_train - value_test) / value_test * 100):.1f} %'
    return df_metrics


def rand_os_name() -> Tuple[Any, Any]:
    """
    Т.к. основная часть признаков в датасете это url, берём для наглядности операционную систему и устройство,
    с которого перешли по url.
    Ф-ция возвращает список со случайным производителем и операционной системой для случайного предсказания.
    :return: Tuple[rand_pred_name, rand_pred_os]
    """
    # загрузка данных
    df = get_dataset('df_submit.parquet')

    # словарь со всеми произв и ос
    my_dict = {
        "name": df['cpe_manufacturer_name'].unique().to_list(),
        "type": df['cpe_model_os_type'].unique().to_list()
    }

    # выбор производителя
    rnd = len(my_dict['name'])
    rand_pred_name = my_dict['name'][randint(0, rnd - 1)]
    # rand_pred_name = my_dict['name'][1] # если нужен Apple и iOS

    if rand_pred_name == 'Apple':
        rand_pred_os = my_dict['type'][1]
        return rand_pred_name, rand_pred_os
    else:
        rand_pred_os = my_dict['type'][0]
        return rand_pred_name, rand_pred_os


def rand_data_pred() -> pd.DataFrame:
    """
    Ф-ция берёт 1 строку из датасета с названиями признаков, заполняет её случайными значениями
    и делает по ним предсказания пола и возраста.
    :return: DataFrame
    """
    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # загрузка данных
    df = get_dataset('df_submit.parquet').drop(['user_id'], axis=1)[:1]  #

    # Заполнение всех признаков случайным числом
    for col in df:
        df[col] = randint(0, 1)

    # Заполнение произв. и ос. случайными значениями
    name, os_type = rand_os_name()
    df[['cpe_manufacturer_name']] = name
    df[['cpe_model_os_type']] = os_type


    # df[['cpe_manufacturer_name', 'cpe_model_os_type', 'is_male', 'age']]
    # df.to_parquet(f"{config['PREP_DATA']}/random_pred_df.parquet")

    return df
