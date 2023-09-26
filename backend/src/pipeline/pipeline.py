"""
Программа: Сборный конвейер для тренировки модели
Версия: 1.0
"""

import pandas as pd
import joblib
import yaml

from typing import Text

from catboost import CatBoostClassifier
from sklearn.preprocessing import label_binarize

from sklearn.metrics import roc_auc_score  # f1_score,

from ..data.get_split_data import get_dataset, split_train_test
from ..train.train import get_metrics_multiclass, get_metrics_gini, get_f1_score_overfitting, \
    check_gini_overfitting, func_rgs_gender, func_rgs_age, rand_data_pred

import warnings

warnings.filterwarnings("ignore")

config_path = '../config/params.yml'


def study_age_1(study_name: Text):
    """
    Ф-ция обучает первую бейслайн модель и записывает результаты метрик в датасет
    :param study_name: Text
    :return: None
    """

    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    X_train, X_test, y_train, y_test, eval_set, cat_features, X_train_, y_train_ = split_train_test(
        "prep_data_with_targets_age.parquet",
        "age_target")

    cat_params = {
        'n_estimators': 1500,
        'random_state': config['RAND'],
        'early_stopping_rounds': 100,
        'custom_loss': ['TotalF1'],
        'cat_features': cat_features
    }

    model = CatBoostClassifier(**cat_params,
                               # task_type='GPU')
                               task_type=config['task_type'])

    model.fit(X_train,
              y_train,
              eval_set=eval_set,
              verbose=False)

    print(f"study_age_1 {study_name} fit")

    # дата сеты с метриками моделей
    y = get_dataset("prep_data_with_targets_age.parquet")["age_target"]
    y_test_bin = label_binarize(y_test, classes=list(set(y)))

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)

    # Метрики
    df_metrics = get_metrics_multiclass(y_test_bin=y_test_bin,
                                        y_test=y_test,
                                        y_pred=y_pred,
                                        y_prob=y_score,
                                        name=study_name,
                                        type_multi='ovo')
    df_metrics.to_parquet(f"{config['PREP_DATA']}/df_age_metrics.parquet")

    # Переобучение
    df_overfitting = get_f1_score_overfitting(model,
                                              X_train,
                                              y_train,
                                              X_test,
                                              y_test,
                                              study_name)
    df_overfitting.to_parquet(f"{config['PREP_DATA']}/df_age_overfitting.parquet")

    print(f"study_age_1 metrics added {study_name} done and saved")


def study_age_2(study_name: Text, key: Text = 'old'):
    """
    Ф-ция выполняет поиск оптимальных параметров если это нужно, записывает модель
    и добавляет новые метрики на новых параметрах.

    :param key: Text : Отвечает за подбор новых параметров randomized_search.
    :param study_name: Text
    :return: None
    """

    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    X_train, X_test, y_train, y_test, eval_set, cat_features, X_train_, y_train_ = split_train_test(
        "prep_data_with_targets_age.parquet",
        "age_target")

    if key == 'new':
        # Если требуется подбор параметров #################
        grid_search_result = func_rgs_age(X_train, X_test, y_train, y_test, eval_set, cat_features)
    elif key == 'old':
        # пользуемся уже подобранными параметрами rgs
        grid_search_result = joblib.load(f"{config['PREP_DATA']}/grid_search_result_age.pkl")

    else:
        print('wrong key')

    # тюнинг модели подобранными параметрами
    cat_rgs = CatBoostClassifier(**grid_search_result,
                                 task_type=config['task_type'],
                                 loss_function='MultiClass')

    cat_rgs.fit(X_train_,
                y_train_,
                cat_features=cat_features,
                eval_set=eval_set,
                verbose=False,
                early_stopping_rounds=100)

    # saving model
    joblib.dump(cat_rgs, f"{config['PREP_DATA']}/model_cat_grid_age.pkl")
    print(f"study_age_2 {study_name} fit")

    # предикты для датасетов с метриками моделей
    y = get_dataset("prep_data_with_targets_age.parquet")["age_target"]
    y_test_bin = label_binarize(y_test, classes=list(set(y)))

    y_pred = cat_rgs.predict(X_test)
    y_score = cat_rgs.predict_proba(X_test)

    # Добавление метрик
    df_metrics = get_dataset('df_age_metrics.parquet')
    df_metrics = pd.concat([df_metrics,
                            get_metrics_multiclass(y_test_bin=y_test_bin,
                                                   y_test=y_test,
                                                   y_pred=y_pred,
                                                   y_prob=y_score,
                                                   name=study_name,
                                                   type_multi='ovo')
                            ])
    df_metrics.to_parquet(f"{config['PREP_DATA']}/df_age_metrics.parquet")

    # Переобучение
    df_overfitting = get_dataset('df_age_overfitting.parquet')
    df_overfitting = pd.concat([df_overfitting,
                                get_f1_score_overfitting(cat_rgs,
                                                         X_train,
                                                         y_train,
                                                         X_test,
                                                         y_test,
                                                         study_name)
                                ])
    df_overfitting.to_parquet(f"{config['PREP_DATA']}/df_age_overfitting.parquet")

    print(f"study_age_2 metrics added {study_name} done and saved")


def study_age_3(study_name: Text):
    """
    Ф-ция обучается на всех данных (без test/train), делает предсказания по нужным сабмитам
    и записывает ответы

    :param study_name: Text
    :return: None
    """

    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    X_train, X_test, y_train, y_test, eval_set, cat_features, X_train_, y_train_ = split_train_test(
        "prep_data_with_targets_age.parquet",
        "age_target")

    # get data
    dataset = get_dataset("prep_data_with_targets_age.parquet")
    cat_features = dataset.select_dtypes('category').columns.tolist()
    X = dataset.drop([config['drop_columns'], 'age_target'], axis=1)
    y = dataset['age_target']
    del dataset

    # Загрузка модели с наилучшими подобранными параметрами
    cat_rgs_all = joblib.load(f"{config['PREP_DATA']}/model_cat_grid_age.pkl")

    # обучение модели на всём датасете
    cat_rgs_all.fit(X,
                    y,
                    cat_features,
                    verbose=False)

    print(f"study_age_3 {study_name} fit")

    # save model
    joblib.dump(cat_rgs_all, f"{config['PREP_DATA']}/model_fin_age.pkl")

    # предсказание возраста по сабмитам
    df_submit = get_dataset('df_submit.parquet')
    fin_submit = df_submit[['user_id']]
    fin_submit['age'] = cat_rgs_all.predict(df_submit.drop(['user_id'], axis=1))

    # save
    fin_submit.to_csv(f"{config['PREP_DATA']}/fin_submit.csv", index=False)
    print(f"study_age_3 {study_name} pred fin submit done and saved")

    # датасеты с метриками моделей
    y = get_dataset("prep_data_with_targets_age.parquet")["age_target"]
    y_test_bin = label_binarize(y_test, classes=list(set(y)))

    y_pred = cat_rgs_all.predict(X_test)
    y_score = cat_rgs_all.predict_proba(X_test)

    # Добавление
    df_metrics = get_dataset('df_age_metrics.parquet')
    df_metrics = pd.concat([df_metrics,
                            get_metrics_multiclass(y_test_bin=y_test_bin,
                                                   y_test=y_test,
                                                   y_pred=y_pred,
                                                   y_prob=y_score,
                                                   name=study_name,
                                                   type_multi='ovo')
                            ])
    df_metrics.to_parquet(f"{config['PREP_DATA']}/df_age_metrics.parquet")

    df_overfitting = get_dataset('df_age_overfitting.parquet')
    df_overfitting = pd.concat([df_overfitting,
                                get_f1_score_overfitting(cat_rgs_all,
                                                         X_train,
                                                         y_train,
                                                         X_test,
                                                         y_test,
                                                         study_name)
                                ])
    df_overfitting.to_parquet(f"{config['PREP_DATA']}/df_age_overfitting.parquet")

    print(f"study_age_3 metrics added {study_name} end")


# Оценка по полу
def study_gender_1(study_name: Text):
    """
    Ф-ция обучает первую бейслайн модель и записывает результаты метрик в датасет.

    :param study_name:
    :return:
    """

    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    X_train, X_test, y_train, y_test, eval_set, cat_features, X_train_, y_train_ = split_train_test(
        "prep_data_with_targets_is_male.parquet",
        "is_male")

    cat_params = {'n_estimators': 1500,
                  'random_state': config['RAND'],
                  'early_stopping_rounds': 100,
                  'cat_features': cat_features
                  }

    model = CatBoostClassifier(**cat_params,
                               task_type=config['task_type'])

    model.fit(X_train,
              y_train,
              eval_set=eval_set,
              verbose=False)

    y_pred = model.predict_proba(X_test)[:, 1]

    print(f"study_gender_1 {study_name} fit")

    # Метрики
    df_metrics = get_metrics_gini(y_test=y_test,
                                  y_pred=y_pred,
                                  name=study_name)
    df_metrics.to_parquet(f"{config['PREP_DATA']}/df_gender_metrics.parquet")

    # Переобучение
    df_overfitting = check_gini_overfitting(model,
                                            X_train,
                                            y_train,
                                            X_test,
                                            y_test,
                                            roc_auc_score,
                                            study_name)
    df_overfitting.to_parquet(f"{config['PREP_DATA']}/df_gender_overfitting.parquet")
    print(f"study_gender_1 metrics added {study_name} done and saved")


def study_gender_2(study_name: Text, key: Text = 'old'):
    """
    Ф-ция выполняет поиск оптимальных параметров если это нужно, записывает модель
    и добавляет новые метрики на новых параметрах.
    :param study_name: Text
    :param key: Text : Отвечает за подбор новых параметров randomized_search.
    :return: None
    """

    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    X_train, X_test, y_train, y_test, eval_set, cat_features, X_train_, y_train_ = split_train_test(
        "prep_data_with_targets_is_male.parquet",
        "is_male")

    if key == 'new':
        # Если требуется подбор параметров
        grid_search_result = func_rgs_gender(X_train, X_test, y_train, y_test, eval_set, cat_features)
    elif key == 'old':
        # пользуемся уже подобранными параметрами rgs
        grid_search_result = joblib.load(f"{config['PREP_DATA']}/grid_search_result_gender.pkl")
    else:
        print('wrong key')

    # тюнинг модели подобранными параметрами
    cat_rgs = CatBoostClassifier(**grid_search_result,
                                 task_type=config['task_type'])

    cat_rgs.fit(X_train_,
                y_train_,
                cat_features=cat_features,
                eval_set=eval_set,
                verbose=False,
                early_stopping_rounds=100)
    y_pred = cat_rgs.predict_proba(X_test)[:, 1]

    # saving model
    joblib.dump(cat_rgs, f"{config['PREP_DATA']}/model_cat_rgs_gender.pkl")
    print(f"study_gender_2 {study_name} fit")

    # датасеты с метриками моделей
    # Добавление метрик
    df_metrics = get_dataset('df_gender_metrics.parquet')
    df_metrics = pd.concat([df_metrics,
                            get_metrics_gini(y_test=y_test,
                                             y_pred=y_pred,
                                             name=study_name)
                            ])
    df_metrics.to_parquet(f"{config['PREP_DATA']}/df_gender_metrics.parquet")

    # Переобучение
    df_overfitting = get_dataset('df_gender_overfitting.parquet')
    df_overfitting = pd.concat([df_overfitting,
                                check_gini_overfitting(cat_rgs,
                                                       X_train,
                                                       y_train,
                                                       X_test,
                                                       y_test,
                                                       roc_auc_score,
                                                       study_name)])
    df_overfitting.to_parquet(f"{config['PREP_DATA']}/df_gender_overfitting.parquet")
    print(f"study_gender_2 metrics added {study_name} done and saved")


def study_gender_3(study_name: Text):
    """
    Ф-ция обучается на всех данных (без test/train), делает предсказания по нужным сабмитам
    и записывает ответы

    :param study_name: Text
    :return: None
    """

    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    X_train, X_test, y_train, y_test, eval_set, cat_features, X_train_, y_train_ = split_train_test(
        "prep_data_with_targets_is_male.parquet",
        "is_male")

    # get data
    dataset = get_dataset("prep_data_with_targets_is_male.parquet")
    cat_features = dataset.select_dtypes('category').columns.tolist()
    X = dataset.drop([config['drop_columns'], 'is_male'], axis=1)
    y = dataset['is_male']
    del dataset

    # Загрузка модели с наилучшими подобранными параметрами
    cat_rgs_all = joblib.load(f"{config['PREP_DATA']}/model_cat_rgs_gender.pkl")

    # обучение модели на всём датасете
    cat_rgs_all.fit(X,
                    y,
                    cat_features,
                    verbose=False)

    y_pred = cat_rgs_all.predict_proba(X_test)[:, 1]

    print(f"study_gender_3 {study_name} fit")

    # save model
    joblib.dump(cat_rgs_all, f"{config['PREP_DATA']}/model_fin_gender.pkl")

    # предсказание гендера по сабмитам
    df_submit = get_dataset('df_submit.parquet')
    fin_submit = pd.read_csv(f"{config['PREP_DATA']}/fin_submit.csv")
    fin_submit['is_male'] = cat_rgs_all.predict_proba(df_submit.drop(['user_id'], axis=1))[:, 1]

    # save
    fin_submit.to_csv(f"{config['PREP_DATA']}/fin_submit_age.csv", index=False)
    print(f"study_gender_3 {study_name} pred submit done")

    # датасеты с метриками моделей
    # Добавление метрик
    df_metrics = get_dataset('df_gender_metrics.parquet')
    df_metrics = pd.concat([df_metrics,
                            get_metrics_gini(y_test=y_test,
                                             y_pred=y_pred,
                                             name=study_name)
                            ])
    df_metrics.to_parquet(f"{config['PREP_DATA']}/df_gender_metrics.parquet")

    # Переобучение
    df_overfitting = get_dataset('df_gender_overfitting.parquet')
    df_overfitting = pd.concat([df_overfitting, check_gini_overfitting(cat_rgs_all,
                                                                       X_train,
                                                                       y_train,
                                                                       X_test,
                                                                       y_test,
                                                                       roc_auc_score,
                                                                       study_name)
                                ])
    df_overfitting.to_parquet(f"{config['PREP_DATA']}/df_gender_overfitting.parquet")
    print(f"study_gender_3 metrics added {study_name} end")


def random_predict():
    """
    Ф-ция делает случайное предсказание пола и возраста на основе обученных ранее моделей пола и возраста.
    :return: DataFrame
    """
    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    cat_rgs_age = joblib.load(f"{config['PREP_DATA']}/model_cat_grid_age.pkl")
    cat_rgs_gendr = joblib.load(f"{config['PREP_DATA']}/model_fin_gender.pkl")

    df = rand_data_pred()
    df['is_male'] = cat_rgs_gendr.predict_proba(df)[:, 1]
    df['age'] = cat_rgs_age.predict(df)
    df[['cpe_manufacturer_name',
        'cpe_model_os_type',
        'is_male',
        'age']].to_parquet(f"{config['PREP_DATA']}/random_pred_df.parquet")
    # return df[['cpe_manufacturer_name', 'cpe_model_os_type', 'is_male', 'age']].to_dict()


def training_all_models():
    """
    Обучение всех моделей в 1 ф-ции
    """
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    key_rgs = config['key_rgs']
    # key1, key2 = key_rgs, key_rgs

    study_age_1("cat_base_age")
    study_age_2("cat_rgs_age", key_rgs)
    study_age_3("cat_rgs_all_age")
    study_gender_1("cat_base_gender")
    study_gender_2("cat_rgs_gender", key_rgs)
    study_gender_3("cat_rgs_all_gender")
