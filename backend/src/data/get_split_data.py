import pandas as pd
import yaml

from typing import Text
from sklearn.model_selection import train_test_split

RAND = 10

config_path = '../config/params.yml'
# config = yaml.load(open(config_path), Loader=yaml.FullLoader)


def get_dataset(dataset_path: Text) -> pd.DataFrame:
    """
    Получение данных по заданному пути
    :param dataset_path: путь до данных
    :return: датасет
    """
    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return pd.read_parquet(f"{config['PREP_DATA']}/{dataset_path}")
    # return pd.read_parquet(f"{config['LOCAL_DATA_PATH']}/{config['PREP_DATA']}/{dataset_path}")


# Split in train/test
def split_train_test(dataset_path: Text, target: Text):  # dataset: pd.DataFrame,
    """
    Разделение данных на train/test
    :param target: название таргета
    :param dataset_path: путь к датасету
    :return: train/test датасеты
    """

    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # get data
    dataset = get_dataset(dataset_path)
    cat_features = dataset.select_dtypes('category').columns.tolist()

    X = dataset.drop([config['drop_columns'], target], axis=1)
    y = dataset[target]

    # Тестовые
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=config['RAND']
    )

    # Валидационные
    X_train_, X_val, y_train_, y_val = train_test_split(X_train,
                                                        y_train,
                                                        test_size=0.15,
                                                        random_state=config['RAND'])

    eval_set = [(X_val, y_val)]

    print(f"split_train_test {target} done")

    return X_train, X_test, y_train, y_test, eval_set, cat_features, X_train_, y_train_
