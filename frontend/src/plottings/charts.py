"""
Программа: Отрисовка графиков. EDA parts.
Версия: 1.0
"""

import pandas as pd
import gc

import matplotlib.pyplot as plt
import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

LOCAL_DATA_PATH = "../data"
DATA_FILE = 'competition_data_final_pqt'

# целевые переменные
TARGET_FILE_AGE = 'targets_age_prep.parquet'
TARGET_FILE_MALE = 'targets_is_male_prep.parquet'

# id, по которым нужно предсказать пол и возраст
SUBMIT_FILE = 'submit_2.pqt'

# папка, куда будут сохраняться предобработанные данные
PREP_DATA = 'preprocessed_data'

LOCAL_DATA_PATH_mts = "../data/ml_cup_data"


def plot_text(ax: plt.Axes):
    """
    Вывод процентов на графике barplot
    """
    for p in ax.patches:
        percentage = '{:.1f}%'.format(p.get_height())
        ax.annotate(
            percentage,  # текст
            # координата xy
            (p.get_x() + p.get_width() / 2., p.get_height()),
            # центрирование
            ha='center',
            va='center',
            xytext=(0, 10),
            textcoords='offset points',
            fontsize=14)


def cat_ages_bars():
    """
    2.1.1. Целевой признак по возрасту.
    """
    targets = pd.read_parquet(f'{LOCAL_DATA_PATH}/{PREP_DATA}/{TARGET_FILE_AGE}')

    # нормирование
    norm_target = (targets
                   .age_target
                   .value_counts(normalize=True)
                   .mul(100)
                   .rename('percent')
                   .reset_index())

    plt.figure(figsize=(15, 7))
    # ax = sns.barplot(x='index', y='percent', data=norm_target)
    ax = sns.barplot(x='age_target', y='percent', data=norm_target)

    plot_text(ax)

    plt.title('Барплот категоризированного возраста', fontsize=20)
    plt.xlabel('Классы возраста \n1 (19-25), 2 (26-35), 3 (36-45), 4 (46-55), 5 (56-65), 6(66+)', fontsize=14)
    plt.ylabel('Проценты', fontsize=14)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    del targets, norm_target
    gc.collect()


def gender_bars():
    """
    2.1.2. Целевой признак по полу.
    """
    targets = pd.read_parquet(f'{LOCAL_DATA_PATH}/{PREP_DATA}/{TARGET_FILE_MALE}')

    # нормирование
    norm_target = (targets
                   .is_male
                   .value_counts(normalize=True)
                   .mul(100)
                   .rename('percent')
                   .reset_index())

    plt.figure(figsize=(10, 7))
    # ax = sns.barplot(x='index', y='percent', data=norm_target)
    ax = sns.barplot(x='is_male', y='percent', data=norm_target)

    plot_text(ax)

    plt.title('Барплот новых возрастных классов', fontsize=20)
    plt.xlabel('Классы возраста', fontsize=14)
    plt.ylabel('Проценты', fontsize=14)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    del targets, norm_target
    gc.collect()


def url_count_gender_bars():
    """
    2.2.1. График гипотезы: гендер или возраст влияет на кол-во переходов по ссылкам.
    """
    # load
    data = pd.read_parquet(f"{LOCAL_DATA_PATH}/{PREP_DATA}/prep_data_with_2_targets.parquet")
    # новый признак подсчета переходов по ссылкам пользователя
    data['url_count'] = data.loc[:, 'day': 'night'].sum(axis=1)

    # График
    fig = plt.subplots(figsize=(15, 5))

    sns.barplot(x='url_count',
                y='age_target',
                data=data,
                hue='is_male',
                palette=["#ff796c", "#0485d1"],
                orient='h').set(title='Переходы по ссылкам относительно возраста в разрезе пола относительно возраста')

    del data, fig
    gc.collect()


def price_gender_bars():
    """
    2.2.2. График гипотезы: стоимость ус-ва влияет на пол или возраст.
    :return: барплот
    """
    # load
    data = pd.read_parquet(f"{LOCAL_DATA_PATH}/{PREP_DATA}/prep_data_with_2_targets.parquet")

    # График
    fig = plt.subplots(figsize=(15, 5))

    sns.barplot(x='age_target',
                y='price',
                data=data,
                hue='is_male',
                palette='magma_r'
                ).set(title='Влияние стоимости ус-ва на возраст в разрезе пола')

    del data, fig
    gc.collect()


def price_gender_boxplot():
    """
    2.2.2. График гипотезы: стоимость ус-ва влияет на пол или возраст.
    :return: boxplot
    """
    # load
    data = pd.read_parquet(f"{LOCAL_DATA_PATH}/{PREP_DATA}/prep_data_with_2_targets.parquet")

    # График
    fig = plt.subplots(figsize=(15, 10))
    sns.boxplot(data=data,
                x='age_target',
                y='price',
                hue='is_male',
                palette='rocket')

    del data, fig
    gc.collect()


def manufacturer_gender_bars():
    """
    4.2.3. Гипотеза: производитель ус-ва влияет на пол.
    :return: barplot пол
    """
    # load
    data = pd.read_parquet(f"{LOCAL_DATA_PATH}/{PREP_DATA}/prep_data_with_2_targets.parquet")
    data = data[['user_id',
                 'cpe_manufacturer_name',
                 'cpe_model_os_type',
                 'is_male', 'age_target']]

    # нормирование по гендеру
    norm_target = (data[['cpe_manufacturer_name', 'is_male']]
                   .value_counts(normalize=True)
                   .mul(100)
                   .rename('percent')
                   .reset_index())

    # фильтр незначительных элементов
    norm_target = norm_target.loc[norm_target['percent'] > 1].reset_index()
    norm_target['cpe_manufacturer_name'] = norm_target.cpe_manufacturer_name.astype('O')

    # График
    plt.figure(figsize=(15, 7))

    ax = sns.barplot(x='percent',
                     y='cpe_manufacturer_name',
                     data=norm_target,
                     hue='is_male',
                     orient='h',
                     palette='magma_r')

    for p in ax.patches:
        width = p.get_width()
        plt.text(0.5 + p.get_width(), p.get_y() + 0.5 * p.get_height(),
                 '{:1.2f}'.format(width),
                 ha='center', va='center')

    plt.title('Влияет ли компания производитель на пол?', fontsize=20)
    plt.xlabel('Проценты', fontsize=14)
    plt.ylabel('Производители', fontsize=14)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.show()

    del data, norm_target, ax
    gc.collect()


def manufacturer_age_bars():
    """
    4.2.3. Гипотеза: производитель ус-ва влияет на возраст.
    :return: barplot возраст
    """
    # load
    data = pd.read_parquet(f"{LOCAL_DATA_PATH}/{PREP_DATA}/prep_data_with_2_targets.parquet")
    data = data[['user_id',
                 'cpe_manufacturer_name',
                 'cpe_model_os_type',
                 'is_male', 'age_target']]

    # нормирование по возрастным классам
    norm_target = (data[['cpe_manufacturer_name', 'age_target']]
                   .value_counts(normalize=True)
                   .mul(100)
                   .rename('percent')
                   .reset_index())

    # фильтр незначительных элементов
    norm_target = norm_target.loc[norm_target['percent'] > 1].reset_index()
    norm_target['cpe_manufacturer_name'] = norm_target.cpe_manufacturer_name.astype('O')

    # График
    plt.figure(figsize=(15, 7))

    ax = sns.barplot(x='percent',
                     y='cpe_manufacturer_name',
                     data=norm_target,
                     hue='age_target',
                     orient='h',
                     palette='magma_r')

    for p in ax.patches:
        width = p.get_width()
        plt.text(0.5 + p.get_width(), p.get_y() + 0.5 * p.get_height(),
                 '{:1.2f}'.format(width),
                 ha='center', va='center')

    plt.title('Влияет ли компания производитель на возраст?', fontsize=20)
    plt.xlabel('Проценты', fontsize=14)
    plt.ylabel('Производители', fontsize=14)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.show()

    del data, norm_target, ax
    gc.collect()


def os_age_bars():
    """
    4.2.4. Гипотеза: Операционная система влияет на возраст.
    :return: barplot
    """
    # load
    data = pd.read_parquet(f"{LOCAL_DATA_PATH}/{PREP_DATA}/prep_data_with_2_targets.parquet")
    data = data[['user_id',
                 'cpe_manufacturer_name',
                 'cpe_model_os_type',
                 'is_male', 'age_target']]

    # нормирование по возрастным классам
    norm_target = (data[['cpe_model_os_type', 'age_target']]
                   .value_counts(normalize=True)
                   .mul(100)
                   .rename('percent')
                   .reset_index())

    # График
    plt.figure(figsize=(15, 7))

    ax = sns.barplot(x='percent',
                     y='cpe_model_os_type',
                     data=norm_target,
                     hue='age_target',
                     orient='h')

    # Подписи процентов к барам
    for p in ax.patches:
        width = p.get_width()
        plt.text(0.5 + p.get_width(), p.get_y() + 0.5 * p.get_height(),
                 '{:1.2f}'.format(width),
                 ha='center', va='center')

    plt.title('Зависимость ОС и возраста', fontsize=20)
    plt.xlabel('Проценты', fontsize=14)
    plt.ylabel('Производители', fontsize=14)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.show()

    del data, norm_target, ax
    gc.collect()


def os_gender_bars():
    """
    4.2.4. Гипотеза: Операционная система влияет на пол.
    :return: barplot
    """
    # load
    data = pd.read_parquet(f"{LOCAL_DATA_PATH}/{PREP_DATA}/prep_data_with_2_targets.parquet")

    # нормирование
    norm_target = (data[['cpe_model_os_type', 'is_male']]
                   .value_counts(normalize=True)
                   .mul(100)
                   .rename('percent')
                   .reset_index())

    # График
    plt.figure(figsize=(15, 6))

    ax = sns.barplot(x='percent',
                     y='cpe_model_os_type',
                     data=norm_target,
                     hue='is_male',
                     orient='h',
                     palette='magma_r')

    # Подписи процентов к барам
    for p in ax.patches:
        width = p.get_width()
        plt.text(1 + p.get_width(), p.get_y() + 0.1 * p.get_height(),
                 '{:1.2f}'.format(width),
                 ha='center', va='center')

    plt.title('Зависимость гендера и OC', fontsize=20)
    plt.xlabel('Проценты', fontsize=14)
    plt.ylabel('OC', fontsize=14)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.show()

    del data, norm_target, ax
    gc.collect()
