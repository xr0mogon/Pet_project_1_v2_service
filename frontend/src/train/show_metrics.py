import requests
import streamlit as st
import sklearn

from ..data.get_split_data import get_dataset


def show_score(key1=0, key2=1, show='n'):
    # загрузка метрик
    metrics_age = (get_dataset('df_age_metrics.parquet')).reset_index(drop=True)
    metrics_gender = get_dataset('df_gender_metrics.parquet').reset_index(drop=True)

    # показ метрик
    if show == 'y':
        st.markdown("**Метрики всех моделей**")
        st.write(metrics_age, metrics_gender)

    old_f1 = metrics_age['F1_weighted'][key1]
    new_f1 = metrics_age['F1_weighted'][key2]
    old_g1 = metrics_gender['Gini'][key1]
    new_g2 = metrics_gender['Gini'][key2]

    st.markdown("**Метрики и Score**")
    # отрисовка f1, gini и score
    f1_, gini_, score_ = st.columns(3)
    f1_.metric(
        "F1_weighted",
        new_f1,
        f"{new_f1 - old_f1:.4f}"
    )
    gini_.metric(
        "Gini",
        new_g2,
        f"{new_g2 - old_g1:.4f}"
    )
    score_.metric(
        "Score",
        f"{(2 * new_f1 + new_g2):.4f}",
        f"{((2 * new_f1 + new_g2) - (2 * old_f1 + old_g1)):.4f}"
    )
