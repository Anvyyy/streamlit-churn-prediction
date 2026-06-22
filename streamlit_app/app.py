from pathlib import Path
import sys

import pandas as pd
import streamlit as st


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import APP_TITLE, DATA_PATH
from views.analytics import render_analytics
from views.models import render_models
from views.overview import render_overview
from views.prediction import render_prediction


st.set_page_config(
    page_title="churn_predict",
    page_icon=":bar_chart:",
    layout="wide",
)


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


df = load_data(DATA_PATH)


with st.sidebar:
    st.title("Настройки")

    prediction_mode = st.radio(
        "Режим прогноза",
        ["Исторический клиент", "Ручной сценарий"],
    )

    threshold = st.slider(
        "Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.6,
        step=0.05,
    )


st.title(APP_TITLE)
st.subheader("Информация о проекте:")
st.write(
    "Цель проекта, на основе данных об отnоке пользователях из букмекерской "
    "конторы, построить модель предсказывающую отток клиента, для возможности "
    "его удержания"
)

tab_overview, tab_analytics, tab_models, tab_prediction = st.tabs([
    "Обзор данных",
    "Анализ датасета",
    "Сравнение моделей",
    "Прогноз клиента",
])

with tab_overview:
    render_overview()

with tab_analytics:
    render_analytics(df)

with tab_models:
    render_models()

with tab_prediction:
    render_prediction(df, prediction_mode, threshold)
