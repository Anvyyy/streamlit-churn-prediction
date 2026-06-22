import pandas as pd
import streamlit as st


def render_models() -> None:
    st.subheader("Метрики моделей")
    st.write(
        "Все модели оценивались на одной тестовой выборке. "
        "Гиперпараметры и threshold выбирались на validation."
    )

    model_results = pd.DataFrame([
        {
            "Модель": "CatBoost",
            "Threshold": 0.60,
            "Accuracy": 0.941,
            "Precision": 0.723,
            "Recall": 0.826,
            "F1": 0.771,
            "ROC-AUC": 0.978,
            "PR-AUC": 0.853,
        },
        {
            "Модель": "XGBoost",
            "Threshold": 0.60,
            "Accuracy": 0.929,
            "Precision": 0.666,
            "Recall": 0.827,
            "F1": 0.738,
            "ROC-AUC": 0.971,
            "PR-AUC": 0.794,
        },
        {
            "Модель": "Logistic Regression",
            "Threshold": 0.75,
            "Accuracy": 0.911,
            "Precision": 0.604,
            "Recall": 0.746,
            "F1": 0.667,
            "ROC-AUC": 0.949,
            "PR-AUC": 0.660,
        },
        {
            "Модель": "Наивный baseline",
            "Threshold": 0.50,
            "Accuracy": 0.880,
            "Precision": 0.000,
            "Recall": 0.000,
            "F1": 0.000,
            "ROC-AUC": 0.500,
            "PR-AUC": 0.120,
        },
    ])

    st.subheader("Лучшая модель: CatBoost")

    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = (
        st.columns(5)
    )

    with metric_col1:
        st.metric(
            "Precision",
            "0.723",
            "+0.057 к XGBoost",
        )

    with metric_col2:
        st.metric(
            "Recall",
            "0.826",
            "-0.001 к XGBoost",
        )

    with metric_col3:
        st.metric(
            "F1",
            "0.771",
            "+0.033 к XGBoost",
        )

    with metric_col4:
        st.metric(
            "ROC-AUC",
            "0.978",
            "+0.007 к XGBoost",
        )

    with metric_col5:
        st.metric(
            "PR-AUC",
            "0.853",
            "+0.059 к XGBoost",
        )

    st.success(
        """
        Для итогового прогнозирования выбрана модель **CatBoost**.

        Она показала лучший результат по F1 и PR-AUC, сохранив практически
        такой же recall, как XGBoost, но заметно повысив precision.
        Это означает, что CatBoost находит примерно столько же реальных
        уходов, но делает меньше ложных предупреждений.
        """
    )


    st.subheader("Итоговые метрики на test")

    st.dataframe(
        model_results,
        hide_index=True,
        width="stretch",
        column_config={
            "Модель": st.column_config.TextColumn(
                "Модель",
                width="large",
            ),
            "Threshold": st.column_config.NumberColumn(
                "Threshold",
                format="%.2f",
            ),
            "Accuracy": st.column_config.ProgressColumn(
                "Accuracy",
                format="%.3f",
                min_value=0,
                max_value=1,
            ),
            "Precision": st.column_config.ProgressColumn(
                "Precision",
                format="%.3f",
                min_value=0,
                max_value=1,
            ),
            "Recall": st.column_config.ProgressColumn(
                "Recall",
                format="%.3f",
                min_value=0,
                max_value=1,
            ),
            "F1": st.column_config.ProgressColumn(
                "F1",
                format="%.3f",
                min_value=0,
                max_value=1,
            ),
            "ROC-AUC": st.column_config.ProgressColumn(
                "ROC-AUC",
                format="%.3f",
                min_value=0,
                max_value=1,
            ),
            "PR-AUC": st.column_config.ProgressColumn(
                "PR-AUC",
                format="%.3f",
                min_value=0,
                max_value=1,
            ),
        },
    )


    with st.expander("Почему для сравнения важен PR-AUC"):
        st.write(
            """
            В датасете присутствует дисбаланс классов: только около 12% клиентов относятся к классу
            оттока. Поэтому одной accuracy недостаточно: модель может
            получить высокую accuracy, почти всегда предсказывая класс
            «остался».

            PR-AUC оценивает, насколько хорошо модель находит редкий
            класс `churn=1` и сохраняет точность таких прогнозов.
            CatBoost получила самое высокое значение PR-AUC: 0.853.
            """
        )

    with st.expander("Конфигурация выбранной модели"):
        st.write(
            """
            - Модель: CatBoostClassifier
            - Количество признаков: 40
            - Threshold: 0.60
            - Количество деревьев: 950
            - Глубина деревьев: 5
            - Learning rate: 0.0632
            - Подбор параметров: Optuna
            - Балансировка классов: SqrtBalanced
            """
        )


