import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st

from data_dictionary import variables


@st.cache_data
def calculate_spearman_correlation(data):
    return data.corr(method="spearman")


def render_analytics(df) -> None:
    st.subheader("Анализ оттока")

    st.write(
        "Краткий обзор клиентской базы и признаков, "
        "связанных с оттоком."
    )

    client_count = df["CupisId"].nunique()
    churn_count = int(df["churn"].sum())
    churn_rate = df["churn"].mean()
    feature_count = df.shape[1]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Количество клиентов",
            f"{client_count:,}".replace(",", " "),
        )

    with col2:
        st.metric(
            "Клиентов в оттоке",
            f"{churn_count:,}".replace(",", " "),
        )

    with col3:
        st.metric(
            "Доля оттока",
            f"{churn_rate:.2%}",
        )

    with col4:
        st.metric(
            "Количество признаков",
            feature_count,
        )
    

    profile_features = [
    "CountBets30Days",
    "DepositCount",
    "Q75DepDateDiff",
    "BetAmountSumLast30",
    "GGR30Days",
    "RecencyLastDeposit",
    "DepositAmountPrevMax",
    "ActivityPerDep",
    ]

    feature_labels = {
    "CountBets30Days": "Ставки за 30 дней",
    "DepositCount": "Количество депозитов",
    "Q75DepDateDiff": "Пауза между депозитами",
    "BetAmountSumLast30": "Сумма ставок за 30 дней",
    "GGR30Days": "GGR за 30 дней",
    "LTV": "LTV",
    "RecencyLastDeposit": "Дней с последнего депозита",
    "DepositAmountPrevMax": "Максимальный депозит",
    "ActivityPerDep": "Активность после депозита",
}

    profile_df = (df.groupby('churn')[profile_features].median().T)

    profile_df = profile_df.rename(columns={
    0: "Остался",
    1: "Ушёл",
    }).reset_index().rename(columns={"index": "Показатель"})

    profile_df.insert(
    1,
    "Описание",
    profile_df["Показатель"].map(variables),
    )

    st.dataframe(
    profile_df,
    width="stretch",
    )



    corr_features = profile_features + ["churn"]

    corr_matrix = calculate_spearman_correlation(
        df[corr_features]
    )

    corr_with_churn = (
        corr_matrix["churn"]
        .drop("churn")
        .sort_values()
    )


    left_chart, right_chart = st.columns(2)

    with left_chart:
        st.subheader("Распределение клиентов")

        churn_counts = (
            df["churn"]
            .value_counts()
            .reindex([0, 1], fill_value=0)
        )

        fig, ax = plt.subplots(figsize=(7, 5))

        bars = ax.bar(
            ["Остался", "Ушёл"],
            churn_counts.values,
            color=["#3B82F6", "#EF4444"],
        )

        ax.set_title("Количество клиентов по статусу")
        ax.set_ylabel("Количество клиентов")
        ax.set_xlabel("Статус клиента")

        for bar, value in zip(bars, churn_counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:,}".replace(",", " "),
                ha="center",
                va="bottom",
            )

        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Класс оттока несбалансирован: ушедшие клиенты "
            "составляют около 12% выборки."
        )

    with right_chart:
        st.subheader("Связь признаков с оттоком")

        correlation_colors = [
            "#EF4444" if value > 0 else "#3B82F6"
            for value in corr_with_churn.values
        ]

        correlation_labels = [
            feature_labels.get(feature, feature)
            for feature in corr_with_churn.index
        ]

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.barh(
            correlation_labels,
            corr_with_churn.values,
            color=correlation_colors,
        )

        ax.axvline(
            0,
            color="black",
            linewidth=0.8,
        )

        ax.set_title("Корреляция Спирмена с churn")
        ax.set_xlabel("Коэффициент корреляции")

        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Красный цвет означает положительную связь с оттоком, "
            "синий цвет означает отрицательную связь."
        )

    with st.expander("Посмотреть распределение отдельного признака"):
        selected_label = st.selectbox(
            "Выберите показатель",
            list(feature_labels.values()),
            key="analytics_feature",
        )

        label_to_feature = {
            label: feature
            for feature, label in feature_labels.items()
        }

        selected_feature = label_to_feature[selected_label]

        plot_df = (
            df[["churn", selected_feature]]
            .dropna()
            .copy()
        )

        plot_df["Статус"] = plot_df["churn"].map({
            0: "Остался",
            1: "Ушёл",
        })

        fig, ax = plt.subplots(figsize=(9, 5))

        sns.boxplot(
            data=plot_df,
            x="Статус",
            y=selected_feature,
            order=["Остался", "Ушёл"],
            hue="Статус",
            palette={
                "Остался": "#3B82F6",
                "Ушёл": "#EF4444",
            },
            legend=False,
            showfliers=False,
            ax=ax,
        )

        ax.set_title(
            f"{selected_label} по статусу клиента"
        )
        ax.set_xlabel("Статус клиента")
        ax.set_ylabel(selected_label)

        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Выбросы скрыты только на графике, но не удалены "
            "из исходного датасета."
        )


    with st.expander("Посмотреть матрицу корреляций"):
        heatmap_labels = {
            **feature_labels,
            "churn": "Отток",
        }

        heatmap_matrix = corr_matrix.rename(
            index=heatmap_labels,
            columns=heatmap_labels,
        )

        mask = np.triu(
            np.ones_like(
                heatmap_matrix,
                dtype=bool,
            ),
            k=1,
        )

        fig, ax = plt.subplots(figsize=(12, 9))

        sns.heatmap(
            heatmap_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={
                "label": "Корреляция Спирмена",
                "shrink": 0.8,
            },
            ax=ax,
        )

        ax.set_title(
            "Матрица корреляций ключевых признаков"
        )

        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Корреляция показывает статистическую связь, "
            "но сама по себе не доказывает причинно-следственную связь."
        )

    with st.expander("Посмотреть описательные статистики"):
        descriptive_stats = (
            df[profile_features]
            .describe()
            .T
            .reset_index()
            .rename(columns={
                "index": "Переменная",
                "count": "Количество",
                "mean": "Среднее",
                "std": "Стандартное отклонение",
                "min": "Минимум",
                "25%": "25%",
                "50%": "Медиана",
                "75%": "75%",
                "max": "Максимум",
            })
        )

        descriptive_stats.insert(
            1,
            "Описание",
            descriptive_stats["Переменная"].map(variables),
        )

        numeric_columns = descriptive_stats.select_dtypes(
            include="number"
        ).columns

        descriptive_stats[numeric_columns] = (
            descriptive_stats[numeric_columns]
            .round(2)
        )

        st.dataframe(
            descriptive_stats,
            hide_index=True,
            width="stretch",
        )


    with st.expander("Посмотреть качество данных"):
        total_missing = int(df.isna().sum().sum())

        columns_with_missing = int(
            (df.isna().sum() > 0).sum()
        )

        missing_share = (
            total_missing
            / df.size
        )

        duplicate_client_ids = int(
            df["CupisId"].duplicated().sum()
        )

        quality_col1, quality_col2, quality_col3 = st.columns(3)

        with quality_col1:
            st.metric(
                "Всего пропусков",
                f"{total_missing:,}".replace(",", " "),
            )

        with quality_col2:
            st.metric(
                "Столбцов с пропусками",
                columns_with_missing,
            )

        with quality_col3:
            st.metric(
                "Повторяющихся CupisId",
                duplicate_client_ids,
            )

        st.caption(
            f"Доля пропущенных ячеек: {missing_share:.3%}"
        )

        missing_values = (
            df.isna()
            .sum()
            .sort_values(ascending=False)
        )

        missing_values = missing_values[
            missing_values > 0
        ]

        missing_df = (
            missing_values
            .rename("Количество пропусков")
            .reset_index()
            .rename(columns={
                "index": "Переменная",
            })
        )

        missing_df.insert(
            1,
            "Описание",
            missing_df["Переменная"].map(variables),
        )

        missing_df["Доля пропусков, %"] = (
            missing_df["Количество пропусков"]
            / len(df)
            * 100
        ).round(3)

        st.dataframe(
            missing_df,
            hide_index=True,
            width="stretch",
        )

