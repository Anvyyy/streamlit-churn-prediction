import streamlit as st

from services.model_service import (
    build_manual_features,
    load_input_template,
    predict_probability,
)


def render_prediction(df, prediction_mode: str, threshold: float) -> None:
    st.subheader("Индивидуальный прогноз")

    if prediction_mode == "Ручной сценарий":
        _render_manual_scenario(threshold)
        return

    _render_historical_client(df, threshold)


def _render_historical_client(df, threshold: float) -> None:

    client_ids = (
        df["CupisId"]
        .dropna()
        .astype(int)
        .sort_values()
        .tolist()
    )

    client_id = st.selectbox(
        "Выберите CupisId клиента",
        options=client_ids,
        index=None,
        placeholder="Начните вводить ID клиента",
    )

    if st.button("Получить прогноз", key="historical_prediction"):
        if client_id is None:
            st.warning("Сначала выберите клиента.")
            return

        client = df.loc[df["CupisId"] == client_id]

        probability = predict_probability(client)
        actual_churn = int(client["churn"].iloc[0])

        _render_result(
            probability,
            threshold,
            actual_churn=actual_churn,
        )


def _render_manual_scenario(threshold: float) -> None:
    st.write(
        "Измените основные показатели клиента. Остальные признаки "
        "будут заполнены типичными значениями из train."
    )

    defaults = load_input_template().iloc[0]

    with st.form("manual_prediction_form"):
        left, right = st.columns(2)

        with left:
            count_bets_30 = st.number_input(
                "Ставок за 30 дней",
                min_value=0,
                value=int(defaults["CountBets30Days"]),
                step=1,
            )
            count_bets_60 = st.number_input(
                "Ставок за 60 дней",
                min_value=0,
                value=int(defaults["CountBets60Days"]),
                step=1,
            )
            deposit_count = st.number_input(
                "Количество депозитов",
                min_value=0,
                value=int(defaults["DepositCount"]),
                step=1,
            )
            q75_dep_date_diff = st.number_input(
                "Пауза между депозитами, дней",
                min_value=0.0,
                value=float(defaults["Q75DepDateDiff"]),
                step=0.5,
            )
            activity_per_dep = st.number_input(
                "Доля активных дней",
                min_value=0.0,
                max_value=1.0,
                value=float(defaults["ActivityPerDep"]),
                step=0.01,
            )

        with right:
            bet_amount_30 = st.number_input(
                "Сумма ставок за 30 дней",
                min_value=0.0,
                value=float(defaults["BetAmountSumLast30"]),
                step=100.0,
            )
            ggr_30 = st.number_input(
                "GGR за 30 дней",
                value=float(defaults["GGR30Days"]),
                step=100.0,
            )
            ltv = st.number_input(
                "LTV",
                min_value=0.0,
                value=float(defaults["LTV"]),
                step=100.0,
            )
            max_deposit = st.number_input(
                "Максимальный депозит",
                min_value=0.0,
                value=float(defaults["DepositAmountPrevMax"]),
                step=100.0,
            )

        submitted = st.form_submit_button("Рассчитать риск")

    if not submitted:
        return

    if count_bets_60 < count_bets_30:
        st.error(
            "Количество ставок за 60 дней не может быть меньше "
            "количества ставок за 30 дней."
        )
        return

    manual_values = {
        "CountBets30Days": count_bets_30,
        "CountBets60Days": count_bets_60,
        "DepositCount": deposit_count,
        "Q75DepDateDiff": q75_dep_date_diff,
        "ActivityPerDep": activity_per_dep,
        "BetAmountSumLast30": bet_amount_30,
        "GGR30Days": ggr_30,
        "LTV": ltv,
        "DepositAmountPrevMax": max_deposit,
    }

    features = build_manual_features(manual_values)
    probability = predict_probability(features)

    _render_result(probability, threshold)

    with st.expander("Посмотреть подготовленные признаки"):
        st.dataframe(features, hide_index=True, width="stretch")


def _render_result(
    probability: float,
    threshold: float,
    actual_churn: int | None = None,
) -> None:
    prediction = int(probability >= threshold)

    st.metric("Вероятность оттока", f"{probability:.2%}")
    st.progress(probability)

    if prediction == 1:
        st.error("Прогноз модели: клиент уйдёт.")
    else:
        st.success("Прогноз модели: клиент останется.")

    if actual_churn is not None:
        actual_label = (
            "клиент ушёл"
            if actual_churn == 1
            else "клиент остался"
        )
        st.write(f"Фактический статус: **{actual_label}**")

    st.caption(f"Использованный threshold: {threshold:.2f}")
