from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier

from config import INPUT_TEMPLATE_PATH, MODEL_PATH


@st.cache_resource
def load_model(path: Path = MODEL_PATH) -> CatBoostClassifier:
    model = CatBoostClassifier()
    model.load_model(str(path))

    return model


@st.cache_data
def load_input_template(
    path: Path = INPUT_TEMPLATE_PATH,
) -> pd.DataFrame:
    return pd.read_csv(path)


def build_manual_features(values: dict[str, float]) -> pd.DataFrame:
    features = load_input_template().copy(deep=True)

    unknown_features = set(values) - set(features.columns)
    if unknown_features:
        raise ValueError(
            f"Неизвестные признаки: {sorted(unknown_features)}"
        )

    for feature, value in values.items():
        features.at[0, feature] = value

    features.at[0, "LTV_log"] = np.log1p(
        max(float(features.at[0, "LTV"]), 0.0)
    )
    features.at[0, "DepositCount_log"] = np.log1p(
        max(float(features.at[0, "DepositCount"]), 0.0)
    )
    features.at[0, "CountBets30Days_log"] = np.log1p(
        max(float(features.at[0, "CountBets30Days"]), 0.0)
    )
    features.at[0, "BetAmountSumLast30_log"] = np.log1p(
        max(float(features.at[0, "BetAmountSumLast30"]), 0.0)
    )

    ggr_30_days = float(features.at[0, "GGR30Days"])
    features.at[0, "GGR30Days_log"] = np.log1p(
        max(ggr_30_days, -0.999)
    )

    count_bets_30_days = float(features.at[0, "CountBets30Days"])
    features.at[0, "BetsSegment"] = _bets_segment(
        count_bets_30_days
    )

    return features


def _bets_segment(count_bets_30_days: float) -> str:
    if count_bets_30_days <= 0:
        return "Missing"
    if count_bets_30_days <= 17:
        return "0-17"
    if count_bets_30_days <= 47:
        return "17-47"
    if count_bets_30_days <= 127:
        return "47-127"
    if count_bets_30_days <= 297:
        return "127-297"
    return "297+"


def predict_probability(features: pd.DataFrame) -> float:
    model = load_model()

    ordered_features = features.loc[:, model.feature_names_].copy()

    for index in model.get_cat_feature_indices():
        feature = model.feature_names_[index]
        ordered_features[feature] = (
            ordered_features[feature]
            .astype("string")
            .fillna("Missing")
            .astype(str)
        )

    probabilities = model.predict_proba(ordered_features)

    return float(probabilities[0, 1])
