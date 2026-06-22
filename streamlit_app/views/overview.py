import pandas as pd
import streamlit as st

from data_dictionary import main_variables, variables


def render_overview() -> None:
    st.subheader("Обзор данных")
    st.write("""
    Датасет представляет собой данные по финансовым операциям игроков в БК.
    В нём содержатся такие переменные, как количество ставок, депозитов и выводов.

    Опираясь на эту информацию, мы попытаемся спрогнозировать, уйдёт ли клиент в отток.

    Ниже представлен обзор всех переменных.
    """)

    st.write("Датасет представляет собой клиентский срез, сформированный осенью 2025 года. Все признаки отражают информацию, зафиксированную на момент формирования среза.")

  
    variables_df = pd.DataFrame(
        variables.items(),
        columns=["Переменная", "Описание"],
    )

    main_variables_df = pd.DataFrame(
        main_variables.items(),
        columns=["Переменная", "Описание"],
    )

    with st.expander("Основные переменные датасета"):
        st.table(main_variables)

    with st.expander("Полное описание переменных"):
        st.table(variables_df)


