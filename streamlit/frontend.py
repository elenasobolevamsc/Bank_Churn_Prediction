import streamlit as st
import pandas as pd
import requests
from io import BytesIO

uploaded_file = st.file_uploader("Choose your csv or xlsx file", type=["csv", "xlsx"])
st.sidebar.header("menu")
if st.sidebar.button('predict'):
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)

        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)

        st.dataframe(df)

        data = df.to_json(orient='split')
        payload = {'data': data}
        result = requests.post('http://api:8000/best_model', json=payload)
        res = pd.read_json(result.json()['pred'], orient='split')
        res.columns = ['Prediction']
        pred_res = pd.concat([df, res], axis=1)

        st.write(pred_res)
        st.download_button(
            label='Download CSV',
            data=pred_res.to_csv(index=False),
            file_name='Data.csv',
            mime='text/csv')

        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            pred_res.to_excel(writer, index=False, sheet_name='Sheet1')
        excel_data = output.getvalue()
        st.download_button(
            label='Download Excel',
            data=excel_data,
            file_name='Data.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
