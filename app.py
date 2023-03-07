import time
import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Modeling
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, pdist
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, Birch

st.set_page_config(page_title="Customer Segmentation Using PAM Algorithm",
                   page_icon="🗞️", layout="centered")


@st.cache(allow_output_mutation=True, show_spinner=False, ttl=3600, max_entries=10)
def build_model():
    with st.spinner("Loading models... this may take awhile! \n Don't stop it!"):
        with open('model_fix.pickle', 'rb') as f:
            itr_imputer, pca_n, model_fix = pickle.load(f)
        inference = itr_imputer, pca_n, model_fix
    return inference


itr_imputer, pca, inference = build_model()

st.title('🗞️ Customer Segmentation Using PAM Algorithm')

with st.expander('📋 About this app', expanded=True):
    st.markdown("""
    * Credit Card Customer Segmentation app is an easy-to-use tool that allows you to predict the segmentation of a given data customer.
    * You can predict one customer data at a time or upload .csv file to bulk predict.
    * Made by [Alpian Khairi](https://www.linkedin.com/in/alpiankhairi/),[Kurnia Minari](https://www.linkedin.com/in/kurniaminari), Dheanita, Zenedin, Munawaroh.
    """)
    st.markdown(' ')

with st.expander('🧠 About prediction model', expanded=False):
    st.markdown("""
    ### Credit Card Customer Segmentation
    * Model are trained using [PAM](https://towardsdatascience.com/k-medoid-clustering-pam-algorithm-in-python-with-solved-example-c0dcb35b3f46) based on [CC General Dataset](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata) from Arjun Bhasin on Kaggle.
    * The best model with Medoids with Cosine Distance, n Cluster = 4.
    * Metrics : Silhouette Score = 0.793798, Devies Bouldin Score = 0.605751,  Calinski Harabasz Score =  15759.281845.
    * **[Source Code](https://github.com/alvian2022/Customer-Segmentation-Using-PAM-Algorithm-and-Implementation-Using-Streamlit.git)**
    """)
    st.markdown(' ')


st.markdown(' ')
st.markdown(' ')

st.header('🔍 Customer Segmentation Prediction')


# INPUT
balance_frequency = st.number_input('balance_frequency')
# for onoff_proportion
oneoff_purchases = st.number_input('oneoff_purchases')
purchases = st.number_input('purchases', value=1)
oneoff_proportion = oneoff_purchases/purchases
if oneoff_proportion > 1:
    oneoff_proportion = 1
# for installments_proportion
installments_purchases = st.number_input('installments_purchases')
installments_proportion = installments_purchases/purchases
if installments_proportion > 1:
    installments_proportion = 1
purchases_frequency = st.number_input('purchases_frequency')
oneoff_frequency = st.number_input('oneoff_frequency')
installments_frequency = st.number_input('installments_frequency')
cash_advance_frequency = st.number_input('cash_advance_frequency')
if cash_advance_frequency > 1:
    cash_advance_frequency = 1
payments_proportion = st.number_input('payments_proportion')

raw = {
    'balance_frequency': [balance_frequency],
    'oneoff_proportion': [oneoff_proportion],
    'installments_proportion': [installments_proportion],
    'purchases_frequency':  [purchases_frequency],
    'oneoff_frequency': [oneoff_frequency],
    'installments_frequency': [installments_frequency],
    'cash_advance_frequency': [cash_advance_frequency],
    'payments_proportion': [payments_proportion],
}
if raw:
    with st.spinner('Loading prediction...'):
        df = pd.DataFrame.from_dict(raw)
        df_pred = pca.transform(df)
        df_pred = pd.DataFrame(df_pred)
        df_pred.columns = ['P1', 'P2']
        result = inference.predict(df_pred)
    st.markdown(f'Cluster for this customer is **[{result}]**')


st.markdown(' ')
st.markdown(' ')

st.header('🗃️ Bulk Customer Dataset Segmentation Prediction')
st.markdown(
    'Only upload .csv file that contains list of news titles separated by comma.')

uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = [str(x).lower() for x in df.columns]
    df_wo_id = df.drop('cust_id', axis=1)
    cash_advance_freq_more_than_1 = df_wo_id[df_wo_id['cash_advance_frequency'] > 1].index
    df_wo_id['cash_advance_frequency'].iloc[cash_advance_freq_more_than_1] = 1
    feat_cols = [col for col in df_wo_id.columns]

    df[feat_cols] = itr_imputer.transform(df[feat_cols])
    df_model = df.copy()
    df_model.drop(['cust_id', 'balance', 'purchases',
                   'cash_advance', 'cash_advance_trx',
                   'purchases_trx', 'credit_limit',
                   'payments', 'minimum_payments', 'tenure'],
                  axis=1, inplace=True, errors='ignore')
    # change oneoff_purchase to oneoff_proportion
    oneoff_proportion = df_wo_id['oneoff_purchases'] / df_wo_id['purchases']
    # change installments_purchase to installments_proportion
    installments_proportion = df_wo_id['installments_purchases'] / \
        df_wo_id['purchases']
    # rename columns
    df_model.rename(columns={'oneoff_purchases_frequency': 'oneoff_frequency',
                             'purchases_installments_frequency': 'installments_frequency',
                             'prc_full_payment': 'payments_proportion'},
                    inplace=True, errors='ignore')
    # Fill NaN value with zero value
    oneoff_proportion.fillna(0, inplace=True)
    installments_proportion.fillna(0, inplace=True)
    oneoff_more_than_1 = oneoff_proportion[oneoff_proportion > 1].index
    oneoff_proportion.iloc[oneoff_more_than_1] = 1

    installments_more_than_1 = installments_proportion[installments_proportion > 1].index
    installments_proportion.iloc[installments_more_than_1] = 1
    df_model.oneoff_purchases = oneoff_proportion
    df_model.installments_purchases = installments_proportion
    df_model.rename(columns={'oneoff_purchases': 'oneoff_proportion',
                             'installments_purchases': 'installments_proportion'},
                    inplace=True)

    df_pred = pca.transform(df_model)
    df_pred = pd.DataFrame(df_pred)
    df_pred.columns = ['P1', 'P2']

    with st.spinner('Loading prediction...'):
        result_labels = inference.predict(df_pred)

        df_results = pd.concat(
            [df, pd.DataFrame({'cluster': result_labels})], axis=1)
        cluster_name = df_results['cluster'].map(
            {'Balance Spender': '0', 'Money Hoarders': '1', 'Potential Customer': '2', 'Credit Lovers': '3'})
        Cluster_Name = []

        for i in df_results['cluster']:
            if i == 0:
                Cluster_Name.append('Balance Spender')
            elif i == 1:
                Cluster_Name.append('Money Hoarders')
            elif i == 2:
                Cluster_Name.append('Potential Customer')
            elif i == 3:
                Cluster_Name.append('Credit Lovers')

        df_results['cluster_name'] = Cluster_Name
    st.markdown('#### Prediction Result')
    st.download_button(
        "Download Result",
        df_results.to_csv(index=False).encode('utf-8'),
        "News Title Category Prediction Result.csv",
        "text/csv",
        key='download-csv'
    )
    st.dataframe(df_results, 1000)
