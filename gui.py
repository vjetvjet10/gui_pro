import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
import streamlit as st
import os
import pickle
column_names = ["CustomerID", "Date", "Quantity", "TotalAmount"]
@st.cache_data
def load_data():
    #column_names = ["CustomerID", "Date", "Quantity", "TotalAmount"]
    if os.path.exists("CDNOW_master.txt"):
        df = pd.read_csv("CDNOW_master.txt", delim_whitespace=True, names=column_names)
        df["Date"] = pd.to_datetime(df["Date"], format='%Y%m%d')
        with open('dataframe.pkl', 'wb') as f:
            pickle.dump(df, f)
        return df
    else:
        st.error("File 'CDNOW_master.txt' not found.")
        return pd.DataFrame()

def rfm_level(row):
    if row['Recency'] <= 207 and row['Frequency'] >= 3 and row['Monetary'] >= 106:
        return 'Champions'
    elif row['Frequency'] >= 3:
        return 'Loyal Customers'
    elif row['Recency'] <= 207 and row['Frequency'] >= 2:
        return 'Potential Loyalists'
    elif row['Recency'] <= 207 and row['Frequency'] == 1:
        return 'New Customers'
    elif row['Recency'] <= 367 and row['Frequency'] == 1:
        return 'Promising'
    elif row['Recency'] > 471 and row['Frequency'] >= 2 and row['Monetary'] >= 106:
        return 'At Risk'
    elif row['Recency'] > 505 and row['Frequency'] >= 3 and row['Monetary'] >= 106:
        return "Cannot Lose Them"
    elif row['Recency'] > 505 and row['Frequency'] == 1:
        return 'Hibernating'
    else:
        return 'Others'
def calculate_rfm(df):
    if os.path.exists('rfm.pkl') and os.path.exists('rfm_agg3.pkl') and os.path.exists('linkage_matrix.pkl'):
        with open('rfm.pkl', 'rb') as f:
            rfm = pickle.load(f)
        with open('rfm_agg3.pkl', 'rb') as f:
            rfm_agg3 = pickle.load(f)
        with open('linkage_matrix.pkl', 'rb') as f:
            linkage_matrix = pickle.load(f)
        return rfm, rfm_agg3, linkage_matrix
    recency = df.groupby('CustomerID')['Date'].max()
    last_date = df['Date'].max()
    recency = (last_date - recency).dt.days
    frequency = df.groupby('CustomerID')['Date'].count()
    monetary = df.groupby('CustomerID')['TotalAmount'].sum()

    rfm = pd.DataFrame({
        'Recency': recency,
        'Frequency': frequency,
        'Monetary': monetary
    })
    df_hi = rfm[['Recency','Frequency','Monetary']]

# Normalize the RFM data
    scaler = StandardScaler()
    data_rfm_normalized = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    data_rfm_normalized = pd.DataFrame(data_rfm_normalized, index=rfm.index, columns=['Recency', 'Frequency', 'Monetary'])
    linkage_matrix = linkage(data_rfm_normalized, method='ward')
    hierarchical = AgglomerativeClustering(n_clusters=4,
                       linkage='ward',
                       metric='euclidean')
    hierarchical.fit(df_hi)
    sil_hi = metrics.silhouette_score(df_hi, hierarchical.labels_, metric='euclidean')
    df_hi = df_hi.copy()
    df_hi["Cluster_hierarachical"] = hierarchical.labels_
    # Calculate average values for each RFM_Level, and return a size of each segment
    rfm_agg3 = df_hi.groupby('Cluster_hierarachical').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'count']}).round(0)

    rfm_agg3.columns = rfm_agg3.columns.droplevel()
    rfm_agg3.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
    rfm_agg3['Percent'] = round((rfm_agg3['Count']/rfm_agg3.Count.sum())*100, 2)

    # Reset the index
    rfm_agg3 = rfm_agg3.reset_index()

    # Change thr Cluster Columns Datatype into discrete values
    rfm_agg3['Cluster_hierarachical'] = 'Cluster_hierarachical'+ rfm_agg3['Cluster_hierarachical'].astype('str')
    # Save the results to pickle files before returning them
    with open('rfm.pkl', 'wb') as f:
        pickle.dump(rfm, f)
    with open('rfm_agg3.pkl', 'wb') as f:
        pickle.dump(rfm_agg3, f)
    with open('linkage_matrix.pkl', 'wb') as f:
        pickle.dump(linkage_matrix, f)

    return rfm, rfm_agg3, linkage_matrix


def load_model():
    with open('rfm.pkl', 'rb') as f:
        rfm = pickle.load(f)
    with open('linkage_matrix.pkl', 'rb') as f:
        linkage_matrix = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    hierarchical = AgglomerativeClustering(n_clusters=4, linkage='ward', metric='euclidean')
    
    return rfm, linkage_matrix, scaler, hierarchical

def calculate_rfm_for_new_customer(data, last_date):
    # Chuyển đổi định dạng ngày tháng
    data['Date'] = pd.to_datetime(data['Date'], format='%Y%m%d')
    
    # Tính toán giá trị RFM
    #last_date = data['Date'].max()
    recency = (last_date - data['Date'].max()).days
    frequency = data['Date'].count()
    monetary = data['TotalAmount'].sum()
    
    return pd.DataFrame({'Recency': [recency], 'Frequency': [frequency], 'Monetary': [monetary]})

def classify_new_customer(data, scaler, hierarchical):
    # Chuẩn hóa dữ liệu với scaler đã lưu
    data_normalized = scaler.transform(data)
    # Phân loại khách hàng mới với mô hình phân cụm hierarchical đã lưu
    cluster = hierarchical.fit_predict(data_normalized)

    return cluster

def main():
    st.title("Data Science Project 1")
    st.write("## Customer Segmentation")

    #uploaded_file = st.sidebar.file_uploader("Choose a file")
    uploaded_file =st.file_uploader("Choose a file", type=['csv'])
    if uploaded_file:
        with open('CDNOW_master.txt', 'wb') as f:
            f.write(uploaded_file.read())
        st.success("File uploaded successfully.")

    if os.path.exists('dataframe.pkl'):
        # Load the DataFrame from the .pkl file
        with open('dataframe.pkl', 'rb') as f:
            df = pickle.load(f)
    else:
        df = load_data()
    
    if df.empty:
        st.write("No data available. Please upload a valid 'CDNOW_master.txt' file.")
        return
    rfm, rfm_agg3,linkage_matrix = calculate_rfm(df)

    menu = ["Business Objective", "Build Project", "Report", "Classify New Customer"]

    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'Business Objective':    
        st.subheader("Business Objective")
        st.write("""
    ###### CDNOW is a renowned retailer specializing in CDs and related music products. The dataset, which captures the purchase history of a distinct customer cohort, offers a wealth of insights into consumer behavior that has the potential to significantly impact strategic decision-making. These insights will be instrumental in fine-tuning marketing strategies, fostering stronger customer connections, and achieving optimal revenue growth. The core business objective is to gain insights into distinct customer segments characterized by their purchasing behavior, encompassing factors such as recency, frequency, and monetary value (RFM analysis). These insights hold the key to refining marketing strategies and crafting personalized customer interactions for enhanced engagement.
    """)  
        st.write("""###### => Problem/ Requirement: Use Machine Learning algorithms in Python for Custumer Segmentation.""")

    elif choice == 'Build Project':
        st.subheader("Build Project")
        #st.write("##### Some data")
        st.write("##### Initial data information")
        # Get DataFrame info as string and display it
        buffer = StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
        st.write("##### A few rows of data")
        st.dataframe(df.head())
        st.write("##### Data preprocessing")
  
        st.write("##### Let’s take a closer look at the data we will need to manipulate.")
        st.write(f'Transactions timeframe from {df["Date"].min()} to {df["Date"].max()}')
        st.write(f'{df[df.CustomerID.isnull()].shape[0]:,} transactions don\'t have a customer id')
        st.write(f'{len(df.CustomerID.unique()):,} unique customer_id')

        st.write("##### Some data after Create RFM analysis for each customers, Calculate RFM quartiles, Concat RFM quartile values to create RFM Segments, Create the custom RFM levels")
        st.dataframe(rfm.head())
        st.write("##### Shape of data")
        st.dataframe(rfm.shape)

    elif choice == 'Report':
        if os.path.exists('linkage_matrix.pkl'):
            # Load the linkage_matrix from the .pkl file
            with open('linkage_matrix.pkl', 'rb') as f:
                linkage_matrix = pickle.load(f)
        else:
            rfm, rfm_agg3, linkage_matrix = calculate_rfm(df)
        
        # Plot the dendrogram
        st.write("#### Plot the Dendrogram")
        plt.figure(figsize=(12, 6))
        dendrogram(linkage_matrix, p=30, truncate_mode='lastp')
        plt.xlabel("Customers")
        plt.ylabel("Euclidean Distance")
        plt.title("Dendrogram")
        st.pyplot(plt)
        
        # Load rfm_agg3 if not already loaded
        if 'rfm_agg3' not in locals():
            rfm, rfm_agg3, _ = calculate_rfm(df)
        
        st.write("#### Results after applying RFM and Hierarchical Clustering Analysis")
        st.dataframe(rfm_agg3)


    elif choice == 'Classify New Customer':

        rfm, linkage_matrix, scaler, hierarchical = load_model()
        st.subheader("Classify New Customer")
        #column_names = ["CustomerID", "Date", "Quantity", "TotalAmount"]
        uploaded_file = st.file_uploader("Choose a file with the new customer data", type=['csv', 'txt'])
        if uploaded_file:
            new_customer_data = pd.read_csv(uploaded_file, delim_whitespace=True, names=column_names)
            last_date = df['Date'].max()
            new_customer_rfm = calculate_rfm_for_new_customer(new_customer_data, last_date)
            #new_customer_rfm = calculate_rfm_for_new_customer(new_customer_data)
            cluster = classify_new_customer(new_customer_rfm, scaler, hierarchical)
            st.write(f"The new customer has been classified into cluster: {cluster[0]}")
     
if __name__ == "__main__":
    main()
