import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
import requests


# Function to load Lottie animations
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load animations
success_animation = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_jcikwtux.json")
# eda_animation = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_2VCbEnQ71t.json")

# Display animation

# cluster_animation = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_5u6v7n6l.json")


phone_usage_classification = pickle.load(open("C:/Users/Admin/OneDrive/Documents/Phone_Usage/randomforestclassifier_best_model.pkl", "rb"))
phone_usage_cluster = pickle.load(open("C:/Users/Admin/OneDrive/Documents/Phone_Usage/KMeans_model.pkl", "rb"))

phone_usage_csv = pd.read_csv("C:/Users/Admin/OneDrive/Documents/Phone_Usage/Phone_Usage_csv.csv")

df = pd.read_csv("C:/Users/Admin/OneDrive/Documents/Phone_Usage/phone_usage_india.csv") 

st.set_page_config(page_title = "Decoding Phone Usage Patterns in India")
# st.write("This app allows users to classify their primary phone usage and analyze user clusters based on device statistics.")
st_lottie(success_animation, height=200, key="title_anim")

# Apply custom styling
st.markdown("""
    <style>
    .stApp {
        background-color: #eef2f3;
    }
    .title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #2E86C1;
    }
    .prediction {
        font-size: 28px;
        font-weight: bold;
        color: #28B463;
        text-align: center;
        padding: 10px;
    }
    .sidebar .block-container {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)



st.sidebar.title("User Input Features")

choice = st.sidebar.radio(label = "Select a section", options = ["Home", "Primary Use Prediction", "Primary Use Clustering", "Exploratory Data Analysis"])

location_mapping = {'Ahmedabad' :0, 'Bangalore': 1, 'Chennai' : 2,  'Delhi': 3, 'Hyderabad': 4,  'Jaipur': 5, 'Kolkata':6, 'Lucknow':7, 'Mumbai':8, 'Pune':9}
phone_brand_mapping = {'Apple':0, 'Google Pixel': 1, 'Motorola': 2, 'Nokia': 3, 'OnePlus': 4, 'Oppo': 5, 'Realme': 6, 'Samsung': 7, 'Vivo': 8, 'Xiaomi': 9}

# expected_features = best

def user_input_features():
    age = st.slider("Age", 10, 80, 30)
    location = st.selectbox("Location", list(location_mapping.keys()))
    phone_brand = st.selectbox("Phone Brand",  list(phone_brand_mapping.keys()))
    screen_time = st.slider("Screen Time (hrs/day)", 0, 12, 4)
    data_usage = st.slider("Data Usage (GB/month)", 0, 100, 5)
    calls_duration = st.slider("Calls Duration (mins/day)", 0, 300, 60)
    apps_installed = st.slider("Number of Apps Installed", 0, 100, 10)
    social_media_time = st.slider("Social Media Time (hrs/day)", 0, 10, 2)
    ecommerce_spend = st.slider("E-commerce Spend (INR/month)", 0, 50000, 1000, 500)
    streaming_time = st.slider("Streaming Time (hrs/day)", 0, 10, 3)
    gaming_time = st.slider("Gaming Time (hrs/day)", 0, 10, 1)
    recharge_cost = st.slider("Monthly Recharge Cost (INR)", 0, 5000, 500, 100)
    
    encoded_location = location_mapping[location]
    encoded_phone_brand = phone_brand_mapping[phone_brand]
    


    data = {
        "Age": age,
        "Location": encoded_location,
        "Phone Brand": encoded_phone_brand,
        "Screen Time (hrs/day)": screen_time,
        "Data Usage (GB/month)": data_usage,
        "Calls Duration (mins/day)": calls_duration,
        "Number of Apps Installed": apps_installed, 
        "Social Media Time (hrs/day)": social_media_time,
        "E-commerce Spend (INR/month)": ecommerce_spend,
        "Streaming Time (hrs/day)": streaming_time,
        "Gaming Time (hrs/day)": gaming_time,
        "Monthly Recharge Cost (INR)": recharge_cost
    }
    return pd.DataFrame([data])

primary_use_mapping = {0: 'Education', 1: 'Entertainment', 2: 'Gaming', 3: 'Social Media', 4: 'Work'}

if choice == "Home" :
    st.title("Decoding Phone Usage Patterns in India")
    st.write("This app allows users to classify their primary phone usage and analyze user clusters based on device statistics.")

elif choice == "Primary Use Prediction" :

    model_features = phone_usage_classification.feature_names_in_
    input_df = user_input_features()
    input_df = input_df[model_features]
    input_df = input_df.astype(float)

    st.subheader("Predict Primary Phone Usage")
    
    # st.write("### User Input Features")
    # st.dataframe(input_df, use_container_width=True)
    if st.button("Predict Primary Use"):
        class_prediction = phone_usage_classification.predict(input_df)[0]
        decoded_class_prediction =  primary_use_mapping[class_prediction]
        
        st.markdown(f"<p class='prediction'> Predicted Primary Use: {decoded_class_prediction}</p>", unsafe_allow_html=True)

elif choice == "Primary Use Clustering":
    st.subheader("Find User Cluster Group")
    input_df = user_input_features()
    input_df = input_df[phone_usage_cluster.feature_names_in_]
    input_df = input_df.astype(float)
    if st.button(" Find Cluster Group"):
        cluster_prediction = phone_usage_cluster.predict(input_df)[0]
        decoded_cluster_prediction = primary_use_mapping[cluster_prediction]
        # st_lottie(cluster_animation, height=150, key="cluster_pred")
        st.markdown(f"<p class='prediction'> Assigned Cluster: {decoded_cluster_prediction}</p>", unsafe_allow_html=True)

elif choice == "Exploratory Data Analysis" :

    # st_lottie(eda_animation, height=200, key="eda")

    st.title("Exploratory Data Analysis")

    option = st.selectbox("Select the Visualiazation", ["Screen time distribution", "Frequency of particular brand users" ,"Heat map", "Average Data Usage by Age Group",
                                                        "Number of Apps Installed by Age Group","Frequency of Primary Use cases",
                                                        "Monthly spent amount on e-commerce by different ages", "Primary use vs Screen time",
                                                        "Primary use vs Social Media Time"])
    if option == "Screen time distribution" :
        plt.figure(figsize=(8,5))
        sns.histplot(df['Screen Time (hrs/day)'], color='lavender')
        plt.title('Screen time distribution')
        plt.ylabel('frequency')
        plt.show()
        fig = plt.gcf()
        st.pyplot(fig)

    elif option == "Frequency of particular brand users" :
        plt.figure(figsize=(8,5))
        sns.countplot(x = df['Phone Brand'])
        plt.xticks(rotation = 45)
        plt.ylabel('Frequency')
        plt.title('Frequency of a particular brand users')
        plt.show()
        fig = plt.gcf()
        st.pyplot(fig)

    elif option == "Heat map" :
        corr_matrix = phone_usage_csv.corr()

        # Plot heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Feature Correlation Heatmap")
        plt.show()
        fig = plt.gcf()
        st.pyplot(fig)
    
    elif option == "Average Data Usage by Age Group" :
        bins = [10, 18, 25, 35, 50, 65]
        labels = ["10-18", "19-25", "26-35", "36-50", "51-65"]
        df["Age Group"] = pd.cut(df["Age"], bins=bins, labels=labels)

        # Calculate mean values per age group
        age_group_stats = df.groupby("Age Group")["Data Usage (GB/month)"].mean()
        age_grp_stats_apps = df.groupby("Age Group")["Number of Apps Installed"].count()
        print(age_group_stats)
        print(age_grp_stats_apps)

        plt.figure(figsize=(8, 5))

        # Bar plot for Data Usage
        sns.barplot(data=df, x="Age Group", y="Data Usage (GB/month)", palette="Blues")
        plt.title("Average Data Usage by Age Group")
        plt.ylabel("Data Usage (GB/month)")
        plt.show()
        fig = plt.gcf()
        st.pyplot(fig)

    elif option ==  "Number of Apps Installed by Age Group" :
        bins = [10, 18, 25, 35, 50, 65]
        labels = ["10-18", "19-25", "26-35", "36-50", "51-65"]
        df["Age Group"] = pd.cut(df["Age"], bins=bins, labels=labels)

        # Calculate mean values per age group
        age_group_stats = df.groupby("Age Group")["Data Usage (GB/month)"].mean()
        age_grp_stats_apps = df.groupby("Age Group")["Number of Apps Installed"].count()
        print(age_group_stats)
        print(age_grp_stats_apps)

        plt.figure(figsize=(7, 5))

        sns.barplot(data=df, x="Age Group", y="Number of Apps Installed", palette="Oranges")
        plt.title("Average Number of Apps Installed by Age Group")
        plt.ylabel("Number of Apps Installed")

        plt.tight_layout()
        plt.show()
        fig = plt.gcf()
        st.pyplot(fig)

    elif option == "Frequency of Primary Use cases" :
        primary_use_counts = df['Primary Use'].value_counts()

        # Plot the pie chart
        plt.figure(figsize=(6, 8))
        plt.pie(primary_use_counts, labels=primary_use_counts.index, autopct='%1.1f%%', colors=plt.cm.Paired.colors)
        plt.title('Frequency of Primary Use cases')
        plt.show()
        fig = plt.gcf()
        st.pyplot(fig)

    elif option == "Monthly spent amount on e-commerce by different ages" :
        plt.figure(figsize=(12,5))
        grouped_data = df.groupby('Age')['E-commerce Spend (INR/month)'].mean().reset_index()
        sns.barplot(grouped_data, x = 'Age', y = 'E-commerce Spend (INR/month)', palette = 'magma')
        plt.title('Monthly spent amount on e-commerce by different ages')
        plt.show()
        fig = plt.gcf()
        st.pyplot(fig)

    elif option == "Primary use vs Screen time" :
        plt.figure(figsize=(8,5))
        ax = sns.boxplot(data = df, x = 'Primary Use', y = 'Screen Time (hrs/day)', palette = 'Purples')
        plt.title("Primary use vs Screen time")
        plt.show()
        fig = plt.gcf()
        st.pyplot(fig)

    elif option == "Primary use vs Social Media Time" :
        plt.figure(figsize=(8,5))
        grouped_data = df.groupby('Primary Use')['Social Media Time (hrs/day)'].mean().reset_index()
        ax = sns.barplot(grouped_data, x = 'Primary Use', y = 'Social Media Time (hrs/day)', palette = 'Blues')
        plt.title("Primary use vs Social Media Time")
        print(grouped_data)
        plt.show()
        fig = plt.gcf()
        st.pyplot(fig)