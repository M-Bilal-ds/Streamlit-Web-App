import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import streamlit as st

#css for styling
st.markdown(
    """
    <style>
    h1 {
        color: #d6eaf8 !important;
        font-family: cursive;
        text-align: center;
        border-bottom: 2px solid #86eaf8;
    }
    /* Background */
    body {
        background: linear-gradient(to right, #34495e, #86eaf8);
        font-family: 'Cursive', sans-serif;
        color: #fbfcfc;
    }

    /* Header */
    h2, h3 {
        color: #d6eaf8;
        text-align: center;
        font-family: 'Cursive', sans-serif;
        border: 2px solid #d6eaf8;  /* Border color */
        border-radius: 5px;  /* Rounded corners */
        background-color: #283747 !important;
        margin: 15px 15px;
    }

    /* Margin-bottom for headings in EDA and Visualizations sections */
    .stMarkdown h3, .stMarkdown h4 {
        margin-bottom: 30px !important;  /* Space below EDA and Visualizations headings */
    }

    /* Styling for buttons */
    .css-1emrehy.edgvbvh3 {
        background-color: #86eaf8 !important;
        color: white !important;
        border-radius: 12px;
        font-size: 18px;
        padding: 10px 20px;
    }

    .css-1emrehy.edgvbvh3:hover {
        background-color: #34495e !important;
    }

    /* Sidebar customization */
    .stSidebar {
        background-color: #283747 !important;
        color: #fbfcfc;
        font-family: 'Cursive', sans-serif;
    }

    /* Apply styles only to custom Markdown inside tabs */
    .custom-tab-markdown {
    background-color: rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    padding: 15px;
    margin-top: 15px;
    }
    
    /* Add space between sections */
    .stSection {
        margin-bottom: 20px;
    }

    /* Add margin above and below Key Insights heading */
    .stMarkdown h3 {
        margin-top: 30px !important;
        margin-bottom: 30px !important;  /* Space before and after Key Insights heading */
    }

    /* Styling for links */
    a {
        color: #86eaf8;
        text-decoration: none;
    }

    a:hover {
        color: #d6eaf8;
        text-decoration: underline;
    }
    .footer {
        background-color: #34495e;
        color: #fbfcfc; 
        text-align: center;  
        padding: 15px;
        font-family: 'Cursive', sans-serif;
        #position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        border-top: 2px solid #86eaf8;
        #display: flex;
        justify-content: center;
        gap: 10px;
        margin-top: 20px;
    }
    .footer a {
        color: #86eaf8;
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Student Depression Analysis")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Introduction :book:", "EDA :bar_chart:", "Model :robot_face:", "Key Insights 💡" ,"Conclusion :memo:"]
)

def load_data():
    data = pd.read_csv("Student Depression Dataset.csv")  
    return data

def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

with tab1:
    # Introduction
    st.markdown("### Introduction")
    st.markdown("""<div class="custom-tab-markdown">
    Mental health issues, particularly depression, have become a growing concern among students due to various academic, social, and personal pressures. This project aims to analyze and predict depression levels among students using a comprehensive dataset and advanced analytical techniques. The results of this analysis will help educators, policymakers, and mental health professionals understand the key factors contributing to student depression and provide actionable insights for interventions.

---                    
#### Dataset Description

The dataset, titled **"Student Depression Dataset"**, contains information about 27,901 students. Each record represents a unique individual, with detailed attributes related to their demographic, academic, and mental health status. The key features of the dataset include:

- **Demographic Data:** `Gender`, `Age`, `City`, `Profession`
- **Academic and Work Pressure:** `Academic Pressure`, `Work Pressure`, `CGPA`, `Study Satisfaction`, `Job Satisfaction`
- **Lifestyle Factors:** `Sleep Duration`, `Dietary Habits`, `Work/Study Hours`
- **Mental Health Indicators:** `Financial Stress`, `Family History of Mental Illness`, `Depression`
- **Suicidal Thoughts:** `Have you ever had suicidal thoughts?`
---
#### Key Statistics
- **Shape:** (27,901 rows × 18 columns)
- **Missing Values:** Only the `Financial Stress` column has 3 missing entries; all other columns are complete.
- **Sample Data:**

| **Feature**                        | **Example**                     |
|------------------------------------|---------------------------------|
| Gender                             | Male, Female                   |
| Age                                | 24, 33                         |
| Sleep Duration                     | 5-6 hours, Less than 5 hours   |
| Depression (Label)                 | 0 (No), 1 (Yes)                |
| Have you ever had suicidal thoughts? | Yes, No                      |

---

#### Notebook Description

The accompanying Jupyter notebook titled **"Student Depression Analysis and Prediction"** is structured into 66 cells, including:

- **Markdown Cells (Introduction and Insights):** Providing contextual information, methodology, and analysis conclusions.
- **Code Cells:** Implementing data preprocessing, exploratory data analysis (EDA), visualization, and prediction models.

---
#### Notable Sections
1. **Data Preprocessing:** Handling missing values, encoding categorical variables, and normalizing numeric data.
2. **Exploratory Data Analysis:** Visualizing trends such as the impact of `Sleep Duration` on `Depression`.
3. **Correlation Analysis:** Examining relationships between variables such as `Academic Pressure`, `Financial Stress`, and `Depression` to identify significant predictors.
4. **Feature Engineering:** Transforming and selecting the most relevant features to enhance model performance, with attention to balancing lifestyle, academic, and psychological factors.
5. **Machine Learning Model:** Utilizing Logistic Regression algorithm to predict depression labels with metrics such as accuracy and F1 score.
</div>""", unsafe_allow_html=True)

# Load sidebar
st.sidebar.header("Data")
data = load_data()
st.sidebar.write(data)

with tab2:
    # EDA Section
    st.markdown("### Exploratory Data Analysis (EDA)")
    overview_data = {
            "Column": data.columns,
            "Non-Null Count": data.notnull().sum(),
            "Dtype": data.dtypes,
        }
    overview_df = pd.DataFrame(overview_data).reset_index(drop=True)
    if st.checkbox("Show Dataset Overview"):
        st.dataframe(overview_df)
    if st.checkbox("Show Dataset Summary"):
        st.write(data.describe())


    # Visualizations
    st.markdown("### Visualizations")

    if st.checkbox("Univariate Numerical Analysis"):
        numerical_columns = data.select_dtypes(include=['number']).columns

        num_cols = len(numerical_columns)
        cols = 3
        rows = (num_cols + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5), constrained_layout=True)
        axes = axes.flatten()

        for i, j in enumerate(numerical_columns):
            sns.histplot(data[j], kde=True, bins=30,  color='darkcyan', ax=axes[i])
            axes[i].set_title(f"Distribution of {j}", fontsize=14)
            axes[i].set_xlabel(j, fontsize=12)
            axes[i].grid(axis='y', linestyle='--', alpha=0.6)
            axes[i].set_ylabel("Frequency", fontsize=12)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        st.pyplot(fig)

    if st.checkbox("Univariate Categorical Analysis"):
        categorical_columns = data.select_dtypes(include=['object']).columns

        num_cols = len(categorical_columns)
        cols = 3
        rows = (num_cols + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 6), constrained_layout=True)
        axes = axes.flatten()

        for i, j in enumerate(categorical_columns):
            sns.countplot(data=data, y=j, order=data[j].value_counts().index, palette="viridis", ax=axes[i])
            axes[i].set_title(f"Frequency of {j}", fontsize=14)
            axes[i].grid(axis='y', linestyle='--', alpha=0.6)
            axes[i].set_xlabel("Count", fontsize=12)
            axes[i].set_ylabel(j, fontsize=12)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        st.pyplot(fig)

    if st.checkbox("Bivariate Numerical Analysis"):
        numerical_columns = [col for col in data.select_dtypes(include=['number']).columns if col != 'Depression']
        n_plots = len(numerical_columns)
        
        n_cols = 2
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 5))
        axes = axes.flatten()

        for i, col in enumerate(numerical_columns):
            sns.boxplot(x='Depression', y=col, data=data, hue='Gender', palette='viridis', ax=axes[i])
            axes[i].set_title(f'{col} Distribution Across Depression Levels', fontsize=14)
            axes[i].set_xlabel('Depression')
            axes[i].set_ylabel(col)
            axes[i].grid(axis='y', linestyle='--', alpha=0.6)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        st.pyplot(fig)

    if st.checkbox("Bivariate Categorical Analysis"):
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns

        n_cols = 2
        n_rows = (len(categorical_columns) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 6))
        axes = axes.flatten() 

        for i, col in enumerate(categorical_columns):
            sns.countplot(data=data, x=col, hue='Depression', palette='viridis',
                        order=data[col].value_counts().index, ax=axes[i])
            axes[i].set_title(f'{col} Distribution Across Depression Levels', fontsize=14)
            axes[i].set_xlabel(col, fontsize=12)
            axes[i].set_ylabel('Count', fontsize=12)
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(axis='y', linestyle='--', alpha=0.6)
            axes[i].legend(title='Depression Levels', fontsize=10)

        for j in range(len(categorical_columns), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        st.pyplot(fig)

    if st.checkbox("Correlation Analysis"):
        numeric_df = data.select_dtypes(include=['number'])
        corr = numeric_df.corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='viridis', ax=ax)
        ax.set_title('Correlation Heatmap')

        st.pyplot(fig)

    if st.checkbox("Pairwise Analysis"):
        numerical_columns = data.select_dtypes(include=['number']).columns
        pairplot = sns.pairplot(data=data[numerical_columns], 
                                diag_kind='kde', 
                                corner=True, 
                                plot_kws={'alpha': 0.7})
        pairplot.fig.suptitle('Pairwise Relationships Between Features', y=1.02)

        st.pyplot(pairplot.fig)

with tab3:
    # Model Section
    st.markdown("### Model Predictions")
    model = load_model()

    def load_data():  
        unique_cities = list(data['City'].unique())
        unique_professions = list(data['Profession'].unique())  
        unique_degrees = list(data['Degree'].unique())
        unique_cities.append("Others")
        unique_professions.append("Others")
        unique_degrees.append("Others")
        return data, unique_cities, unique_professions, unique_degrees

    data, unique_cities, unique_professions, unique_degrees = load_data()
    binary_map = {'Yes': 1, 'No': 0}
    ordinal_map = {
        'Sleep Duration': {"Less than 5 hours": 1, "5-6 hours": 2, "7-8 hours": 3, "More than 8 hours": 4, "Others": 0},
        'Dietary Habits': {"Unhealthy": 1, "Moderate": 2, "Healthy": 3, "Others": 0}
    }

    col1, col2 = st.columns(2)

    with col1:
        id = st.text_input('ID')
        age = st.number_input("Age", value=0, min_value=0)
        profession = st.selectbox("Profession", unique_professions)
        if profession == "Others":
            profession_vector = [0 for _ in unique_professions[1:]]  # Represent "Others" as a zero vector
        else:
            profession_vector = [1 if profession == unique else 0 for unique in unique_professions[1:]]
        sleep_duration = st.selectbox("Sleep Duration", options=["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours", "Others"])
        degree = st.selectbox("Degree", unique_degrees)
        if degree == "Others":
            degree_vector = [0 for _ in unique_degrees[1:]]  # Represent "Others" as a zero vector
        else:
            degree_vector = [1 if degree == unique else 0 for unique in unique_degrees[1:]]
        family_history = st.radio("Family History of Mental Illness", ["Yes", "No"])

    with col2:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        city = st.selectbox("City", unique_cities)
        if city == "Others":
            city_vector = [0 for _ in unique_cities[1:]]  # Represent "Others" as a zero vector
        else:
            city_vector = [1 if city == unique else 0 for unique in unique_cities[1:]]
        cgpa = st.number_input("CGPA", value=0.0, min_value=0.0, max_value=10.0)
        dietary_habits = st.selectbox("Dietary Habits", options=["Healthy", "Moderate", "Unhealthy", "Others"])
        work_study_hours = st.number_input("Work/Study Hours (hours per day)", value=0.0)
        suicidal_thoughts = st.radio("Have you ever had suicidal thoughts?", ["Yes", "No"])

    academic_pressure = st.slider("Academic Pressure", 0, 10, 0)
    work_pressure = st.slider("Work Pressure", 0, 10, 0)
    study_satisfaction = st.slider("Study Satisfaction", 0, 10, 0)
    job_satisfaction = st.slider("Job Satisfaction", 0, 10, 0)
    financial_stress = st.slider("Financial Stress", 0, 10, 0)
    
        

    if st.button("Predict Depression Level :mag_right:"):
        features = [
            academic_pressure,
            cgpa,
            study_satisfaction,
            ordinal_map['Sleep Duration'][sleep_duration],
            ordinal_map['Dietary Habits'][dietary_habits],
            binary_map[suicidal_thoughts],
            work_study_hours,
            financial_stress,
            binary_map[family_history],
        ]

        prediction = model.predict([features])
        prediction_label = "Depression Detected" if prediction[0] == 1 else "No Depression Detected"
        prediction_color = "red" if prediction[0] == 1 else "green"
        prediction_icon = "⚠️" if prediction[0] == 1 else "✅"
        
        st.markdown(
            f"""
            <div style="
                background-color: {prediction_color}; 
                color: white; 
                padding: 15px; 
                border-radius: 10px; 
                text-align: center;
                font-size: 18px;
                font-weight: bold;">
                {prediction_icon} {prediction_label}
            </div>
            """,
            unsafe_allow_html=True,
        )
  
with tab4:
    st.markdown("### Key Insights")

    st.info("Most students experiencing depression report less than 5 hours of sleep.")
    st.success("Higher study satisfaction correlates with lower depression levels.")
    st.warning("Students with high academic pressure are significantly more likely to experience depression.")
    st.error("Financial stress is a major contributor to depression among students.")
    st.info("Students with a family history of mental illness show a higher prevalence of depression.")
    st.success("Job satisfaction is inversely related to depression levels, with higher satisfaction leading to better mental health.")
    st.warning("A significant portion of students with depression reported experiencing suicidal thoughts.")
    st.info("Balanced sleep and work schedules are associated with lower depression rates.")
    st.success("Supportive family and social environments reduce the likelihood of depression.")
    st.error("Students reporting unhealthy dietary habits are more prone to depression.")
    st.warning("There is a noticeable correlation between prolonged work/study hours and increased depression risk.")

with tab5:
    st.markdown("### Conclusion")
    st.markdown("""<div class="custom-tab-markdown">
    The <b>Student Depression Analysis and Prediction<b> project underscores the multifaceted nature of mental health challenges 
among students. By exploring the interplay of academic, lifestyle, and psychological factors, this analysis provides a robust
framework for understanding and addressing depression.

---                
#### Recommendations:
- **For Institutions:** Introduce stress management programs, enhance academic counseling services, and ensure better workload 
    distribution.
- **For Students:** Promote healthier lifestyles, including adequate sleep, balanced diets, and time management strategies.
- **For Policymakers:** Increase funding for student mental health resources and create awareness campaigns to reduce stigma around 
    seeking help.

---
#### Future Work:
- Expanding the dataset to include longitudinal studies for tracking mental health trends over time.
- Incorporating advanced models like deep learning for improved predictive capabilities.
- Collaborating with mental health professionals to design targeted intervention programs based on data-driven insights.

---                
This project not only advances the understanding of student mental health but also provides practical tools and strategies for making a positive impact in educational settings.
</div>""", unsafe_allow_html=True)

st.sidebar.success("Select a section above to explore!")

st.markdown(
    """
    <div class="footer">
        <p><i class="fas fa-user icon"></i>© Developed by Muhammad Bilal</p>
        <p><a href="https://www.kaggle.com/bilalabdulmalik" target="_blank">Kaggle Profile</a></p>
        <p><a href="https://www.kaggle.com/code/bilalabdulmalik/student-depression-analysis-and-prediction" target="_blank">Relevant Notebook</a></p>
    </div>
    """,
    unsafe_allow_html=True,
)

