# Student Depression Analysis and Prediction

## Overview

This project explores the factors contributing to depression among students through a comprehensive dataset and predictive modeling. It aims to analyze key academic, lifestyle, and psychological factors to provide actionable insights for educators, policymakers, and mental health professionals.

The repository contains the following:
- A dataset of student mental health attributes.
- A Jupyter notebook for detailed data analysis and modeling.
- A web application for user-friendly exploration and prediction.

---

## Files and Directories

- **`Student Depression Dataset.csv`**: The dataset contains 27,901 entries with 18 attributes such as demographic data, academic pressures, lifestyle habits, and mental health indicators.
- **`Student Depression Analysis and Prediction.ipynb`**: A Jupyter notebook performing data preprocessing, exploratory data analysis (EDA), visualization, and predictive modeling using Logistic Regression.
- **`app.py`**: A Streamlit web application providing a user-friendly interface for exploring data insights and predicting depression levels.

---

## Features

### Dataset
The dataset includes:
- **Demographic Attributes**: Gender, Age, City, and Profession.
- **Academic and Work Pressure**: Academic Pressure, Work Pressure, CGPA, Study Satisfaction, and Job Satisfaction.
- **Lifestyle Factors**: Sleep Duration, Dietary Habits, and Work/Study Hours.
- **Mental Health Indicators**: Financial Stress, Family History of Mental Illness, and Depression labels.
- **Other Features**: Responses to questions about suicidal thoughts.

### Notebook Highlights
1. **Data Preprocessing**: Handling missing values, encoding categorical features, and normalizing numeric data.
2. **EDA and Visualization**:
   - Univariate, bivariate, and multivariate analyses.
   - Heatmaps, histograms, and box plots for key insights.
3. **Correlation Analysis**: Exploring relationships between variables like Academic Pressure, Financial Stress, and Depression.
4. **Machine Learning**: Logistic Regression model with performance metrics such as accuracy and F1 score.

### Web Application
The Streamlit app includes:
- Interactive data visualization and summary.
- Key insights highlighting major contributors to student depression.
- A model interface for predicting depression levels based on user input.

---

## Usage

### Prerequisites
- Python 3.7 or higher
- Required Python libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `streamlit`, `scikit-learn`

### Running the Notebook
1. Install required dependencies using `pip install -r requirements.txt`.
2. Open the notebook in Jupyter Lab/Notebook.
3. Run the cells sequentially to explore the dataset, perform EDA, and train the model.

### Running the Web App
1. Place the dataset (`Student Depression Dataset.csv`) and the trained model (`model.pkl`) in the root directory.
2. Run the application:
   ```bash
   streamlit run app.py

## Insights and Recommendations

### Key Insights
- **Sleep Duration**: Less than 5 hours correlates strongly with depression.
- **Academic Pressure**: High levels significantly increase the likelihood of depression.
- **Financial Stress**: Major contributor to mental health issues among students.
- **Family History**: A predictor for higher depression levels.

### Recommendations
- **For Institutions**: Implement stress management and counseling programs.
- **For Students**: Encourage balanced lifestyles and effective time management.
- **For Policymakers**: Increase funding for mental health resources and awareness campaigns.

---

## Future Work
- Expand the dataset with longitudinal studies to track mental health trends over time.
- Experiment with advanced predictive models like deep learning for enhanced accuracy.
- Collaborate with mental health professionals to design targeted interventions based on the findings.

---

## Author

Developed by Muhammad Bilal. Explore related projects on [Kaggle Profile](https://www.kaggle.com/bilalabdulmalik).
