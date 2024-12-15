import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def loan():
    st.markdown(
        "<h1 style='text-align: center;'>Analysis</h1>",
        unsafe_allow_html=True
    )
    # dummy1, title_col, dummy2 = st.columns([2, 5, 2])
    # with title_col:
    #     st.header('ANALYSIS')

    train_df = pd.read_csv("dataset/train.csv")
    train_df["Credit_History"] = train_df["Credit_History"].fillna(train_df["Credit_History"].mode()[0])

    train_df["LoanAmount"] = train_df["LoanAmount"].fillna(train_df["LoanAmount"].mean())

    train_df["Loan_Amount_Term"] = train_df["Loan_Amount_Term"].fillna(train_df["Loan_Amount_Term"].mode()[0])

    train_df["Gender"] = train_df["Gender"].fillna(train_df["Gender"].mode()[0])
    train_df["Married"] = train_df["Married"].fillna(train_df["Married"].mode()[0])
    train_df["Dependents"] = train_df["Dependents"].fillna(train_df["Dependents"].mode()[0])
    train_df["Self_Employed"] = train_df["Self_Employed"].fillna(train_df["Self_Employed"].mode()[0])
    # train_df.head(10)
    # train_df.info()
    # st.write("Shape of the dataset (rows, columns):", train_df.shape)
    # st.write("Descriptive Statistics of the dataset:")
    # st.dataframe(train_df.describe().T)
    # # Display descriptive statistics for object (categorical) columns
    # categorical_stats = train_df.describe(include='object')
    # st.write("Descriptive Statistics for Categorical Columns:")
    # st.dataframe(categorical_stats)
    #
    #
    # loan_status_count = train_df["Loan_Status"].value_counts(normalize=True)
    # st.write("Normalized Value Counts of Loan Status:")
    # st.write(loan_status_count)
    col1, col2, col3 = st.columns(3)
    with col1:
        # Gender count
        GenderAnalysis = train_df['Gender'].value_counts(dropna=False)

        # Convert to DataFrame for Plotly
        gender_df = GenderAnalysis.reset_index()
        gender_df.columns = ['Gender', 'Count']

        # Bar chart with Plotly Express
        fig = px.bar(
            gender_df,
            x='Gender',
            y='Count',
            color='Gender',
            title='Gender Analysis',
            color_discrete_sequence=px.colors.sequential.Plasma
        )

        st.plotly_chart(fig)

    with col2:
        # Dependents count
        DependentsAnalysis = train_df['Dependents'].value_counts(dropna=False)

        # Convert to DataFrame for Plotly
        dependents_df = DependentsAnalysis.reset_index()
        dependents_df.columns = ['Dependents', 'Count']

        # Bar chart
        fig = px.bar(
            dependents_df,
            x='Dependents',
            y='Count',
            color='Dependents',
            title='Dependents Analysis',
            color_discrete_sequence=px.colors.sequential.Plasma
        )

        st.plotly_chart(fig)

    with col3:
        # Education count
        EducationAnalysis = train_df['Education'].value_counts(dropna=False)

        education_df = EducationAnalysis.reset_index()
        education_df.columns = ['Education', 'Count']

        # Bar chart
        fig = px.bar(
            education_df,
            x='Education',
            y='Count',
            color='Education',
            title='Education Analysis',
            color_discrete_sequence=px.colors.sequential.Plasma
        )

        st.plotly_chart(fig)

    cola, colb, colc = st.columns(3)

    with cola:
        # 'Self_Employed' feature count
        Self_EmployedAnalysis = train_df['Self_Employed'].value_counts(dropna=False)
        self_employed_df = Self_EmployedAnalysis.reset_index()
        self_employed_df.columns = ['Self_Employed', 'Count']

        # Bar chart
        fig = px.bar(self_employed_df,
                     x='Self_Employed',
                     y='Count',
                     color='Self_Employed',
                     title='Self Employed Analysis',
                     color_discrete_sequence=px.colors.sequential.Plasma)

        st.plotly_chart(fig)

    with colb:
        # 'Credit_History' feature count
        Credit_HistoryAnalysis = train_df['Credit_History'].value_counts(dropna=False)

        credit_history_df = Credit_HistoryAnalysis.reset_index()
        credit_history_df.columns = ['Credit_History', 'Count']

        # Bar chart
        fig = px.bar(credit_history_df,
                     x='Credit_History',
                     y='Count',
                     color='Credit_History',
                     title='Credit History Analysis',
                     color_discrete_sequence=px.colors.sequential.Plasma)

        st.plotly_chart(fig)

    with colc:
        # 'Property_Area' feature count
        Property_AreaAnalysis = train_df['Property_Area'].value_counts(dropna=False)

        property_area_df = Property_AreaAnalysis.reset_index()
        property_area_df.columns = ['Property_Area', 'Count']

        # Bar chart
        fig = px.bar(property_area_df,
                     x='Property_Area',
                     y='Count',
                     color='Property_Area',
                     title='Property Area Analysis',
                     color_discrete_sequence=px.colors.sequential.Plasma)

        st.plotly_chart(fig)

    cold, cole, colf = st.columns(3)

    with cold:
        # 'Loan_Status' feature count
        Loan_StatusAnalysis = train_df['Loan_Status'].value_counts(dropna=False)

        loan_status_df = Loan_StatusAnalysis.reset_index()
        loan_status_df.columns = ['Loan_Status', 'Count']

        # Bar chart
        fig = px.bar(loan_status_df,
                     x='Loan_Status',
                     y='Count',
                     color='Loan_Status',
                     title='Loan Status Analysis',
                     color_discrete_sequence=px.colors.sequential.Plasma)

        st.plotly_chart(fig)

    with cole:
        # cross-tabulation of 'Education' and 'Loan_Status'
        education_loan_status_df = pd.crosstab(train_df['Education'], train_df['Loan_Status'])

        # bar chart
        fig = px.bar(education_loan_status_df,
                     x=education_loan_status_df.index,
                     y=education_loan_status_df.columns,
                     title='Education Status VS Loan Status',
                     labels={'x': 'Education', 'y': 'Count'},
                     barmode='group',  # Groups bars for each loan status
                     color=education_loan_status_df.columns,  # Color bars based on Loan_Status
                     color_discrete_sequence=px.colors.qualitative.Set1)  # Color palette

        st.plotly_chart(fig)
    with colf:
        # correlation matrix
        correlation_matrix = train_df.corr(numeric_only=True)

        # Create the heatmap using Seaborn and Matplotlib
        plt.figure(figsize=(15, 7.5))
        sns.heatmap(correlation_matrix, annot=True, cmap='flare')

        plt.title('Correlation Matrix')

        st.pyplot(plt)
    colg, colh, coli = st.columns(3)
    # Unique figure for 'Education' and 'Loan_Status'
    with colg:
        education_loan_status_df = pd.crosstab(train_df['Education'], train_df['Loan_Status']).reset_index()
        fig = px.bar(
            education_loan_status_df,
            x='Education',
            y=['N', 'Y'],
            title='Education vs Loan Status',
            labels={'Education': 'Education', 'value': 'Count', 'variable': 'Loan Status'},
            barmode='group',
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        st.plotly_chart(fig)

    # Unique figure for 'Credit_History' and 'Loan_Status'
    with colh:
        credit_history_loan_status_df = pd.crosstab(train_df['Credit_History'], train_df['Loan_Status']).reset_index()
        fig = px.bar(
            credit_history_loan_status_df,
            x='Credit_History',
            y=['N', 'Y'],
            title='Credit History vs Loan Status',
            labels={'Credit_History': 'Credit History', 'value': 'Count', 'variable': 'Loan Status'},
            barmode='group',
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        st.plotly_chart(fig)

    # Unique figure for 'Property_Area' and 'Loan_Status'
    with coli:
        property_area_loan_status_df = pd.crosstab(train_df['Property_Area'], train_df['Loan_Status']).reset_index()
        fig = px.bar(
            property_area_loan_status_df,
            x='Property_Area',
            y=['N', 'Y'],
            title='Property Area vs Loan Status',
            labels={'Property_Area': 'Property Area', 'value': 'Count', 'variable': 'Loan Status'},
            barmode='group',
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        st.plotly_chart(fig)


    colj, colk = st.columns(2)
    with colj:

        # value counts for the 'Married' feature
        MarriedAnalysis = train_df['Married'].value_counts(dropna=False)

        # Create a pie chart using Plotly
        fig = px.pie(
            names=MarriedAnalysis.index,  # Labels for the pie chart
            values=MarriedAnalysis.values,  # Values for the pie chart
            title='Marital Status Distribution',  # Title of the pie chart
            color=MarriedAnalysis.index,  # Color the segments based on the labels
            color_discrete_sequence=px.colors.sequential.Plasma,  # Color palette
            hole=0,  # This is to make a full pie chart (hole=0 means no hole)
            labels={'Married': 'Status'}  # Labels for the chart
        )

        # Display the interactive plot in Streamlit
        st.plotly_chart(fig)

    with colk:
        # st.write("Ratio of People with Income Above Average to Below Average")
        # Calculate the average income
        average_income = train_df['ApplicantIncome'].mean()
        # st.write(f"The Average Income: {average_income:.2f}")

        # Count incomes higher and lower than average
        above_average_count = (train_df['ApplicantIncome'] > average_income).sum()
        below_average_count = (train_df['ApplicantIncome'] <= average_income).sum()

        # Calculate ratio and display results
        ratio = above_average_count / below_average_count
        # st.write(f"The ratio of people with income above average to below average: {ratio * 100:.2f}")
        # st.write(f"Number of people with income above the average: {above_average_count}")
        # st.write(f"Number of people with income below the average: {below_average_count}")

        # Create a DataFrame for plotting with Plotly
        data = {
            'Income Category': ['Above Average', 'Below Average'],
            'Count': [above_average_count, below_average_count]
        }
        df_income = pd.DataFrame(data)

        # bar chart
        fig = px.bar(
            df_income,
            x='Income Category',
            y='Count',
            title='Ratio of People with Income Above Average to Below Average',
            labels={'Income Category': 'Income Status', 'Count': 'Count'},
            color='Income Category',  # Color bars by income status
            color_discrete_sequence=px.colors.sequential.Cividis  # Color palette
        )

        st.plotly_chart(fig)




if __name__ == "__main__":
    st.set_page_config(
        page_title="Analysis",
        page_icon="chart_with_upwards_trend",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    loan()