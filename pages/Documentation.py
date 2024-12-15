import streamlit as st


def doc():
    # st.header("Code Documentation")
    st.markdown(
        "<h1 style='text-align: center;'>Code Documentation</h1>",
        unsafe_allow_html=True
    )
    st.write("The dataset used in this project was obtained from kaggle.com : https://www.kaggle.com/datasets/krishnaraj30/finance-loan-approval-prediction-data/data")
    st.write("This dataset consists of key demographic and financial information about loan applicants, including factors such as gender, marital status, income, credit history, loan amount, and property location.")
    st.write("The goal of this analysis is to build a predictive model that can assess the eligibility of loan seekers by analyzing historical data. By doing so, the model will help financial institutions make informed decisions, improving approval accuracy and reducing the risk of loan defaults.")

    st.subheader("Importation of Libraries and Data Injestion")

    st.code("""
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings('ignore')
    
    
    train_df = pd.read_csv("/content/drive/MyDrive/datasets/train.csv")
    test_df = pd.read_csv("/content/drive/MyDrive/datasets/test.csv")
    
    train_df.head(30)   
    
    """)

    st.write("Dataset information")
    st.code("""
    train_df.info()
    """)

    st.write("Dataset shape")
    st.code("""
    train_df.shape
    """)

    st.write("Dataset summary for numerical features")
    st.code("""
    train_df.describe().T
    """)

    st.write("Dataset summary for categorical features")
    st.code("""
    train_df.describe(include=object)
    """)

    st.subheader("Data cleaning")

    st.write("Checking for missing values")
    st.code("""
    train_df.isna().sum()
    """)
    st.write("Handling missing values")
    st.write("For credit history, gender, married, dependents, self employed and loan term amount, mode imputing has been used.")
    st.write("For loan amount, mean imputation had been used.")
    st.code("""
    train_df["Credit_History"] = train_df["Credit_History"].fillna(train_df["Credit_History"].mode()[0])
    
    train_df["LoanAmount"] = train_df["LoanAmount"].fillna(train_df["LoanAmount"].mean())
    
    train_df["Loan_Amount_Term"] = train_df["Loan_Amount_Term"].fillna(train_df["Loan_Amount_Term"].mode()[0])
    
    train_df["Gender"] = train_df["Gender"].fillna(train_df["Gender"].mode()[0])
    train_df["Married"] = train_df["Married"].fillna(train_df["Married"].mode()[0])
    train_df["Dependents"] = train_df["Dependents"].fillna(train_df["Dependents"].mode()[0])
    train_df["Self_Employed"] = train_df["Self_Employed"].fillna(train_df["Self_Employed"].mode()[0])
    """)

    st.subheader("Encoding categorical features to numerical features")
    st.write("Used label encoding technique to encode categorical features.")
    st.code("""
    from sklearn.preprocessing import LabelEncoder
    
    train_df.drop(['Loan_ID'], axis = 1 , inplace = True)# dropped since it's not being used
    
    # Create a LabelEncoder object
    label_encoder = LabelEncoder()
    train_df["Gender"] = label_encoder.fit_transform(train_df["Gender"])
    train_df["Married"] = label_encoder.fit_transform(train_df["Married"])
    train_df["Education"] = label_encoder.fit_transform(train_df["Education"])
    train_df["Self_Employed"] = label_encoder.fit_transform(train_df["Self_Employed"])
    train_df["Property_Area"] = label_encoder.fit_transform(train_df["Property_Area"])
    train_df["Loan_Status"] = label_encoder.fit_transform(train_df["Loan_Status"])
    train_df.head(30)
    """)
    st.write("Type converting the dependents column")
    st.code("""
    import random
    
    train_df.loc[train_df["Dependents"] == "3+", "Dependents"] = 3
    train_df["Dependents"].value_counts()
    
    train_df["Dependents"] = train_df["Dependents"].astype(int)
    train_df.info()
    """)

    st.subheader("Handling outliers")

    st.write("Outlier detection and handling using the Interquartile Range (IQR) method. It replaces values that are outside the lower and upper bounds with the respective bound values to mitigate the impact of outliers.")
    st.code("""
    import numpy as np
    
    # IQR Scaling
    Q1 = train_df.astype(np.float32).quantile(0.25)
    Q3 = train_df.astype(np.float32).quantile(0.75)
    
    # Calculate the Interquartile Range (IQR)
    IQR = Q3 - Q1
    
    # Calculate lower and upper bounds for outlier detection
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Fill outliers with lower or upper bound values
    train_df = train_df.apply(lambda x: np.where(x < lower_bound[x.name], lower_bound[x.name],
                                                  np.where(x > upper_bound[x.name], upper_bound[x.name], x)))
    
    # printing shape
    print(train_df.shape)
    """)

    st.subheader("Model training")
    st.write("Importation of libraries")
    st.code("""
    from sklearn import preprocessing
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.metrics import accuracy_score,recall_score, f1_score ,classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import ConfusionMatrixDisplay
    
    import warnings
    
    warnings.filterwarnings('ignore')
    """)

    st.write("Separating the predictors (features) and the target variable (label) for use in a machine learning model.")
    st.code("""
    X = train_df.drop(columns=['Loan_Status'])
    y = train_df['Loan_Status']
    """)

    st.write("Scaling and standardization of data")
    st.code("""
    scale=StandardScaler()
    X_scaled=scale.fit_transform(X)
    """)

    st.write("Splitting into training and testing sets")
    st.code("""
    X_train , X_test ,y_train ,y_test=train_test_split(X_scaled,y,test_size=0.3,random_state=42)
    """)

    st.write("Performing class balancing using SMOTE technique")
    st.code("""
    from imblearn.over_sampling import SMOTE
    
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    # Check the distribution of the classes after resampling
    print("Original dataset shape:", y_train.value_counts())
    print("Resampled dataset shape:", y_resampled.value_counts())
    """)

    st.write("Defining the algorithms to be used")
    st.code("""
    log_reg=LogisticRegression()
    rf_clf=RandomForestClassifier()
    svc=SVC()
    dt_clf = DecisionTreeClassifier()
    
    columns=['LogisticRegression','RandomForestClassifier','SVC', 'DecisionTreeClassifier']
    train_score=[]
    test_score=[]
    rec_score=[]
    f_score=[]
    """)

    st.write("Training the model")
    st.code("""
    def all(model):
        model.fit(X_resampled, y_resampled)
        y_pred=model.predict(X_test)
        accuracy_test=accuracy_score(y_pred,y_test)*100
        accuracy_train=model.score(X_resampled,y_resampled)*100
        recall_result=recall_score(y_pred,y_test)*100
        f1_result=f1_score(y_pred,y_test)*100
    
        test_score.append(accuracy_test)
        train_score.append(accuracy_train)
        rec_score.append(recall_result)
        f_score.append(f1_result)
    
        print('Accuracy after train the model is :',accuracy_train)
        print('Accuracy after test the model is :',accuracy_test)
        print('Result recall score is :',recall_result)
        print('Result F1 score is :',f1_result)
    """)

    st.write("Printing of Logistic Regression results")
    st.code("""
    print("Logistic Regression")
    all(log_reg)
    """)

    st.write("Printing of Random Forest results")
    st.code("""
    print("Random Forest Classifier")
    all(rf_clf)
    """)

    st.write("Printing of SVC results")
    st.code("""
    print("Support Vector Classifier")
    all(svc)
    """)

    st.write("Printing of Decision Tree Classifier")
    st.code("""
    print("Decision Tree Classifier")
    all(dt_clf)
    """)

    st.write("Visualizing model performance")
    st.code("""
    plt.figure(figsize=(15, 8))
    bar_width = 0.2
    xpos=np.arange(len(columns))
    bars1=plt.bar(xpos - 0.3, train_score, width=bar_width, label="train_score",color='lightblue')
    bars2=plt.bar(xpos - 0.1, test_score, width=bar_width, label="test_score",color='lightpink')
    bars3=bars=plt.bar(xpos + 0.1, rec_score, width=bar_width, label="recall_score",color='lightgreen')
    bars4=plt.bar(xpos + 0.3, f_score, width=bar_width, label="f1_score",color='plum')
    
    plt.xticks(xpos, columns)
    plt.legend()
    plt.xlabel("Models",fontsize=15)
    plt.ylabel("Scores",fontsize=15)
    plt.title("Model Performance Comparison",fontsize=30)
    plt.show()
    """)

    st.write("Visualizing feature importance for logistic regression")
    st.code("""
    log_reg.fit(X,y)
    coefficients = log_reg.coef_[0]
    # Create a DataFrame with the feature importances
    feature_importances_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': coefficients
    })
    feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)
    bars=sns.barplot(x='Importance',y='Feature',data=feature_importances_df,color='lightpink')
    bars.patches[0].set_hatch('/')
    plt.ylabel('Feature',fontsize=12)
    plt.xlabel('Importance',fontsize=12)
    plt.title('Feature Importances from Logistic Regression Model ',fontsize=15)
    plt.xticks(rotation=90)
    plt.show()
    """)

    st.write("Visualizing feature importance for random forest classifier")
    st.code("""
    rf_clf.fit(X,y)
    feature_importances = rf_clf.feature_importances_
    
    # Create a DataFrame with the feature importances
    feature_importances_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    })
    plt.figure(figsize=(11, 8))
    
    feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)
    bars=sns.barplot(x='Importance',y='Feature',data=feature_importances_df,color='lightpink')
    bars.patches[0].set_hatch('/')
    plt.ylabel('Feature',fontsize=15)
    plt.xlabel('Importance',fontsize=15)
    plt.title('Feature Importances from Random Forest Model',fontsize=18)
    plt.xticks(rotation=90)
    plt.show()
    """)

    st.write("Visualizing feature importance for SVC classifier")
    st.code("""
    svc.fit(X,y)
    coefficients = log_reg.coef_[0]
    # Create a DataFrame with the feature importances
    feature_importances_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': coefficients
    })
    feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)
    bars=sns.barplot(x='Importance',y='Feature',data=feature_importances_df,color='lightpink')
    bars.patches[0].set_hatch('/')
    plt.ylabel('Feature',fontsize=12)
    plt.xlabel('Importance',fontsize=12)
    plt.title('Feature Importances from SVC Model ',fontsize=15)
    plt.xticks(rotation=90)
    plt.show()
    """)

    st.write("Visualizing feature importance for Decision Tree classifier")
    st.code("""
    dt_clf.fit(X,y)
    coefficients = log_reg.coef_[0]
    # Create a DataFrame with the feature importances
    feature_importances_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': coefficients
    })
    feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)
    bars=sns.barplot(x='Importance',y='Feature',data=feature_importances_df,color='lightpink')
    bars.patches[0].set_hatch('/')
    plt.ylabel('Feature',fontsize=12)
    plt.xlabel('Importance',fontsize=12)
    plt.title('Feature Importances from Decision Tree classifier Model ',fontsize=15)
    plt.xticks(rotation=90)
    plt.show()
    """)

    st.subheader("Confusion Matrix")
    st.code("""
    def cm(model):
        y_pred=model.predict(X_test)
    
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
    
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model.classes_)
        disp.plot(cmap='Blues')
        plt.title('Confusion Matrix')
        print(model)
        plt.show()
        
    cm(log_reg)
    cm(rf_clf)
    cm(svc)
    cm(dt_clf)
    
    """)

    st.subheader("Model saving")
    st.code("""
    import pickle
    
    with open("model.pkl", "wb") as file:
        pickle.dump(SVC, file)
    """)

if __name__ == "__main__":
    st.set_page_config(
            page_title="Documentation",
            page_icon="📝",
            layout="wide",
            initial_sidebar_state="expanded"
    )

    doc()