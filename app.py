import pandas as pd
import numpy as np 
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from PIL import Image

not_user = Image.open('grandma.jpg')
is_user = Image.open('linkedin_user.jpg')

st.title('LinkedIn User Predictions')

def run_status():
    latest_iteration = st.empty()
    bar = st.progress(0)
    for i in range(100):
        latest_iteration.text(f'Percent Complete {i+1}')
        bar.progress(i+1)
        time.sleep(0.1)
        st.empty()

st.subheader('A Logistic Regression Prediction')



#clean_sm function to convert columns to binary
def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return(x)


@st.cache
def load_data():
    ss = pd.read_csv('social_media_usage.csv')
    ss["sm_li"] = ss["web1h"].apply(clean_sm) #create variable that we are going to predict
    ss = ss[['web1h', 'sm_li', 'income', 'educ2', 'par', 'marital', 'gender', 'age']] #select only the variables we are interested in
    ss["par"] = ss["par"].apply(clean_sm) #make parent binary, where it equals 1 if someone is a parent and 0 otherwise
    ss["marital"] = np.where(ss["marital"] == 1, 1, 0) #make marital binary, where it equals 1 if someone is married and 0 otherwise
    ss["gender"] = np.where(ss["gender"] == 2, 1, 0) #make gender binary, where it equals 1 if someone is female and 0 otherwise
    ss = ss.rename(columns = {'gender': 'female'}) #rename gender to be female
    ss["income"] = np.where(ss["income"] > 9, np.nan, ss["income"]) #convert any income values above 9 to NaN
    ss["educ2"] = np.where(ss["educ2"] > 8, np.nan, ss["educ2"]) #convert any education values above 8 to NaN
    ss["age"] = np.where(ss["age"] > 98, np.nan, ss["age"]) #convert any age values above 98 to NaN
    ss = ss.dropna() #drop any missing values
    ss = ss.drop(columns = 'web1h') #drop original web1h column
    return ss


ss = load_data()


st.sidebar.subheader('User Details')
# Sidebar Options:
params={
   'age' : st.sidebar.slider('Age', 18, 98), #slider from 18 to max age in dataset
   'marital' : st.sidebar.selectbox('Married', ('Yes', 'No')), #1 is married, 0 is not married
   'female' : st.sidebar.selectbox('Gender', ('Female', 'Male')),  #1 is female, 0 is male
   'par' : st.sidebar.selectbox('Parent', ('Yes', 'No')),  #1 is parent, 0 is not parent
   'income': st.sidebar.selectbox('Income', ("Less than $10,000",
                                                 "10 to under $20,000",
                                                  "20 to under $30,000",
                                                  "30 to under $40,000",
                                                  "40 to under $50,000",
                                                  "50 to under $75,000",
                                                  "75 to under $100,000",
                                                  "100 to under $150,000",
                                                  "$150,000 or more")),
   'educ2' : st.sidebar.selectbox('Education', ("Less than high school",
                                                    "High school incomplete",
                                                    "High school graduate",
                                                    "Some college, no degree",
                                                    "Two-year associate degree",
                                                    "Four-year college degree",
                                                    "Some postgraduate",
                                                    "Postgraduate degree"))
}

marital_param = 1 if params['marital'] == 'Yes' else 0
female_param = 1 if params['female'] == 'Female' else 0
parent_param = 1 if params['par'] == 'Yes' else 0


if params['income'] == "Less than $10,000":
    income_param = 1
elif params['income'] == "10 to under $20,000":
    income_param = 2
elif params['income'] == "20 to under $30,000":
    income_param = 3
elif params['income'] == "30 to under $40,000":
    income_param = 4
elif params['income'] == "40 to under $50,000":
    income_param = 5
elif params['income'] == "50 to under $75,000":
    income_param = 6
elif params['income'] == "75 to under $100,000":
    income_param = 7
elif params['income'] == "100 to under $150,000":
    income_param = 8
else:
    income_param = 9


if params['educ2'] == "Less than high school":
    educ_param = 1
elif params['educ2'] == "High school incomplete":
    educ_param = 2
elif params['educ2'] == "High school graduate":
    educ_param = 3
elif params['educ2'] == "Some college, no degree":
    educ_param = 4
elif params['educ2'] == "Two-year associate degree":
    educ_param = 5
elif params['educ2'] == "Four-year college degree":
    educ_param = 6
elif params['educ2'] == "Some postgraduate":
    educ_param = 7
else:
    educ_param = 8


# Target (y) and feature(s) selection (X)

y = ss['sm_li']
x = ss[["income", "educ2", "par", "marital", "female", "age"]]
# Split data into training and test set
x_train, x_test, y_train, y_test = train_test_split(x,
                                                y,
                                                stratify=y,       # same number of target in training & test set
                                                test_size=0.2,    # hold out 20% of data for testing
                                                random_state=987) # set for reproducibility
lr = LogisticRegression() # Initialize algorithm 
lr.fit(x_train, y_train) # Fit algorithm to training data
y_pred = lr.predict(x_test) # Make predictions using the model and the testing data



def run_data():
    newdata = [[
        income_param,
        educ_param,
        parent_param,
        marital_param,
        female_param,
        params['age']
        ]]

    newdata_df = pd.DataFrame(newdata, columns = ['income', 'educ2', 'par', 'marital', 'female', 'age'])

    #newdata_df

    newdata_df["prediction_on_linkedin"] = lr.predict(newdata_df)
    outcome = newdata_df.iloc[0]['prediction_on_linkedin']
    #st.write(f'outcome: {outcome}')
    #print output
    if outcome == 1:
        st.markdown('Given your parameters, the model is predicting that you **_are_** a linkedin user')
        st.image(is_user)
    else:
        st.markdown('Given your parameters, the model is predicting that you are **_not_** a linkedin user')
        st.image(not_user)



btn = st.sidebar.button("Predict")
if btn:
    run_data()
else:
    pass




