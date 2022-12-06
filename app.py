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
   'marital' : st.sidebar.selectbox('Married', (1, 0)), #1 is married, 0 is not married
   'female' : st.sidebar.selectbox('Gender', ('1', '0')),  #1 is female, 0 is male
   'par' : st.sidebar.selectbox('Parent', ('1', '0')),  #1 is parent, 0 is not parent
   'income': st.sidebar.selectbox('Income', (1, 2, 3, 4, 5, 6, 7, 8, 9)),
   'educ2' : st.sidebar.selectbox('Education', (1, 2, 3, 4, 5, 6, 7, 8))
}


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
        params['income'],
        params['educ2'],
        params['par'],
        params['marital'],
        params['female'],
        params['age']
        ]]

    newdata_df = pd.DataFrame(newdata, columns = ['income', 'educ2', 'par', 'marital', 'female', 'age'])

    newdata_df

    newdata_df["prediction_on_linkedin"] = lr.predict(newdata_df)
    outcome = newdata_df.iloc[0]['prediction_on_linkedin']
    #st.write(f'outcome: {outcome}')
    #print output
    if outcome == 1:
        st.write('Given your parameters, the model is predicting that you are a linkedin user')
        st.image(is_user)
    else:
        st.write('Given your parameters, the model is predicting that you are not a linkedin user')
        st.image(not_user)



btn = st.sidebar.button("Predict")
if btn:
    run_data()
else:
    pass




