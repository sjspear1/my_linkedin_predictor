import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#load file
ss = pd.read_csv(r'C:\Users\sspea\OneDrive\Documents\Georgetown Courses\Programming II - Python\Final_Project\social_media_usage.csv')

#clean_sm function to convert columns to binary
def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return(x)

ss["sm_li"] = ss["web1h"].apply(clean_sm)

ss = ss[['web1h', 'sm_li', 'income', 'educ2', 'par', 'marital', 'gender', 'age']]

ss["par"] = ss["par"].apply(clean_sm) #make parent binary, where it equals 1 if someone is a parent and 0 otherwise

ss["marital"] = np.where(ss["marital"] == 1, 1, 0) #make marital binary, where it equals 1 if someone is married and 0 otherwise

ss["gender"] = np.where(ss["gender"] == 2, 1, 0) #make gender binary, where it equals 1 if someone is female and 0 otherwise
ss = ss.rename(columns = {'gender': 'female'}) #rename gender to be female

#convert any income values above 9 to NaN
ss["income"] = np.where(ss["income"] > 9, np.nan, ss["income"])

#convert any education values above 8 to NaN
ss["educ2"] = np.where(ss["educ2"] > 8, np.nan, ss["educ2"])

#convert any age values above 98 to NaN
ss["age"] = np.where(ss["age"] > 98, np.nan, ss["age"])


#drop any missing values
ss = ss.dropna()

#drop original web1h column
ss = ss.drop(columns = 'web1h')



# Target (y) and feature(s) selection (X)
y = ss["sm_li"]
x = ss[["income", "educ2", "par", "marital", "female", "age"]]


# Split data into training and test set
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    stratify=y,       # same number of target in training & test set
                                                    test_size=0.2,    # hold out 20% of data for testing
                                                    random_state=987) # set for reproducibility



# Initialize algorithm 
lr = LogisticRegression()


# Fit algorithm to training data
lr.fit(x_train, y_train)


# Make predictions using the model and the testing data
y_pred = lr.predict(x_test)

#create new data to make a prediction
newdata = pd.DataFrame({
    "income": [8],
    "educ2": [7],
    "par": [0],
    "marital": [1],
    "female": [1],
    "age": [42]
})

print(newdata)



# Use model to make predictions
newdata["prediction_on_linkedin"] = lr.predict(newdata)

#check the prediction
print(newdata)