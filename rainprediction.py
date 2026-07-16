import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

raw_df = pd.read_csv("AUSweather.csv")
# print(raw_df)

print(raw_df.info())

##drop all rows with N/A for raintoday and tomorow cause its imp info and no info on it doesnt help to predict and train model
raw_df.dropna(subset = ['RainToday' , 'RainTomorrow'], inplace = True)
print(raw_df.info())


sns.set_style("darkgrid")

matplotlib.rcParams["font.size"] = 14
matplotlib.rcParams["figure.figsize"] = (10, 6)
matplotlib.rcParams["figure.facecolor"] = "#00000000"
print(raw_df.Location.nunique())
fig = px.histogram(raw_df,x="Location" ,title="Location v/s Rainy Days" , color= 'RainToday')
# fig.show()

fig = px.histogram(raw_df, x='Temp3pm' , title = "temp at 3pm vs rain tomorrow" , color = 'RainTomorrow')
# fig.show()

fig = px.histogram(raw_df, x='RainTomorrow' , title = "Rain tomorrow vs rain today" , color = 'RainToday')
# fig.show()

fig = px.scatter(raw_df,x="MinTemp",y="MaxTemp", title="min v/s max temp", color = "RainToday")
# fig.show()

  ## similarly temp3pm vs humidity3pm using color="raintoday"

  ###When working with many many rows use this!!!!!

use_sample = False

sample_fraction = 0.1

if use_sample:
    raw_df = raw_df.sample(frac=sample_fraction).copy()

from sklearn.model_selection import train_test_split

train_val_df, test_df = train_test_split(raw_df , test_size=0.2 , random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.25 , random_state=42)

# print("train_df.shap:", train_df.shape)
# print("val_df.shap:", val_df.shape)
# print("test_df.shap:", test_df.shape)

##EXTRA CARE with dates!!!

plt.title("no. of rows per year")
sns.countplot(x=pd.to_datetime(raw_df.Date).dt.year)
# plt.show()

year = pd.to_datetime(raw_df.Date).dt.year

train_df = raw_df[year<2015]
val_df = raw_df[year==2015]
test_df = raw_df[year>2015]

##WE'RE ignoring the prev testtrainsplit data cause we now created anew one based on the year so remember this

print("train_df.shap:", train_df.shape)
print("val_df.shap:", val_df.shape)
print("test_df.shap:", test_df.shape)


##identify i/p o/p column

input_cols = list(train_df.columns)[1:-1]
target_col = 'RainTomorrow'

# print(input_cols)
# print(target_col)

train_inputs = train_df[input_cols].copy()
train_target = train_df[target_col].copy()

val_inputs = val_df[input_cols].copy()
val_target = val_df[target_col].copy()

test_inputs = test_df[input_cols].copy()
test_target = test_df[target_col].copy()

# print(train_inputs)

numeric_cols = train_inputs.select_dtypes(include=np.number).columns.to_list()
# print(numeric_cols)
categorical_cols = train_inputs.select_dtypes('object').columns.to_list()
# print(categorical_cols) 
##this works only because there are no strings if there were strings it wouldnt have worked





# print(train_inputs[numeric_cols].describe())
# print(train_inputs[categorical_cols].nunique())



##imputing...
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy= "mean")

# print(raw_df[numeric_cols].isna().sum())
# print(train_inputs[numeric_cols].isna().sum())


imputer.fit(raw_df[numeric_cols])
#cacl mean is stored in statistics and then after this use in the data set
# print(list(imputer.statistics_))

train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])

# print(train_inputs[numeric_cols].isna().sum())

# print(raw_df[numeric_cols].describe())
##LETS DO SCALING TO 0 to 1 or -1 to 1

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(raw_df[numeric_cols])

train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])

# print(train_inputs[numeric_cols].describe())


from sklearn.preprocessing import OneHotEncoder

encoder =OneHotEncoder(sparse_output= False , handle_unknown = 'ignore')
raw_df2 = raw_df[categorical_cols].fillna('Unknown')
encoder.fit(raw_df2[categorical_cols])
# print(encoder.categories_)

encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
# print(encoded_cols)


train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])

# print(train_inputs)

train_inputs.to_parquet('train_inputs.parquet')
val_inputs.to_parquet('val_inputs.parquet')
test_inputs.to_parquet('test_inputs.parquet')

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='liblinear')

model.fit(train_inputs[numeric_cols + encoded_cols] , train_target)
#logistic regression can work with categorical data in the target scction but not theinput section##

print(model.coef_.tolist())
print(model.intercept_)

# weight_df = pd.DataFrame(model.coef_)
# print(weight_df.to_string())

X_train = train_inputs[numeric_cols + encoded_cols]
X_val = val_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]

train_preds = model.predict(X_train)
train_probs = model.predict_proba(X_train)
print("ACCURACY IS : ", train_probs)

from sklearn.metrics import accuracy_score

print(accuracy_score(train_target,train_preds))


from sklearn.metrics import confusion_matrix

x= confusion_matrix(train_target,train_preds,normalize="true")
print(x)
## TOO MANY FALSE -VEs ...bad to predict if you should hold a baseball game in the open tomorrow
 

val_preds = model.predict(X_val)
val_probas = model.predict_proba(X_val)
print(val_probas)
#85% accuracy


test_preds = model.predict(X_test)
test_probs = model.predict_proba(X_test)
print(test_probs)
#84% accurate on test data

## 2 random ahh models 1) random yes/no 20 ONLY no always model

def random_guess(inputs):
    return np.random.choice(["No" , "Yes"], len(inputs))

def all_no (inputs):
    return np.full(len(inputs), "No")

print(accuracy_score(test_target , random_guess(X_test)))
print(accuracy_score(test_target , all_no(X_test)))


##Making predictions on single input

new_input ={'Date' : '2021-06-19',
            'Location' : 'Katherine',
            'MinTemp' : 23.2,
            'MaxTemp' : '33.2',
            'Rainfall' : 10.2,
            'Evaporation' : 4.2,
            'Sunshine' : np.nan,
            "WindGustDir" : 'NNW',
            'WindGustSpeed' : 52.0,
            'WindDir9am' : 'NW',
            'WindDir3pm' : 'NNE',
            'WindSpeed9am' : 13.0,
            'WindSpeed3pm' : 20.2,
            'Humidity9am' : 89.0,
            'Humidity3pm' : 58.0,
            'Pressure9am' : 1004.8,
            'Pressure3pm' : 1001.5,
            'Cloud9am' : 8.0,
            'Cloud3pm' : 5.0,
            'Temp9am' : 25.7,
            'Temp3pm' : 33,
            'RainToday' : 'Yes',
}

new_input_df = pd.DataFrame([new_input])

new_input_df[numeric_cols] = imputer.transform(new_input_df[numeric_cols])
new_input_df[numeric_cols] = scaler.transform(new_input_df[numeric_cols])
new_input_df[encoded_cols] = encoder.transform(new_input_df[categorical_cols])


X_new_input = new_input_df[numeric_cols + encoded_cols]

prediction = model.predict(X_new_input)
print("PREDICTION IS :" , prediction)

probab = model.predict_proba(X_new_input)
print("PROBAB IS : ", probab)

# MODEL ISNT VERY CONFIDENT ABOUT ITS PREDICTIONS ONLY 51.82% YES RAIN


def predict_input(single_input):
    input_df= pd.DataFrame([single_input])
    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    input_df[encoded_cols] = encoder.transform(input_df[categorical_cols])
    X_input = input_df[numeric_cols + encoded_cols]
    pred = model.predict(X_input)
    prob = model.predict_proba(X_input)
    return pred, prob


print(predict_input(new_input))



import joblib

aussie_rain = {
    'model' : model,
     'imputer' : imputer ,
      'scaler' : scaler,
       'encoder' : encoder ,
        'input_cols' :input_cols ,
         'target_col' : target_col,
          'numeric_cols' : numeric_cols,
           'categorical_cols' : categorical_cols ,
            'encoded_cols' : encoded_cols,
}

joblib.dump(aussie_rain , 'aussie_rain.joblib')


####LITERALLY SMALLER CODE
#1) import all the libraries
#2) download data and store in dataframe
#3) create testtrainsplit or test validate and train sets
#4) create inputs and targets
#5) identify numeric cols and categorical cols
#6) impute missing data
#7) scale numeric feature
#8) one hot encoding for categorical data
#9) save data to disc
#10) load data from disc
#############          THATS IT!!!! ##########


#######################################demo############################################

# # =========================
# # Imports
# # =========================
# import opendatasets as od
# import pandas as pd
# import numpy as np

# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


# # =========================
# # Download & Load Dataset
# # =========================
# od.download("https://www.kaggle.com/jsphyg/weather-dataset-rattle-package")

# raw_df = pd.read_csv(
#     "weather-dataset-rattle-package/weatherAUS.csv"
# )

# # Drop rows where target info is missing
# raw_df.dropna(subset=["RainToday", "RainTomorrow"], inplace=True)


# # =========================
# # Time-based Train / Val / Test split
# # =========================
# year = pd.to_datetime(raw_df.Date).dt.year

# train_df = raw_df[year < 2015]
# val_df   = raw_df[year == 2015]
# test_df  = raw_df[year > 2015]


# # =========================
# # Inputs & Target
# # =========================
# input_cols = list(train_df.columns)[1:-1]   # drop Date & target
# target_col = "RainTomorrow"

# train_inputs = train_df[input_cols].copy()
# train_targets = train_df[target_col].copy()

# val_inputs = val_df[input_cols].copy()
# val_targets = val_df[target_col].copy()

# test_inputs = test_df[input_cols].copy()
# test_targets = test_df[target_col].copy()


# # =========================
# # Identify Numeric & Categorical Columns
# # =========================
# numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
# categorical_cols = train_inputs.select_dtypes(include="object").columns.tolist()


# # =========================
# # Impute Missing Numeric Values
# # =========================
# imputer = SimpleImputer(strategy="mean")
# imputer.fit(raw_df[numeric_cols])

# train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
# val_inputs[numeric_cols]   = imputer.transform(val_inputs[numeric_cols])
# test_inputs[numeric_cols]  = imputer.transform(test_inputs[numeric_cols])


# # =========================
# # Scale Numeric Features
# # =========================
# scaler = MinMaxScaler()
# scaler.fit(raw_df[numeric_cols])

# train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
# val_inputs[numeric_cols]   = scaler.transform(val_inputs[numeric_cols])
# test_inputs[numeric_cols]  = scaler.transform(test_inputs[numeric_cols])


# # =========================
# # One-Hot Encode Categorical Features
# # =========================
# encoder = OneHotEncoder(
#     sparse_output=False,   # sklearn >=1.2
#     handle_unknown="ignore"
# )

# encoder.fit(raw_df[categorical_cols].fillna("Unknown"))

# encoded_cols = encoder.get_feature_names_out(categorical_cols)

# train_inputs[encoded_cols] = encoder.transform(
#     train_inputs[categorical_cols].fillna("Unknown")
# )
# val_inputs[encoded_cols] = encoder.transform(
#     val_inputs[categorical_cols].fillna("Unknown")
# )
# test_inputs[encoded_cols] = encoder.transform(
#     test_inputs[categorical_cols].fillna("Unknown")
# )

# # Optional but CLEAN: drop raw categorical columns
# train_inputs.drop(columns=categorical_cols, inplace=True)
# val_inputs.drop(columns=categorical_cols, inplace=True)
# test_inputs.drop(columns=categorical_cols, inplace=True)


# # =========================
# # Save Processed Data
# # =========================
# train_inputs.to_parquet("train_inputs.parquet")
# val_inputs.to_parquet("val_inputs.parquet")
# test_inputs.to_parquet("test_inputs.parquet")

# pd.DataFrame(train_targets).to_parquet("train_targets.parquet")
# pd.DataFrame(val_targets).to_parquet("val_targets.parquet")
# pd.DataFrame(test_targets).to_parquet("test_targets.parquet")


# # =========================
# # Load Back (Optional Check)
# # =========================
# train_inputs = pd.read_parquet("train_inputs.parquet")
# val_inputs   = pd.read_parquet("val_inputs.parquet")
# test_inputs  = pd.read_parquet("test_inputs.parquet")

# train_targets = pd.read_parquet("train_targets.parquet")[target_col]
# val_targets   = pd.read_parquet("val_targets.parquet")[target_col]
# test_targets  = pd.read_parquet("test_targets.parquet")[target_col]

# print("Train:", train_inputs.shape)
# print("Val:", val_inputs.shape)
# print("Test:", test_inputs.shape)
