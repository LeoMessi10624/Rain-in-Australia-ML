import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

sns.set_style("darkgrid")

matplotlib.rcParams["font.size"] = 14
matplotlib.rcParams["figure.figsize"] = (10, 6)
matplotlib.rcParams["figure.facecolor"] = "#00000000"

raw_df = pd.read_csv("weatherAus.csv")
# print(raw_df.info())

raw_df.dropna(subset=["RainTomorrow"] , inplace = True)

# plt.title("No. of rows per Year")
# sns.countplot(x=pd.to_datetime(raw_df.Date).dt.year)
# plt.show()

year = pd.to_datetime(raw_df.Date).dt.year

train_df = raw_df[year<2015]
val_df = raw_df[year==2015]
test_df = raw_df[year>2015]

# print("train_df.shap:", train_df.shape)
# print("val_df.shap:", val_df.shape)
# print("test_df.shap:", test_df.shape)


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


numeric_cols = train_inputs.select_dtypes(include=np.number).columns.to_list()
# print(numeric_cols)
categorical_cols = train_inputs.select_dtypes('object').columns.to_list()
# print(categorical_cols) 


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


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(raw_df[numeric_cols])

train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])

# print(train_inputs[numeric_cols].describe())


from sklearn.preprocessing import OneHotEncoder


train_inputs[categorical_cols] = train_inputs[categorical_cols].fillna("Unknown")
val_inputs[categorical_cols] = val_inputs[categorical_cols].fillna("Unknown")
test_inputs[categorical_cols] = test_inputs[categorical_cols].fillna("Unknown")

encoder =OneHotEncoder(sparse_output= False , handle_unknown = 'ignore').fit(train_df[categorical_cols])
# print(encoder.categories_)

encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
# print(encoded_cols)


train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])


X_train = train_inputs[numeric_cols + encoded_cols]
X_val = val_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]


from sklearn.tree import DecisionTreeClassifier
## THIS IS A CLASSIFICATION TYPE PROBLEM!
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train , train_target)

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# train_preds = model.predict(X_train)
# print(train_preds)
# print(pd.value_counts(train_preds))

# print(accuracy_score(train_preds,train_target))

# train_probs = model.predict_proba(X_train)
# print(train_probs)
# print("ACCURACY IS : ", train_probs)

# y = model.score(X_val , val_target)
# print(y)
## SEEMS LIKE MODEL HAS LEARNED THE TRAINING EXS PERFECTLY BUT DOESNT GENRALISE WELL TO PREVIOUSLY UNSEEN EXS. 
## THIS IS CALLED AS "OVERFITTING" AND REDUCING OVERFITTING IS ONE OF THE MOST IMP PARTS OF ML MODEL

from sklearn.tree import plot_tree,export_text

# plt.figure(figsize=(80,20))
# plot_tree(model,feature_names=X_train.columns , max_depth=2, filled= True)
# plt.show()
# print(model.tree_max_depth)

tree_text = export_text(model, max_depth=10, feature_names=list(X_train.columns))
# print(tree_text[:5000])

# print(model.feature_importances_)

importance_df = pd.DataFrame({
    'feature' : X_train.columns,
    'importance' : model.feature_importances_
}).sort_values('importance' , ascending=False)

# print(importance_df.head(10))

# plt.title("Feature importance")
# sns.barplot(data=importance_df.head(10) , x='importance' , y='feature')
# plt.show()


## HYPERPARAMETER TUNING and OVERFITTING

# model = DecisionTreeClassifier(max_depth=3,random_state=42)
# model.fit(X_train , train_target)

# z = model.score(X_train , train_target)
# print(z)

# q = model.score(X_val , val_target)
# print(q)

# print(export_text(model , feature_names = list(X_train.columns)))

# print(model.max_features_)

def max_depth_error (md):
    model = DecisionTreeClassifier(max_depth=md , random_state=42)
    model.fit(X_train , train_target)
    train_error = 1 - model.score(X_train ,train_target)
    val_error = 1 - model.score(X_val ,val_target)
    return{"Max Depth" : md, "Training Error" : train_error , "Validation Error" : val_error}

##TAKES A LIL TIME TO CALCULATE HAVE PATIENCE BITCH
# errors_df = pd.DataFrame([max_depth_error(md) for md in range (1,21)])
# print(errors_df.to_string())


# plt.figure()
# plt.plot(errors_df["Max Depth"], errors_df["Training Error"])
# plt.plot(errors_df["Max Depth"], errors_df["Validation Error"])
# plt.title("Training vs Validation Error")
# plt.xticks(range(0,21,2))
# plt.xlabel("Max Depth")
# plt.ylabel("Prediction Error (1-Accuracy)")
# plt.legend(["Training" , "Validation"])
# plt.show()


# model = DecisionTreeClassifier(max_depth=7,random_state=42)
# model.fit(X_train , train_target)


# z = model.score(X_train , train_target)
# print(z)

# q = model.score(X_val , val_target)
# print(q)


# print(model.get_n_leaves)



# model = DecisionTreeClassifier(max_leaf_nodes=128 , random_state=42)
# model.fit(X_train , train_target)

# z = model.score(X_train , train_target)
# print(z)

# q = model.score(X_val , val_target)
# print(q)

# c = model.tree_.max_depth
# print(c)




# ### ENSEMBLE AND RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier

# model = RandomForestClassifier(n_jobs=-1 , random_state=42)
# model.fit(X_train , train_target)

# z = model.score(X_train , train_target)
# print(z)

# q = model.score(X_val , val_target)
# print(q)

# train_probs = model.predict(X_train)
# print(train_probs)

# print(model.estimators_[0])

# plt.figure(figsize=(80,20))
# plot_tree(model.estimators_[0] , max_depth=2 , feature_names=X_train.columns , filled =True , rounded=True)
# plt.show()

# plt.figure(figsize=(80,20))
# plot_tree(model.estimators_[15] , max_depth=2 , feature_names=X_train.columns , filled =True , rounded=True)
# plt.show()


importance_df = pd.DataFrame({
    'feature' : X_train.columns,
    'importance' : model.feature_importances_
}).sort_values('importance' , ascending=False)

print(importance_df.head(10))




base_model = RandomForestClassifier(n_jobs=-1 , random_state=42)
base_model.fit(X_train , train_target)

base_train_acc = base_model.score(X_train, train_target)
base_val_acc = base_model.score(X_val , val_target)

# print(base_train_acc)
# print(base_val_acc)



#default n_estimators is 100.
# model = RandomForestClassifier(n_jobs=-1 , random_state=42 , n_estimators=10)
# model.fit(X_train,train_target)
# d = model.score(X_train,train_target)
# f = model.score(X_val,val_target)
# print(d)
# print(f)




# n estimators increases randomness and not complexity.
# model = RandomForestClassifier(n_jobs=-1 , random_state=42 , n_estimators=1000)
# model.fit(X_train,train_target)
# d = model.score(X_train,train_target)
# f = model.score(X_val,val_target)
# print(d)
# print(f)




def test_param(**param):
    model = RandomForestClassifier(random_state=42, n_jobs=1,**param).fit(X_train,train_target)
    return model.score(X_train , train_target), model.score(X_val , val_target)


# print(test_param(max_depth = 5 , max_leaf_nodes = 1024 , n_estimators = 200))
# print(test_param(max_depth = 26))
# print(test_param(max_leaf_nodes = 2**5))
# print(test_param(max_leaf_nodes = 2**20))


# print(test_param(max_features= 'log2'))
# print(test_param(max_features= 3))
# print(test_param(max_features= 6))
# print(test_param(max_features= 20))

# print(test_param(min_samples_split =5, min_samples_leaf = 2 ))
# print(test_param(min_samples_split =100, min_samples_leaf = 60 ))

# print(test_param(min_impurity_decrease = 1e-7))
# print(test_param(min_impurity_decrease = 1e-2))
# print(test_param(min_impurity_decrease = 1e-6))



# print(test_param(bootstrap = False))
# print(test_param(max_samples=0.9))


# print(test_param(class_weight ='balanced'))
# print(test_param(class_weight ={'No' : 1 , 'Yes': 2}))


model =RandomForestClassifier(
    n_jobs=1,
    random_state=42,
    n_estimators=500,
    max_features=7,
    max_depth=30,
    class_weight={'No':1 , 'Yes':2}
)

model.fit(X_train, train_target)

t = model.score(X_train,train_target),model.score(X_val,val_target)
print(t)
