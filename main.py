import joblib
import pandas as pd
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, f1_score
from xgboost import XGBClassifier

""" Setting Display Options"""
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

df = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample = pd.read_csv("sample_submission.csv", index_col=False)

df = df.drop(["PassengerId", "Name"], axis=1)
df.describe().T
df.isnull().sum()
df.value_counts()
df["HomePlanet"].nunique()

df["CryoSleep"] = df["CryoSleep"].replace([True], 1)
df["CryoSleep"] = df["CryoSleep"].replace([False], 0)
df["VIP"] = df["VIP"].replace([True], 1)
df["VIP"] = df["VIP"].replace([False], 0)
df["Transported"] = df["Transported"].replace([True], 1)
df["Transported"] = df["Transported"].replace([False], 0)

df[['Deck', 'Number', 'Side']] = df['Cabin'].str.split('/', expand=True)
df = df.drop(["Number", "Cabin"], axis=1)


"""Fill NA"""

df["CryoSleep"] = df["CryoSleep"].fillna(df["CryoSleep"].mode().iloc[0])
df["HomePlanet"] = df["HomePlanet"].fillna(df["HomePlanet"].mode().iloc[0])
df["Destination"] = df["Destination"].fillna(df["Destination"].mode().iloc[0])
df["VIP"] = df["VIP"].fillna(df["VIP"].mode().iloc[0])
df["Deck"] = df["Deck"].fillna(df["Deck"].mode().iloc[0])
df["Side"] = df["Side"].fillna(df["Side"].mode().iloc[0])
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["RoomService"].loc[df["CryoSleep"] == 1.0] = df["RoomService"].loc[df["CryoSleep"] == 1].fillna(0)
df["FoodCourt"].loc[df["CryoSleep"] == 1.0] = df["FoodCourt"].loc[df["CryoSleep"] == 1].fillna(0)
df["ShoppingMall"].loc[df["CryoSleep"] == 1.0] = df["ShoppingMall"].loc[df["CryoSleep"] == 1].fillna(0)
df["Spa"].loc[df["CryoSleep"] == 1.0] = df["Spa"].loc[df["CryoSleep"] == 1].fillna(0)
df["VRDeck"].loc[df["CryoSleep"] == 1.0] = df["VRDeck"].loc[df["CryoSleep"] == 1].fillna(0)
df["RoomService"] = df["RoomService"].fillna(df["RoomService"].mode().iloc[0])
df["FoodCourt"] = df["FoodCourt"].fillna(df["FoodCourt"].mode().iloc[0])
df["ShoppingMall"] = df["ShoppingMall"].fillna(df["ShoppingMall"].mode().iloc[0])
df["Spa"] = df["Spa"].fillna(df["Spa"].mode().iloc[0])
df["VRDeck"] = df["VRDeck"].fillna(df["VRDeck"].mode().iloc[0])


"""Feature Extraction"""

df["Spent"] = df["RoomService"] + df["FoodCourt"] + df["ShoppingMall"] + df["Spa"] + df["VRDeck"]
df = df.drop(["FoodCourt", "ShoppingMall", "FoodCourt", "Spa", "VRDeck"], axis=1)

df = pd.get_dummies(df, drop_first=True)

"""Preparing df for XGB"""

X = df.drop("Transported", axis=1)
y = df["Transported"]


"""Hyperparameter tuning with Optuna"""
def objective(trial):
    # define hyperparameters to optimize for
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 1),
        'subsample': trial.suggest_uniform('subsample', 0.1, 1),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1),
        'gamma': trial.suggest_uniform('gamma', 0, 1),
        'alpha': trial.suggest_loguniform('alpha', 2, 5),
        'lambda': trial.suggest_loguniform('lambda', 2, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300)
    }

    # create XGBClassifier model with optimized hyperparameters
    model = XGBClassifier(**params, random_state=0)

    # evaluate model using cross-validation
    score = cross_val_score(model, X, y, cv=5).mean()

    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=500)
best_params = study.best_params
print(f'Best hyperparameters: {best_params}')


"""Model"""
model = XGBClassifier(**best_params)
model.fit(X,y)
predictions = model.predict(X)

joblib.dump(model, "model.joblib")
mdl = joblib.load("model.joblib")
predictions = mdl.predict(df)


"""Performance metrics"""
accuracy = accuracy_score(y, predictions)
precision = precision_score(y, predictions)
f1 = f1_score(y, predictions)
roc_auc = roc_auc_score(y, predictions)

sample["Transported"] = predictions
sample["Transported"].value_counts()
sample = sample.drop("Unnamed: 0", axis=1)
sample = sample.set_index("PassengerId")
sample

sample["Transported"] = sample["Transported"].replace([1], True)
sample["Transported"] = sample["Transported"].replace([0], False)
sample.to_csv("sample_submission.csv")
