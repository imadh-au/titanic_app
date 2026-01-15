import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

# 1. Load Data
data = pd.read_csv('train.csv')

# 2. Select Features (The Simple Best Ones)
features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare']
X = data[features]
y = data['Survived']

# 3. Define Pipeline (Preprocessing + Model)
numerical_transformer = SimpleImputer(strategy='median')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, ['Fare', 'Pclass', 'SibSp', 'Parch']),
        ('cat', categorical_transformer, ['Sex'])
    ])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)])

# 4. Train the Model
clf.fit(X, y)
print("✅ Model Trained successfully!")

# 5. Save the Model to a file
joblib.dump(clf, 'titanic_model.pkl')
print("✅ Model Saved as 'titanic_model.pkl'")