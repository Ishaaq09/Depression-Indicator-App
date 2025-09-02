import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from torch.utils.data import DataLoader, TensorDataset

Num_feat = ['Age', 'Work/Study Hours']
ordinal_feat = ['Work Pressure', 'Job Satisfaction', 'Financial Stress']
nominal_feat = [
    'Gender', 'City', 'Working Professional or Student', 'Profession',
    'Sleep Duration', 'Dietary Habits', 'Degree',
    'Have you ever had suicidal thoughts ?',
    'Family History of Mental Illness'
]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])
ordinal_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent"))
])
nominal_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, Num_feat),
    ("ord", ordinal_transformer, ordinal_feat),
    ("nom", nominal_transformer, nominal_feat)
])

def preprocess_data():
    df = pd.read_csv("cleaned_dataset.csv")

    X = df.drop("Depression", axis=1)
    y = df["Depression"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    X_train_tensor = torch.tensor(X_train_transformed.toarray(), dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_transformed.toarray(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_dim = X_train_tensor.shape[1]

    return train_loader, test_loader, input_dim, (X_test_tensor, y_test_tensor)

if __name__ == "__main__":
    train_loader, test_loader, input_dim, _ = preprocess_data()
    print(f"Preprocessing complete. Input dimension = {input_dim}")