import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn

# Enable autologging
mlflow.sklearn.autolog()

# Load preprocessed dataset
data = pd.read_csv("foodspoiled_preprocessing.csv")
target = "Status"
features = [col for col in data.columns if col != target]

X = data[features]
y = data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Fit model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)