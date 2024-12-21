from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import joblib

data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)

lr = LogisticRegression()
lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))

joblib.dump(lr, "model.joblib")