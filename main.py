import math
import pickle
print(math.sqrt(100))
print(math.sqrt(400))
print(math.sqrt(900))

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X,y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = RandomForestClassifier()
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

acc = accuracy_score(y_test,y_pred)

print (f"Accuracy score for iris dataset is {acc}")

filename = 'finalized_model'
pickle.dump(clf, open(filename, 'wb'))

print("Bau Bau")



