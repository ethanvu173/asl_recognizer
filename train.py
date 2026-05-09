import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


data = np.loadtxt('output.csv', delimiter=',', dtype=str)
x = data[:,1:].astype(float)
y = data[:,0]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(x_train, y_train)

print(classification_report(y_test, classifier.predict(x_test)))

joblib.dump(classifier, 'asl_alpha_classifier.pkl')
print("Saved model to asl_alpha_classifier.pkl")
