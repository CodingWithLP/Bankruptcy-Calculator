from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv('C:\\Users\\Laksh-Games\\OneDrive\\Desktop\\Coding Files\\Py Stuff\\Supervised ML\\KNN\\american_bankruptcy.csv')

X = df.drop(['company_name','status_label','year','X1','X3','X4','X7','X9','X13','X14','X15','X18'], axis=1)
y = df['status_label']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.8,random_state=42)

knn = KNeighborsClassifier(n_neighbors=100)

knn.fit(X_train, y_train)

with open('knnmodel.pickle', 'wb') as f:
    pickle.dump(knn, f)

pred = knn.predict(X_test)
print(accuracy_score(pred, y_test))
