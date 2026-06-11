import pandas as pd
import pymysql
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

conn = pymysql.connect(
    host="localhost",
    user="root",
    password="123456789@",
    database="churn"
)

query = "SELECT * FROM covid_toy"
df = pd.read_sql(query, conn)

'''
print(df.head())
print(df.shape)



df = df.drop(columns=['User ID', 'Gender'])
x = df.drop(columns=['Purchased'])
y = df['Purchased']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
'''

df = df.dropna()

lb = LabelEncoder()

df['gender'] = lb.fit_transform(df['gender'])
df['cough'] = lb.fit_transform(df['cough'])
df['city'] = lb.fit_transform(df['city'])
df['has_covid'] = lb.fit_transform(df['has_covid'])


df.head(2)

x = df.drop(columns=['has_covid'])
y = df['has_covid']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()



classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)





import joblib

joblib.dump(classifier, 'rf_model.pkl')
print("Model trained and saved as rf_model.pkl")