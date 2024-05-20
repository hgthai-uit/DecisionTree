# Import các thư viện
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu
df = pd.read_csv("/content/data.csv")
df.head()

# Xem classification
df['Classification'].value_counts()
df['Classification'] = df['Classification'] - 1

# Tạo dữ liệu để train model
y = df['Classification'].values.reshape(-1,1)
X = df.drop(columns=['Classification'])

print(X.shape)
print(y.shape)

# Phân chia dữ liệu
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size=0.1)

# Xây dựng cây
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

Dtree = DecisionTreeClassifier()
Dtree.fit(X_train, y_train)

# Dự đoán trên dữ liệu test
y_pred = Dtree.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)

plot_confusion_matrix(Dtree, X_test, y_test)