import numpy as np
import pandas as pd

df = pd.read_csv('Housing Price data set.csv')
size = df['lotsize'].count()


temp={'yes':1,'no':0}
df.driveway=[temp[i] for i in df.driveway]
df.recroom=[temp[i] for i in df.recroom]
df.fullbase=[temp[i] for i in df.fullbase]
df.gashw=[temp[i] for i in df.gashw]
df.airco=[temp[i] for i in df.airco]
df.prefarea=[temp[i] for i in df.prefarea]
df.head()

X = df.iloc[:, 2:].values
Y = df.iloc[:,1:2].values

one = np.ones(size, dtype=int)
X = np.insert(X, 0, one, axis=1)

X = (X - np.mean(X)) / np.std(X)
Y = (Y - np.mean(Y)) / np.std(Y)

xT = np.transpose(X)
xTx = np.matmul(xT, X)
xTxI = np.linalg.inv(xTx)
xTy = np.matmul(xT, Y)
w = np.matmul(xTxI, xTy)
print(w)

y_predict1 = np.matmul(X, w)
error = y_predict1 - Y
error = np.square(error)
print("total squared error = ")
error = np.sum(error)
print(error/(2*len(y_predict1)))

'''Y = df[['price']]
X = df[['lotsize','bedrooms','bathrms','stories','driveway','recroom','fullbase','gashw','airco','garagepl','prefarea']]
#X.head()'''

'''size=df['lotsize'].count()
tempX = np.mat(X)
one=np.ones(size,dtype=int)
matx = np.insert(tempX,0,one,axis=1)
maty = np.mat(Y)
xT = np.transpose(matx)
xTx = np.dot(xT, matx)
xTxI = np.linalg.inv(xTx)
w = np.dot(xTxI, maty)
print(w[0])'''