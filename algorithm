import pandas as pd
import numpy as np
from numpy import sqrt
import random
import scipy.stats
from scipy.stats import norm
import plotly.express as px

inte = 50
q0 = 0.00001
r0 = 0.0001
dt = 1

X, Ytrue, Ycount, Error = [], [], [], []
#начальные условия
x0 = np.array([0, 0]).T
A = np.array([[1, dt], [0, 1]])
H = np.array([1, 0])
Qk = q0 * np.array([[pow(dt, 3)/3, pow(dt, 2)/2], [pow(dt, 2)/2, dt]]) #дисперсия w
Rk = r0 * np.array([1]) # дисперсия v
alpha = 0.3
#norm = norm(0, sqrt(Rk)) # нормальное распределение с математическим ожиданием – 0.
ravn = np.random.uniform(0, sqrt(Rk)) # равномерное распределение для шума с дисперсией R.
z0 = np.dot(H, x0) + norm.ppf(ravn)/1000 #измеряемая переменная z
#step 0
x0_pos = x0 #pos - positive, neg - negative
P_pos = np.dot(x0-x0_pos, (x0-x0_pos).T)
e0 = z0 - np.dot(H, x0_pos)
ek = e0

Ycount.append(np.linalg.norm(x0_pos))
Error.append(np.linalg.norm(e0))
Ytrue.append(np.linalg.norm(z0))
X.append(0)
eps_counter = 1
eps_sum = e0
x = x0_pos
I = np.array([[1, 0], [0, 1]])

Rk = Qk * 10000

for i in range(inte):
#step 1
  x_neg = A.dot(x)
  P_neg = np.dot(np.dot(A, P_pos), A.T) + Qk
#step 2
  zk = np.dot(H, x_neg) + norm.ppf(ravn)/1000
  dk = zk - np.dot(H, x_neg)
  ekhat = eps_sum / eps_counter
  Rk = alpha * Rk + (1-alpha) * (np.dot(ekhat, ekhat.T) + np.dot(np.dot(H, P_neg), H.T))
#print()
  Kk = np.dot(np.multiply(P_neg, H.T), np.linalg.pinv(np.multiply(np.multiply(H, P_neg), H.T) + Rk))
  x_pos = x_neg + np.multiply(Kk, dk)
  ek = zk - np.dot(H, x_pos)
  eps_counter = eps_counter + 1
  eps_sum = eps_sum + ek

P_pos = np.dot((I - np.dot(Kk, H)), P_neg)
Qk = alpha * Qk + (1-alpha) * np.multiply(np.multiply(np.multiply(Kk, dk), dk.T), Kk.T)
P_neg = np.dot(np.dot(A, P_pos), A.T) + Qk
if(i%5==0):
  Ycount.append(np.linalg.norm(x_pos))
  Ytrue.append(np.linalg.norm(zk))
  X.append(i)
  Error.append(np.linalg.norm(ek/zk))
#print("z true = ", zk, " e error = ", ek, " x found = ", x_neg)
#print("x pos = ", x_pos, "x neg = ", x_neg)
#print()
#print("абсолютн ошибка = ek = ", ek, " Относит ошибка = ek/zk = ", ek/zk)
#print("norma = ", np.linalg.norm(ek/zk))
x = x_pos
Y = Ytrue + Ycount
X = X #+ X
df = pd.DataFrame({'x': X,'y': Error})
fig = px.line(df, x = "x", y = "y")
fig.show()
