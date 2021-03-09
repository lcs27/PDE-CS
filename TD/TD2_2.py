import numpy as np
import matplotlib.pyplot as plt
h=0.1
num=round(10/h)
answer_num=np.zeros((num,1))
answer_real=np.zeros((num,1))
error=np.zeros((num,1))
answer_num[0]=1
answer_real[0]=1
error[0]=0
x=list(range(num))
for i in range(num-1):
    answer_num[i+1]=answer_num[i]+h*(-3*answer_num[i])
    answer_real[i+1]=np.exp(-3*x[i+1])
    error[i+1]=answer_real[i+1]-answer_num[i+1]
    x[i]=x[i]*h
x[num-1]=x[num-1]*h
plt.plot(x,answer_num)
plt.plot(x,answer_real)
plt.show()
plt.plot(x, error)
plt.show()
