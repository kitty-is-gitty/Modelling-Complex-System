
# coding: utf-8

# In[1]:


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from math import log
from scipy.stats import beta
import pandas as pd
import random
random.seed(7)
from scipy.stats import norm
import scipy


# In[87]:



# Copula
x = np.random.uniform(0.01,0.99,1000); y = np.random.uniform(0.01,0.99,1000)
X_CDF= scipy.stats.norm.cdf(x,y)
X_PDF = norm.ppf(X_CDF)

def Kumaraswamy(X,Y):
    #A1 = 0.3; A2 = 2*A1; Theta = 0.5
    A1 = 1; A2 = 2*A1; Theta = 0.2
    x1, x2 = np.meshgrid(X, Y)
    Nom = Theta*A1*A2*(x1**(A1-1))*(x2**(A2-1))*2
    Den1 = (1-x1**A1)*(1-x2**A2)
    Den2 = (Theta - np.log10(Den1))**3
    Fx = Nom/(Den1*Den2)
    return(Fx)

def TwoDBeta(x1,x2):
    a = 1; B = 2
    BetaFx = beta.pdf(x1,a,B)*beta.pdf(x2,a,B)
    return(BetaFx)
    
X1 = np.linspace(0.01,0.99,1000)
X2 = np.linspace(0.01,0.99,1000)
Z = Kumaraswamy(X1,X2)
print(X1.shape)
print(Z.shape)

samplex = x
sampley = y
## Rejection Sampling
accept = [None] * len(samplex)
KS = np.zeros(len(samplex))
for i in range(0,len(samplex)):
    BetEst = X_PDF[i]
    C = 10
    U = 1
    KS[i] = Kumaraswamy(samplex[i],sampley[i])
    if(C*U*BetEst <= KS[i] ):
        accept[i] = 'Yes'
    elif(C*U*BetEst > KS[i]):
        accept[i] = 'No'
data = [samplex,sampley,KS,accept]
data = list(map(list, zip(*data)))
Table = pd.DataFrame(data, columns = ['samplex','sampley','KS_Value','accept'])
ACC = 'Yes'
Xplot= Table.loc[Table['accept']==ACC,['samplex']]
Yplot= Table.loc[Table['accept']==ACC,['sampley']]
Zplot= Table.loc[Table['accept']==ACC,['KS_Value']]

fig = plt.figure()
ax = plt.axes(projection='3d')
X1, X2 = np.meshgrid(X1, X2)
#ax.contour3D(X1, X2, Z,1000, cmap=cm.coolwarm)
ax.plot_surface(X1, X2, Z, cmap=cm.coolwarm)
ax.scatter(Xplot,Yplot,Zplot,c='red')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()







