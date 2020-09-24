from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
#from random import random, seed
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score
import time as t


#print(plt.matplotlib.__version__)


def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def MSE(y_data, y_model):
    n = len(y_data[:,0]) * len(y_data[0,:])
    return np.sum((y_data - y_model)**2)/n


def R2(y_data, y_model):
    y_mean = np.mean(y_data)
    num = np.sum((y_data-y_model)**2)
    denum = np.sum((y_data-y_mean)**2)
    return 1 - (num/denum)


# Produce dataset
def franke_func(N, seed=False):
    if seed == True:
        np.random.seed(153)
    x = np.random.rand(N)
    y = np.random.rand(N)
    X,Y = np.meshgrid(x,y)
    franke = FrankeFunction(X,Y) + np.random.normal(0,1,(N,N))      # the dataset
    return x, y, franke


kfold = 5
N = kfold * 24
poly = 5

x, y, franke = franke_func(N, seed=True)

X_complex = np.array([0, 1,0, 2,1,0, 3,2,1,0, 4,3,2,1,0, 5,4,3,2,1,0, 5,4,3,2,1, 5,4,3,2, 5,4,3, 5,4, 5]).astype("int")
Y_complex = np.array([0, 0,1, 0,1,2, 0,1,2,3, 0,1,2,3,4, 0,1,2,3,4,5, 1,2,3,4,5, 2,3,4,5, 3,4,5, 4,5, 5]).astype("int")
#X_complex = np.array([1,0, 2,1,0, 3,2,1,0, 4,3,2,1,0, 5,4,3,2,1,0, 5,4,3,2,1, 5,4,3,2, 5,4,3, 5,4, 5]).astype("int")
#Y_complex = np.array([0,1, 0,1,2, 0,1,2,3, 0,1,2,3,4, 0,1,2,3,4,5, 1,2,3,4,5, 2,3,4,5, 3,4,5, 4,5, 5]).astype("int")

model_complex = np.linspace(1, (poly+1)**2, (poly+1)**2).astype("int")
#model_complex = np.linspace(1, len(X_complex), len(X_complex).astype("int")

"""
##### Without resampling (part a) #######
print("------- WITHOUT RESAMPLING (OLS) --------")
X_design = np.zeros((N, (poly+1)**2))
k=0

t0 = t.time()
train_error = np.zeros(len(model_complex))
test_error = np.zeros(len(model_complex))
for i,j in zip(X_complex, Y_complex):
    #print(i,j,i+j)
    X_design[:,k] = x**i * y**j

    X_train, X_test, y_train, y_test = train_test_split(X_design, franke, test_size=0.2)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lam = 1e-10
    XT_X = X_train_scaled.T @ X_train_scaled + lam*np.identity(len(X_train_scaled[0,:]))
    beta = np.linalg.inv(XT_X) @ X_train_scaled.T @ y_train

    ytilde = X_train_scaled @ beta
    ypredict = X_test_scaled @ beta

    train_error[k] = MSE(y_train, ytilde)
    test_error[k] = MSE(y_test, ypredict)
    k += 1
print("Runtime: %.1f s \n" % (t.time()-t0))
#print("Training R2 for OLS")
#print(R2(y_train, ytilde))
print("Training MSE for OLS (non-resampling)")
print(MSE(y_train, ytilde))
print("")

#print("Test R2 for OLS")
#print(R2(y_test, ypredict))
print("Test MSE OLS (non-resampling)")
print(MSE(y_test, ypredict))
print("")
"""




"""
##### Bootstrapping (part b) #######
print("------- BOOTSTRAPPING (OLS) --------")
X_design = np.zeros((N, (poly+1)**2))
k=0

t0 = t.time()
train_error_bs = np.zeros(len(model_complex))
test_error_bs = np.zeros(len(model_complex))
trials = N
for i,j in zip(X_complex, Y_complex):
    X_design[:,k] = x**i * y**j
    for I in range(trials):
        X_train, X_test, y_train, y_test = train_test_split(X_design, franke, test_size=0.3)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        lam = 1e-10
        XT_X = X_train_scaled.T @ X_train_scaled + lam*np.identity(len(X_train_scaled[0,:]))
        beta = np.linalg.inv(XT_X) @ X_train_scaled.T @ y_train

        ytilde = X_train_scaled @ beta
        ypredict = X_test_scaled @ beta

        train_error_bs[k] += MSE(y_train, ytilde)
        test_error_bs[k] += MSE(y_test, ypredict)

        print("Degree: %i/%i        Trial: %i %%          " % (k, len(X_complex), 100*I/trials), end="\r")

    train_error_bs[k] /= trials
    test_error_bs[k] /= trials

    k += 1
print("\n Runtime: %.1f s" % (t.time()-t0))
#print("Training R2 for OLS")
#print(R2(y_train, ytilde))
print("Training MSE for OLS (bootstrapping)")
print(MSE(y_train, ytilde))
print("")

#print("Test R2 for OLS")
#print(R2(y_test, ypredict))
print("Test MSE OLS (bootstrapping)")
print(MSE(y_test, ypredict))
print("")
"""





"""
##### Cross-validation (part c) #######
print("------- Cross-validations (OLS) --------")
X_design = np.zeros((N, (poly+1)**2))
X_design[:,0] = 1.0

t0 = t.time()
train_error_cv = np.zeros(len(model_complex))
test_error_cv = np.zeros(len(model_complex))

# shuffle dataset through both axis
x_shuffle = np.linspace(0,N-1,N).astype("int")
y_shuffle = np.linspace(0,N-1,N).astype("int")
np.random.shuffle(x_shuffle)
np.random.shuffle(y_shuffle)
x[:] = x[x_shuffle]
y[:] = y[x_shuffle]
for i in range(N):
    franke[i,:] = franke[i,y_shuffle]
for i in range(N):
    franke[:,i] = franke[x_shuffle,i]

k = kfold               # number of k-foldings
folds = KFold(n_splits=k)
scores = np.zeros((len(model_complex), k))

# scale?

split = int(N/k)
data_split = np.zeros((k, split, N))
x_split = np.zeros((k, split))
y_split = np.zeros((k, split))

for i in range(N):
    K = int(i/split)
    s = int(i-K*split)
    #print(i, K, s)
    data_split[K,s,:] = franke[:,i]
    x_split[K,s] = x[i]
    y_split[K,s] = y[i]


train_sets = [i for i in range(k)]
test_set = np.random.randint(0,k)
train_sets.remove(test_set)
#print(train_sets, test_set)



M = 0
for i,j in zip(X_complex, Y_complex):
    X_design[:,M] = x**i * y**j

    m = 0
    for train in train_sets:
        X_train = X_design[train*split:(train+1)*split]
        y_train = data_split[train,:,:]

        X_test = X_design[test_set*split:(test_set+1)*split]
        y_test = data_split[test_set,:,:]

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # OLS
        lam = 1e-10
        XT_X = X_train_scaled.T @ X_train_scaled + lam*np.identity(len(X_train_scaled[0,:]))
        beta = np.linalg.inv(XT_X) @ X_train_scaled.T @ y_train
        #XT_X = X_train.T @ X_train + lam*np.identity(len(X_train[0,:]))
        #beta = np.linalg.inv(XT_X) @ X_train.T @ y_train

        #ytilde = X_train_scaled @ beta
        ypredict = X_test_scaled @ beta


        scores[M,m] = MSE(y_test, ypredict)
        #scores[M,m] = MSE(y_train, ytilde)

        m += 1

    M += 1


##### scikit predictions #####
Xd_skl = np.zeros((N, (poly+1)**2))
Xd_skl[:,0] = 1.0
polynomial = np.zeros((poly+1)**2)
mse_skl = np.zeros((poly+1)**2)
for degree in range(1, (poly+1)**2):
    polynomial[degree] = degree
    for deg in range(degree):
        Xd_skl[:,deg] = x**[X_complex[deg]] * y**[Y_complex[deg]]
        OLS = LinearRegression()

    mse_skl_fold = cross_val_score(OLS, X=Xd_skl, y=franke, scoring='neg_mean_squared_error', cv=k)
    mse_skl[degree] = np.mean(-mse_skl_fold)
"""





##### Ridge Regression (inversion) (part d) #######
print("------- Ridge Regression --------")
x,y,franke = franke_func(N,seed=True)
X_design = np.zeros((N, (poly+1)**2))
lambdas = np.logspace(-10,0,100)
I = np.identity((poly+1)**2)
mse_ridge_train = np.zeros((poly+1)**2)
mse_ridge_test = np.zeros((poly+1)**2)

m = 0
for i,j in zip(X_complex, Y_complex):
    X_design[:,m] = x**i * y**j
    m += 1

m = 0
for i,j in zip(X_complex, Y_complex):
    #X_design[:,m] = x**i * y**j
    X_train, X_test, y_train, y_test = train_test_split(X_design, franke, test_size=0.2)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    for lamb in lambdas:
        #beta_ridge = (X_train.T @ X_train + lamb*I) @ X_train.T @ y_train
        beta_ridge = (X_train_scaled.T @ X_train_scaled + lamb*I) @ X_train_scaled.T @ y_train

        #ytilde = X_train @ beta_ridge
        ytilde = X_train_scaled @ beta_ridge
        ypredict = X_test @ beta_ridge

        mse_ridge_train[m] += MSE(y_train, ytilde)
        mse_ridge_test[m] += MSE(y_test, ypredict)

    mse_ridge_train[m] /= len(lambdas)
    mse_ridge_test[m] /= len(lambdas)
    m += 1



plt.plot(model_complex, np.log10(mse_ridge_train))
plt.plot(model_complex, np.log10(mse_ridge_test))
plt.show()






"""################## PLOTTING ########################"""

"""
###### Plotting random data using Franke function #######
fig_ran = plt.figure("Random data")
ax_ran = fig_ran.gca(projection="3d")

ran = ax_ran.scatter(X, Y, franke, s=2, c=franke)
fig_ran.colorbar(ran, shrink=0.5, aspect=5)
ax_ran.set_xlabel('X axis')
ax_ran.set_ylabel('Y axis')
ax_ran.set_zlabel('Z axis')
plt.show()
"""


"""
###### Plotting train and test MSE for various methods (part a and b plot)############
fig_error, ax_error = plt.subplots()
check = len(X_complex)
#ax_error.plot(model_complex[:check], train_error[:check], label="Train (OLS)", color="black")
#ax_error.plot(model_complex[:check], test_error[:check], label="Test (OLS)", color="red")
#ax_error.plot(model_complex[:check], train_error_bs[:check], "--", label="Train (OLS BS)", color="black")
#ax_error.plot(model_complex[:check], test_error_bs[:check], "--", label="Test (OLS BS)", color="red")
ax_error.grid(); ax_error.legend()
fig_error.suptitle("train/test error")#; fig_error.tight_layout()
ax_error.set_xlabel("Complexity"); ax_error.set_ylabel("MSE")
plt.show()
"""


"""
##### Plotting test MSE (OLS) using CV (part c plot) ######
mse_cv = np.mean(scores, axis=1)
plt.plot(model_complex, np.log10(mse_cv), label="Produced code")
plt.plot(polynomial, np.log10(mse_skl), label="skl code")
plt.title("MSE of test data using CV (OLS)")
plt.grid(); plt.legend()
plt.xlabel("Complexity"); plt.ylabel("Mean square error (log10)")
plt.show()
"""






#
