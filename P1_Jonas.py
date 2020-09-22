from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
#from random import random, seed
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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




np.random.seed(153)
N = 100
x = np.random.rand(N)
y = np.random.rand(N)
X,Y = np.meshgrid(x,y)
poly = 5

franke = FrankeFunction(X,Y)
noise = np.random.normal(0, 1, (N,N))
franke = franke + noise             # random data (dataset)


X_complex = np.array([0, 1,0, 2,1,0, 3,2,1,0, 4,3,2,1,0, 5,4,3,2,1,0, 5,4,3,2,1, 5,4,3,2, 5,4,3, 5,4, 5]).astype("int")
Y_complex = np.array([0, 0,1, 0,1,2, 0,1,2,3, 0,1,2,3,4, 0,1,2,3,4,5, 1,2,3,4,5, 2,3,4,5, 3,4,5, 4,5, 5]).astype("int")
#X_complex = np.array([1,0, 2,1,0, 3,2,1,0, 4,3,2,1,0, 5,4,3,2,1,0, 5,4,3,2,1, 5,4,3,2, 5,4,3, 5,4, 5]).astype("int")
#Y_complex = np.array([0,1, 0,1,2, 0,1,2,3, 0,1,2,3,4, 0,1,2,3,4,5, 1,2,3,4,5, 2,3,4,5, 3,4,5, 4,5, 5]).astype("int")

model_complex = np.linspace(0, (poly+1)**2, (poly+1)**2).astype("int")


"""
##### Without resampling #######
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
##### Bootstrapping #######
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



##### Cross-validation #######
print("------- Cross-validations (OLS) --------")
X_design = np.zeros((N, (poly+1)**2))

t0 = t.time()
train_error_cv = np.zeros(len(model_complex))
test_error_cv = np.zeros(len(model_complex))
k = 5

# shuffle dataset in both axis
for i in range(N):
    np.random.shuffle(franke[:,i])
for i in range(N):
    np.random.shuffle(franke[i,:])


groups = np.zeros((k, int(N/k), N))
print(groups.size == franke.size)

for i in range(k):

    groups[i,:,:] =







"""
print("franke shape:        ", franke.shape)
print("ytilde shape:        ", ytilde.shape)
print("ypredict shape:      ", ypredict.shape)
print("X_train shape:       ", X_train.shape)
print("X_test shape:        ", X_test.shape)
print("y_train shape:       ", y_train.shape)
print("y_test shape:        ", y_test.shape)
"""





"""################## PLOTTING ########################"""

"""
fig_ran = plt.figure("Random")
fig_pred = plt.figure("Predict")
ax_ran = fig_ran.gca(projection="3d")
ax_pred = fig_pred.gca(projection="3d")

ran = ax_ran.scatter(X, Y, franke, s=2, c=franke)
fig_ran.colorbar(ran, shrink=0.5, aspect=5)
ax_ran.set_xlabel('X axis')
ax_ran.set_ylabel('Y axis')
ax_ran.set_zlabel('Z axis')
plt.show()
"""

"""
pred = ax_pred.scatter(X, Y, ytilde, s=2, c=franke)
#pred = ax_pred.plot_surface(X, Y, ytilde)
fig_pred.colorbar(pred, shrink=0.5, aspect=5)
ax_pred.set_xlabel('X axis')
ax_pred.set_ylabel('Y axis')
ax_pred.set_zlabel('Z axis')
plt.show()
"""

"""
###### Plotting train and test MSE for various methods ############
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







#
