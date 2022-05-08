# required: install python -- anaconda
# python packages: pytorch, matplotlib, pandas, numpy
# sklearn, statsmodels if you run the comparison models


import torch
from torch.nn import Parameter
import matplotlib.pyplot as plt
import math

class one_layer(torch.nn.Module):
    def __init__(self, inputSize: int, outputSize: int, act=None, dr=0) -> None:
        super(one_layer, self).__init__()
        self.outputSize = outputSize    # hidden size
        self.w = Parameter(torch.randn(inputSize, outputSize))
        self.b = Parameter(torch.randn(outputSize))
        self.act = act
        self.dr = dr
        self.reset_parameters() # initialize model weights. This could be done differently.

    def forward(self, input):
        """return the index of the cluster closest to the input"""
        Y = torch.matmul(input,self.w)+self.b # + is a smart plus
        #Y = self.linear(input)
        if self.act is not None:
            Y = self.act(Y)
        return torch.dropout(Y,p=self.dr,train=self.training)

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.w.size(0))
        for para in self.parameters():
            para.data.uniform_(-std, std)

def main():
    import numpy as np
    import os
    import pandas as pd
    os.chdir(r'G:\OneDrive - The Pennsylvania State University\CE 597\CourseMaterials\Datasets')
    runoff_pd = pd.read_csv('CAMELS\\runoff_mm.csv') # working directory must be above CAMELS. If not, use os.chdir to change.
    # some data wrangling from source data to easy-to-use format
    # get [years, basin]
    # pd is a pandas object with which you can run a lot of data operations https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
    modelOption = 2

    runoff_mean = runoff_pd.mean(0).to_numpy()[1:] #[1:] is to remove the basin number # get basin-averaged runoff by averaging along the 0-th dimension (years)
    runoff_mean = np.expand_dims(runoff_mean,1) # see the effect of this statement on shape
    precip_pd = pd.read_csv('CAMELS\\precipitation_mm.csv')
    att = pd.read_csv('CAMELS\\attributes.csv')
    pet_mean =att['pet_mean']*365
    precip_mean = precip_pd.mean(0).to_numpy()[1:] # turn into a numpy ndarray
    precip_mean = np.expand_dims(precip_mean,1) # get basin-averaged runoff
    ax,fig=plt.subplots()
    plt.plot(precip_mean,runoff_mean,'.')
    plt.xlabel('long-term-average precipitation (mm/year)')
    plt.ylabel('long-term-average runoff (mm/year)')

    # (1) ##########################################
    # The scikit-learn version
    from sklearn import linear_model
    from sklearn.metrics import r2_score
    regr = linear_model.LinearRegression()
    regr.fit(precip_mean, runoff_mean)
    #print ('Coefficients: ', regr.coef_)
    #print ('Intercept: ',regr.intercept_)
    #plt.plot(precip_mean, regr.coef_[0][0]*precip_mean + regr.intercept_[0], '-r')

    # (2) ##########################################
    # The statsmodel version
    import statsmodels.api as sm
    X2 = sm.add_constant(precip_mean)
    mod = sm.OLS(runoff_mean,X2)
    fii = mod.fit()
    fii.summary()

    # (3) The PyTorch version ##########################################
    import torch
    torch_option = 3
    precip1 = (precip_mean - precip_mean.mean())/precip_mean.std()
    runoff1 = (runoff_mean - runoff_mean.mean())/runoff_mean.std()
    xT = torch.Tensor(precip1) # scaled
    y = torch.Tensor(runoff1) # scaled
    learning_rate = 0.003; niter=500

    if torch_option == 0:
        # a. manual gradient calculation + explicit update
        d = 1 # only one regressor.
        w = torch.randn(1,d,requires_grad=True)
        b = torch.randn(1,requires_grad=True)

        for t in range(niter):
            y_pred = torch.matmul(xT,w)+b # first dimension in xT is minibatch
            # the statement above can be generalized to y = model(x)
            err = (y_pred - y)
            loss = err.pow(2.0).mean() # mean squared error
            print(f"iteration {t}, loss is: {loss}, w is: {float(w)}, b is {float(b)}")
            prod = err * xT
            w_grad = 2.0*prod.mean()
            b_grad = 2.0*err.mean()
            with torch.no_grad():
                w -= learning_rate * w_grad # you can also use prod.mean()
                b -= learning_rate * b_grad # you can also use err.mean()
                w_grad.zero_()
                b_grad.zero_()
    elif torch_option == 1:
        # b. explicit modeling with automatic differentiation + explicit update
        d = 1 # only one regressor.
        w = torch.randn(1,d,requires_grad=True)
        b = torch.randn(1,requires_grad=True)

        for t in range(niter):
            y_pred = torch.matmul(xT,w)+b # first dimension in xT is minibatch
            # the statement above can be generalized to y = model(x)
            err = (y_pred - y)
            loss = err.pow(2.0).mean() # mean squared error
            loss.backward() # run backpropagation
            # we can now verify that the gradients are in agreement with what we can calculate by hand
            #prod = err * xT
            #print(w.grad - 2.0*prod.mean()) # verified to be 0
            #print(b.grad - 2.0*err.mean())  # verified to be 0
            print(f"iteration {t}, loss is: {loss}, w is: {float(w)}, b is {float(b)}")
            with torch.no_grad():
                w -= learning_rate * w.grad # you can also use prod.mean()
                b -= learning_rate * b.grad # you can also use err.mean()
                w.grad.zero_()
                b.grad.zero_()
    elif torch_option == 2:
        # c. explicit modeling with automatic differentiation + optim step
        d = 1 # only one regressor.
        w = torch.randn(1,d,requires_grad=True)
        b = torch.randn(1,requires_grad=True)
        optim = torch.optim.SGD([w,b],lr=3e-3)

        for t in range(niter):
            y_pred = torch.matmul(xT,w)+b # first dimension in xT is minibatch
            # the statement above can be generalized to y = model(x)
            err = (y_pred - y)
            loss = err.pow(2.0).mean() # mean squared error
            loss.backward() # run backpropagation

            optim.step()
            optim.zero_grad()
            print(f"iteration {t}, loss is: {loss}, w is: {float(w)}, b is {float(b)}")
    elif torch_option == 3:
        # d. pytorch modules
        d = 1 # only one regressor.
        model = one_layer(d,1) # input and output dimensions
        optim = torch.optim.SGD(model.parameters(),lr=3e-3)

        for t in range(niter):
            y_pred = model(xT) # it can be any model!
            loss = torch.nn.functional.mse_loss(y_pred, y)
            loss.backward() # run backpropagation

            optim.step()
            optim.zero_grad()
            print(f"iteration {t}, loss is: {loss}, w is: {float(model.w)}, b is {float(model.b)}")

            pred = model(xT)

        pred_runoff = (pred * runoff_mean.std()+ runoff_mean.mean()).detach().numpy() # detach() so it can be plotted

        plt.plot(precip_mean, pred_runoff,  '-r')

        ax.legend(['observations','model'])
        plt.show()


    else:
        print(0)
if __name__ == "__main__": main()
