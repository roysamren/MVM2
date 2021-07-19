class reg():
    from sklearn.kernel_ridge import KernelRidge
    import numpy as np
    import time
    global model 
    model = KernelRidge(alpha=0.00001)
    global shape
    global i 
    i = 0
    
    # RidgeRegression for multioutput regression
    
    
    def reg_train(X , thres):
        import numpy as np
        import time
        
        #loading the data from previous iterations and form a dataset so after every 
        #iter model trains on bigger dataset
        #----------------------------------------------------------------------------
        global i
        global shape
        
        if(i>0):
                concatX = np.load("temp_train_data_X.npy")
                concatY = np.load("temp_train_data_Y.npy")
        
        
        Order_location = X>thres
        #regression Model
        
        from sklearn.metrics import r2_score
        shape = X.shape
        train = X
        mat_size = train.size
        Amp = np.reshape(train, (1,mat_size))
        order = Amp[Amp>thres]
        order = np.abs(order)
        order = np.reshape(order, (1,order.size))
        if(i>0):
                Amp = np.concatenate((concatY, Amp),axis = 0)
                order = np.concatenate((concatX, order),axis = 0)
        
        
        
        start = time.time()
        model.fit(order, Amp)
        end = time.time()
        print(end - start)
        np.save("temp_train_data_X",order)
        np.save("temp_train_data_Y",Amp)
        i = i+1
        
        return Order_location
    
    
    def reg_predict(X):
        import numpy as np
        #removes zeros from the tensors so that only order amplitude remains
        #shape = X.shape
        X = X[X !=0]
        #reshaping in (1,order_size)
        X = np.reshape(X,(1,X.size))
        #taking absolute values
        X = np.abs(X)
        pred = model.predict(X)
        pred = np.reshape(pred,shape)
        return pred
    
    
    def accu(X,Y):
        #calculating accuracy of original and predicted tensor, run only after reg_predict
        import numpy as np
        from sklearn.metrics import r2_score
        size = X.size
        x1 = np.reshape(X, (size))
        y1 = np.reshape(Y, (size))
        print(r2_score(x1, y1))
        return (r2_score(x1, y1))
    
        





def example():
    
    import numpy as np    
    #fortraining_loading dataset
    train = np.load('CCSD_t2_amp_iter_6.npy')
    #training the model on that dataset, we get a  matrix of the 
    #same size as the dataset with only order location as 1 else 0
    order_loc = reg.reg_train(train,thres = .03)
    #loading test data
    test = np.load('CCSD_t2_amp_iter_7.npy')
    #multiplying testdata with order location matrix to remove lower amplitudes
    order1 = np.multiply(test,order_loc)
    #predicting the test data reg.reg_predict function which 
    #returns the tensor with all the predicted amplitudes
    prediction = reg.reg_predict(order1)
    #to test accuracy of predicted and original data
    reg.accu(test,prediction)
    
    import matplotlib as pl
    pl.pyplot.plot(np.reshape(np.abs(prediction), prediction.size))
    pl.pyplot.savefig("graph.svg")
    
    




