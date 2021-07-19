class reg(object):  
    from sklearn.kernel_ridge import KernelRidge
    import numpy as np
    global np
    import time
    global model 
    model = KernelRidge(alpha = 0)
    global shape1 ,shape2 , size1 , size2
    global i 
    global thres
    global Amp1
    i = 0
    
    # RidgeRegression for multioutput regression
    
    
    
    def get_order_parameters_and_indices(t1,t2,thres1):
        # Get shapes, will help later in unravelling
        global shape1 ,shape2 , size1 , size2 , thres
        
        thres = thres1
        s1=t1.shape;s2=t2.shape
        
        #for recontruscting the output tensor
        size1 = t1.size
        size2 = t2.size
        shape1 = s1
        shape2 = s2
        
        #Flatten the array for further processing, stuff works better with linear arrays
        t1=t1.flatten();t2=t2.flatten()
    
        #Get the number of order parameters in each amplitude matrix
        l1=len(t1[abs(t1)>thres]);l2=len(t2[abs(t2)>thres])

    
        #If the number is zero, no contribution from the arrays, else get the exact indices, and value of order parameters
        if l1==0:
            ls1=[0];op1=[]
        else:
            op1,ls1 = reg.largest_indices(t1,l1,s1)
        if l2==0:
            ls2=[0];op2=[]
        else:
            op2,ls2= reg.largest_indices(t2,l2,s2)
        #Full order parameters are to be constructed
        order_params=np.concatenate((op1,op2))
    
        return order_params,ls1,ls2
    
    def largest_indices(arr,n,s):
        #Get the highest absolutes values 
        indices=np.argpartition(abs(arr),-n)[-n:]
        
        #Arrange them in decreasing format, can be bypassed
        indices=indices[np.argsort(-abs(arr[indices]))]
    
        #Return order parameter values, and their indices in proper form
        return arr[indices],np.transpose(np.array(np.unravel_index(indices,s)))
      
      
      
      
    def reg_train(t1 , t2 , thres):
        import time
        #saving the currently passed data to retrain later
        global i
        global shape
        global Amp1
        global model
        Amp = np.concatenate((t1.flatten(),t2.flatten()),axis = 0)
        Amp = np.reshape(Amp,(1,Amp.size))
     
        order, ls1 , ls2 =  reg.get_order_parameters_and_indices(t1,t2,thres)
        order = np.reshape(np.abs(order),(1,order.size))
        
        
        
        #loading the data from previous iterations and form a dataset so after every 
        #iter model trains on bigger dataset
        #----------------------------------------------------------------------------
        if(i>0):
                try:
                    concatX = np.load("temp_train_data_Order.npy")
                    concatY = np.load("temp_train_data_Amp.npy")
                except Exception:
                                pass
                
        
        #concatenating the previous iter data with current data for weights update in model
        #--------------------------------------------------------------------------------
        if(i>0):
                try:
                    Amp = np.concatenate((concatY, Amp),axis = 0)
                    order = np.concatenate((concatX, order),axis = 0)
                except Exception:
                                pass
        
        #---------------------------------------------------------------------------------
        
        start = time.time()
    
        #fitting  the dataset into the model----------------------------------------------
        model.fit(order, Amp)
    
        end = time.time()
        print(end - start)
        
        
        #saving the updated dataset into the current directory for later use--------------
        #delete these file after you change the the thres value or started a new process
        np.save("temp_train_data_Order",order)
        np.save("temp_train_data_Amp",Amp)
        i = i+1
        
        #returns the two matrices with position of order amplitudes
        return ls1 , ls2
      
      
    def reg_predict(X):
        
        global shape1 ,shape2 , size1 , size2,thres
        X = X.flatten()
        X = np.abs(X)
        X =np.reshape(X,(1,X.size))
        
        pred = model.predict(X)
        
        
        t1_ten = pred[0,:size1]
        t2_ten = pred[0,size1:]
        t1_ten = np.reshape(t1_ten,shape1)
        t2_ten = np.reshape(t2_ten,shape2)
        
        
        #pred = np.reshape(pred,shape)
        
        
        return t1_ten , t2_ten
    
    
    def accu(X,Y):
        #calculating accuracy of original and predicted tensor, run only after reg_predict
        from sklearn.metrics import r2_score
        size = X.size
        x1 = np.reshape(X, (size))
        y1 = np.reshape(Y, (size))
        print(r2_score(x1, y1))
        return (r2_score(x1, y1))
        

    
