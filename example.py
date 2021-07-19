from MVM2 import reg
import numpy as np    

#fortraining_loading dataset

train = np.load('CCSD_t1_amp_iter_6.npy')
train2 = np.load('CCSD_t2_amp_iter_6.npy')
#training the model on that dataset, we get a  matrix of the the order location
order_loc = reg.reg_train(train,train2,thres = .03)
print(order_loc)


#loading test data
test = np.load('CCSD_t1_amp_iter_18.npy')
test2 = np.load('CCSD_t2_amp_iter_18.npy')
Test , t1 , t2 = reg.get_order_parameters_and_indices(test,test2,0.03)

#predicting the test data reg.reg_predict function which 
#returns the two tensors with all the predicted amplitudes
t1_pred , t2_pred = reg.reg_predict(Test)
reg.accu(t1_pred,train)
reg.accu(t2_pred,train2)
