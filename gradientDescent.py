
import numpy as np
from Solutions import *

def stochastic_gradient_solution(design_matrix_train, training_labels, lamda, num_basis):
    n = 1
    boost_factor = 1.25
    degrade_factor = 0.8
    del_error = 100000
    weights = np.random.uniform(-1.0,1.0,size=(1,num_basis))[0]
    eta1 = []
    error_iteration = []
    num_iter = 0
    # We shall choose a small value for change in the relative error between two passes
    # as the termination condition, but sometimes the model performs additional unnecessary passes
    # before achieving this small error change. Hence we prune the number of passes and allow the model
    # to make constant passes, as additional passes don't lead to any significant gain.


    while del_error > 0.00001 and num_iter < 5:
        complete_train_data = np.insert(design_matrix_train, 0, training_labels, axis=1)
        np.random.shuffle(complete_train_data)
        training_labels = complete_train_data[:,0]
        design_matrix_train = np.delete(complete_train_data,0,axis=1)

        for i in range(len(training_labels)):
            error_iteration.append(calculate_error(design_matrix_train, weights, training_labels))
            temp1 = training_labels[i] - np.dot(weights, design_matrix_train[i,:].transpose())
            temp2 = -1 * temp1 * design_matrix_train[i,:]
            temp3 = temp2 + lamda * weights
            eta1.append(n)
            new_weights = weights - n * temp3
            new_weight_vec = np.sum(np.square(new_weights))
            old_weight_vec = np.sum(np.square(weights))
            if np.sqrt(np.abs(new_weight_vec - old_weight_vec)) < 0.0001:
                n = n * boost_factor
            else:
                n = n * degrade_factor
            weights = new_weights

        train_error = calculate_error(design_matrix_train, weights, training_labels)
        if num_iter == 0:
            init_error = train_error
            del_error = 100000
        else:
            del_error = init_error - train_error
            init_error = train_error
        num_iter +=1
    return weights, train_error, eta1, error_iteration