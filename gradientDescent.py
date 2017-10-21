
import numpy as np
from Solutions import *



'''

w(τ+1) = w(τ) +  −η(τ)* ∇E
w(τ+1) = w(τ) + η(τ) * ((yi − transpose (w(τ)) * ϕ(xi))*ϕ(xi) - λ * w(τ))

'''
def SGD_sol(design_matrix_train,
            training_labels,
            lamda,
            num_basis):


    η = lamda

    del_error = 100000
    weights = np.random.uniform(-1.0,1.0,size=(1,num_basis))[0]
    eta1 = []
    error_iteration = []
    num_iter = 0


    design_matrix_train, trainin_labels = shuffle(design_matrix_train,training_labels)


    while num_iter < 10:

            for i in range(len(training_labels)):
                error_iteration.append(calculate_error(design_matrix_train,
                                                       weights,
                                                       training_labels))

                # (yi − transpose(w(τ))

                temp1 = training_labels[i] - np.dot(weights, design_matrix_train[i,:].transpose())

                # −(yi − transpose(w(τ)) * ϕ(xi))
                temp2 = -1 * temp1 * design_matrix_train[i,:]

                # ∇E = ∇Ed + λ *∇Ew
                temp3 = temp2 + lamda * weights

                # See if learning rate converges
                eta1.append(η)

                # Weight Updates
                new_weights = weights - η * temp3
                new_weight_vec = np.sum(np.square(new_weights))
                old_weight_vec = np.sum(np.square(weights))


                errors = np.abs(new_weight_vec - old_weight_vec)



                weights = new_weights

            train_ERMS = calculate_error(design_matrix_train, weights, training_labels)
            if num_iter == 0:
                init_error = train_ERMS
                del_error = 100000
            else:
                del_error = init_error - train_ERMS
                init_error = train_ERMS
            num_iter +=1



    plt.plot(error_iteration, 'r-', label='Train Error')
    plt.axis([0, 50, 0.7, 1.8])
    plt.ylabel('RMSE Training')
    plt.xlabel('gradientDescent Iteration')
    plt.title('Change in Training Error vs gradientDescent steps(SGD)')
    l = plt.legend()
    plt.show()

    plt.plot(eta1, 'g-', label='Learning Rate')
    plt.axis([0, 50, 0, 1])
    plt.ylabel('Learning Rate Eta Value')
    plt.xlabel('gradientDescent Iteration')
    plt.title('Learning Rate Decay as Model converges')
    l = plt.legend()
    plt.show()
    return weights, train_ERMS, eta1, error_iteration




def SGD_sol_momentum(design_matrix_train,
            training_labels,
            lamda,
            num_basis):

    # Learning Rate
    η = 1
    boost_factor = 1.25
    degrade_factor = 0.8
    del_error = 100000
    weights = np.random.uniform(-1.0,1.0,size=(1,num_basis))[0]
    eta1 = []
    error_iteration = []
    num_iter = 0



    while del_error > 0.00001 and num_iter < 5:
        # shuffle
        complete_train_data = np.insert(design_matrix_train, 0, training_labels, axis=1)
        np.random.shuffle(complete_train_data)
        training_labels = complete_train_data[:,0]
        design_matrix_train = np.delete(complete_train_data,0,axis=1)

        for i in range(len(training_labels)):
            error_iteration.append(calculate_error(design_matrix_train,
                                                   weights,
                                                   training_labels))

            # (yi − transpose(w(τ))

            temp1 = training_labels[i] - np.dot(weights, design_matrix_train[i,:].transpose())

            # −(yi − transpose(w(τ)) * ϕ(xi))
            temp2 = -1 * temp1 * design_matrix_train[i,:]

            # ∇E = ∇Ed + λ *∇Ew
            temp3 = temp2 + lamda * weights

            # See if learning rate converges
            eta1.append(η)

            # Weight Updates
            new_weights = weights - η * temp3
            new_weight_vec = np.sum(np.square(new_weights))
            old_weight_vec = np.sum(np.square(weights))


            errors = np.abs(new_weight_vec - old_weight_vec)


            # Adaptive Learning Rate
            if np.sqrt(errors) < 0.0001:
                η = η * boost_factor
            else:
                η = η * degrade_factor
            weights = new_weights

        train_error = calculate_error(design_matrix_train, weights, training_labels)
        if num_iter == 0:
            init_error = train_error
            del_error = 100000
        else:
            del_error = init_error - train_error
            init_error = train_error
        num_iter +=1



        plt.plot(error_iteration, 'r-', label='Train Error')
        plt.axis([0, 50, 0.7, 1.8])
        plt.ylabel('RMSE Training')
        plt.xlabel('gradientDescent Iteration')
        plt.title('Change in Training Error vs gradientDescent steps(Momentum)')
        l = plt.legend()
        plt.show()

        plt.plot(eta1, 'g-', label='Learning Rate')
        plt.axis([0, 100, 0, 1])
        plt.ylabel('Learning Rate Eta Value')
        plt.xlabel('gradientDescent Iteration')
        plt.title('Learning Rate Decay as Model converges')
        l = plt.legend()
        plt.show()


    return weights, train_error, eta1, error_iteration


