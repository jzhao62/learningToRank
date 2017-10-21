from Solutions import *
from gradientDescent import *


# Used to define the best learning rate and the number of basis function
# called in Task 1,
def Tuning1(lamda_values,
            error_matrix_letor,
            training_data_letor,training_labels_letor,
            valid_data_letor,valid_labels_letor,
            test_data_letor,test_labels_letor):

    for lamda in lamda_values:
        print ("lamda = ", lamda)
        # error_matrix_letor['closed_form']['train'][lamda] = []
        error_matrix_letor['closed_form']['validation'][lamda] = []
        for num_basis in range(1, 30,3):
            design_matrix_train_letor, \
            sigma_inv_letor,\
            rbf_centers_letor = priorDM(training_data_letor,
                                                                training_labels_letor,
                                                                lamda,
                                                                num_basis)

            weights_letor, \
            rmse_train_letor = closedForm_weightTraining(design_matrix_train_letor,
                                                                   sigma_inv_letor,
                                                                   training_labels_letor,
                                                                   lamda,
                                                                   num_basis)


            design_matrix_validation_letor = resultingDM(valid_data_letor,
                                                                       sigma_inv_letor,
                                                                       rbf_centers_letor,
                                                                       num_basis)

            rmse_validation_letor = calculate_error(design_matrix_validation_letor,
                                                    weights_letor,
                                                    valid_labels_letor)

            # error_matrix_letor['closed_form']['train'][lamda].append(rmse_train_letor)

            error_matrix_letor['closed_form']['validation'][lamda].append(rmse_validation_letor)

            print (num_basis, rmse_validation_letor)
    return error_matrix_letor





# Used to define the best learning rate and the number of basis function
# called in Task 2
def Tuning2(lamda_values, error_matrix_letor,
            training_data_letor,training_labels_letor,
            valid_data_letor,valid_labels_letor,
            test_data_letor, test_labels_letor):

    for lamda in lamda_values:
        print ("lamda = ", lamda)
        error_matrix_letor['gradientDescent']['train'][lamda] = []
        error_matrix_letor['gradientDescent']['validation'][lamda] = []
        for num_basis in range(1, 31):
            design_matrix_train_letor, sigma_inv_letor, rbf_centers_letor = priorDM(training_data_letor, training_labels_letor, lamda, num_basis)

            weights_letor, rmse_train_letor, learning_rate_changes, RMSE_records = SGD_sol(design_matrix_train_letor, training_labels_letor, lamda, num_basis)

            design_matrix_validation_letor = resultingDM(valid_data_letor, sigma_inv_letor, rbf_centers_letor, num_basis)

            rmse_validation_letor = calculate_error(design_matrix_validation_letor, weights_letor, valid_labels_letor)

            error_matrix_letor['gradientDescent']['train'][lamda].append(rmse_train_letor)

            error_matrix_letor['gradientDescent']['validation'][lamda].append(rmse_validation_letor)

            print (num_basis, rmse_train_letor, rmse_validation_letor)
    return error_matrix_letor



# Used to define the best learning rate and the number of basis function
# called in Task 3
def Tuning3(lamda_values,
            error_matrix_syn,
            training_data_syn,training_labels_syn,
            valid_data_syn, valid_labels_syn,
            test_data_syn,
            test_labels_syn):
    for lamda in lamda_values:
        print ("lamda = ", lamda)
        error_matrix_syn['closed_form']['train'][lamda] = []
        error_matrix_syn['closed_form']['validation'][lamda] = []
        for num_basis in range(1, 11):


            design_matrix_train_syn,\
            sigma_inv_syn, \
            rbf_centers_syn = priorDM(training_data_syn,
                                      training_labels_syn,
                                      lamda,
                                      num_basis)

            weights_syn,\
            rmse_train_syn = closedForm_weightTraining(design_matrix_train_syn,
                                                                 sigma_inv_syn,
                                                                 training_labels_syn,
                                                                 lamda,
                                                                 num_basis)

            design_matrix_validation_syn = resultingDM(valid_data_syn,
                                                       sigma_inv_syn,
                                                       rbf_centers_syn,
                                                        num_basis)


            rmse_validation_syn = calculate_error(design_matrix_validation_syn,
                                                  weights_syn,
                                                  valid_labels_syn)

            error_matrix_syn['closed_form']['train'][lamda].append(rmse_train_syn)
            error_matrix_syn['closed_form']['validation'][lamda].append(rmse_validation_syn)
            print (num_basis, rmse_train_syn,rmse_validation_syn)
    return error_matrix_syn


# Used to define the best learning rate and the number of basis function
# called in Task 4

def Tuning4(lamda_values,error_matrix_syn,training_data_syn,
            training_labels_syn,
            valid_data_syn,
            valid_labels_syn,
            test_data_syn,
            test_labels_syn):
    for lamda in lamda_values:
        print ("lamda = ", lamda)
        error_matrix_syn['gradientDescent']['train'][lamda] = []
        error_matrix_syn['gradientDescent']['validation'][lamda] = []
        for num_basis in range(1, 11):
            DM_Training_Task4, \
            InverseSig_Task4, \
            rbf_centers_syn_GradientDescent = priorDM(training_data_syn,
                                                      training_labels_syn,
                                                      lamda,
                                                      num_basis)

            weights_syn_GradientDescent,\
            rmse_train_syn_GradientDescent, \
            learning_rate_changes_syn, RMSE_records = SGD_sol(DM_Training_Task4,
                                                                                       training_labels_syn,
                                                                                       lamda,
                                                                                       num_basis)

            design_matrix_validation_syn_GradientDescent = resultingDM(valid_data_syn, InverseSig_Task4, rbf_centers_syn_GradientDescent, num_basis)

            rmse_validation_syn_GradientDescent = calculate_error(design_matrix_validation_syn_GradientDescent, weights_syn_GradientDescent, valid_labels_syn)

            error_matrix_syn['gradientDescent']['train'][lamda].append(rmse_train_syn_GradientDescent)

            error_matrix_syn['gradientDescent']['validation'][lamda].append(rmse_validation_syn_GradientDescent)

            print (num_basis, rmse_train_syn_GradientDescent, rmse_validation_syn_GradientDescent)
    return error_matrix_syn




def performanceTuning1(lamda_values,
                            error_matrix_letor,
                            training_data_letor,
                            training_labels_letor,
                            valid_data_letor,
                            valid_labels_letor,
                            test_data_letor,
                            test_labels_letor):

    error_matrix_letor = Tuning1(lamda_values,
                                 error_matrix_letor,
                                 training_data_letor,
                                 training_labels_letor,
                                 valid_data_letor, valid_labels_letor,
                                 test_data_letor, test_labels_letor)

    ''' The plots can be generated using the error matrix obtained from training.These plots help us choose the optimum hyperparameters M and lamda'''

    for lamda in sorted(error_matrix_letor['closed_form']['train']):
        print("Plot for RMSE Validation across different number of Gaussian Basis Functions, for lamda = ",
              lamda)

        plot_data(error_matrix_letor['closed_form']['validation'][lamda],
                  error_matrix_letor['closed_form']['train'][lamda],
                  lamda,
                  'RMSE Validation',
                  'RMSE Training',
                  [0, 30, 0.5, 0.7])




def performanceTuning2(lamda_values,
                               error_matrix_letor,
                               training_data_letor,
                               training_labels_letor,
                               valid_data_letor,
                               valid_labels_letor,
                               test_data_letor,
                               test_labels_letor):


    error_matrix_letor = Tuning2(lamda_values,
                                 error_matrix_letor,
                                 training_data_letor,
                                 training_labels_letor,
                                 valid_data_letor,
                                 valid_labels_letor,
                                 test_data_letor,
                                 test_labels_letor)

    '''The plots can be generated using the error matrix obtained from training. These plots help us choose the optimum hyperparameters M and lamda'''

    for lamda in sorted(error_matrix_letor['gradientDescent']['train']):
        print("Plot for RMSE Validation across different number of Gaussian Basis Functions, for lamda = ", lamda)
        plot_data(error_matrix_letor['gradientDescent']['validation'][lamda],
                  error_matrix_letor['gradientDescent']['train'][lamda], lamda, 'RMSE Validation', 'RMSE Training',
                  [0, 30, 0.5, 0.7])




def performanceTuning3(lamda_values,
                         error_matrix_syn,
                        training_data_syn,
                        training_labels_syn,
                        valid_data_syn,
                        valid_labels_syn,
                        test_data_syn,
                        test_labels_syn):

    error_matrix_syn = Tuning3(lamda_values,
                               error_matrix_syn,
                               training_data_syn,
                               training_labels_syn,
                               valid_data_syn,
                               valid_labels_syn,
                               test_data_syn,
                               test_labels_syn)

    ''' The plots can be generated using the error matrix obtained from training. These plots help us choose the optimum hyperparameters M and lamda'''

    for lamda in sorted(error_matrix_syn['closed_form']['train']):
        print("Plot for RMSE Validation across different number of Gaussian Basis Functions, for lamda = ", lamda)
        plot_data(error_matrix_syn['closed_form']['validation'][lamda], error_matrix_syn['closed_form']['train'][lamda],
                  lamda, 'RMSE Validation', 'RMSE Training', [1, 9, 0.6, 0.8])







def performanceTuning4(lamda_values,
                         error_matrix_syn,
                        training_data_syn,
                        training_labels_syn,
                        valid_data_syn,
                        valid_labels_syn,
                        test_data_syn,
                        test_labels_syn):

    error_matrix_syn = Tuning4(lamda_values, error_matrix_syn, training_data_syn, training_labels_syn, valid_data_syn, valid_labels_syn, test_data_syn, test_labels_syn)



    ''' The plots can be generated using the error matrix obtained from training. These plots help us choose the optimum hyperparameters M and lamda'''
    for lamda in sorted(error_matrix_syn['gradientDescent']['train']):
        print("Plot for RMSE Validation across different number of Gaussian Basis Functions, for lamda = ", lamda)
        plot_data(error_matrix_syn['gradientDescent']['validation'][lamda],
                  error_matrix_syn['gradientDescent']['train'][lamda], lamda, 'RMSE Validation', 'RMSE Training',
                  [0, 10, 0.7, 1.2])





