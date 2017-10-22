
from parameterTuning import *
import matplotlib.pyplot as plt
from scipy.interpolate import spline



'''
Task 1 LETOR and closedForm
'''
def performanceCF_LETR(lamda_values,
                            error_matrix_letor,
                            training_data_letor,
                            training_labels_letor,
                            valid_data_letor,
                            valid_labels_letor,
                            test_data_letor,
                            test_labels_letor):


    lamda = 0.01
    numOfBasisFunction = 41

    design_matrix_train_cf, sigmaInv_closedForm, rbf_centers_cf = priorDM(training_data_letor,
                                                                          training_labels_letor,
                                                                          lamda,numOfBasisFunction)


    weights_cf,\
    rmse_train_cf = cF_weightAdjustment(design_matrix_train_cf,
                                                                    sigmaInv_closedForm,
                                                                    training_labels_letor,
                                                                    lamda,
                                                                    numOfBasisFunction)


    design_matrix_validation_cf = resultingDM(valid_data_letor,
                                                            sigmaInv_closedForm,
                                                            rbf_centers_cf,
                                                            numOfBasisFunction)


    design_matrix_test_cf = resultingDM(test_data_letor,
                                                      sigmaInv_closedForm,
                                                      rbf_centers_cf,
                                                      numOfBasisFunction)


    rmse_validation_cf = calculate_error(design_matrix_validation_cf,
                                         weights_cf,
                                         valid_labels_letor)



    rmse_test_cf = calculate_error(design_matrix_test_cf,
                                   weights_cf,
                                   test_labels_letor)


    print ("Performance of Linear Fit using closed form model")
    print ("\n")
    print ("RMSE on Training Set: ", rmse_train_cf)
    print ("\n")
    print ("RMSE on Validation Set: ", rmse_validation_cf)
    print ("\n")
    print ("RMSE on Test Set: ", rmse_test_cf)
    print ("\n")
    print ("Weights Vector(w) for the trained Model:\n", weights_cf)







'''
Task 2 LETOR and gradientDescent
'''
def performanceGD_LETR(lamda_values,
                    error_matrix_letor,
                    training_data_letor,
                    training_labels_letor,
                    valid_data_letor,
                    valid_labels_letor,
                    test_data_letor,
                    test_labels_letor):


    ''' The 2 parameters are determined , compute performance'''
    lamda = 0.01
    numOfBasisFunction = 11

    design_matrix_train_GradientDescent, \
    sigma_inv_GradientDescent, \
    rbf_centers_GradientDescent = priorDM(training_data_letor,
                                          training_labels_letor,lamda,
                                          numOfBasisFunction)

    weights_GradientDescent,\
    rmse_train_GradientDescent, \
    learning_rate_changes_GradientDescent, \
    error_iteration_letor_GradientDescent = SGD_sol_momentum(design_matrix_train_GradientDescent,
                                                    training_labels_letor,
                                                    lamda,numOfBasisFunction)


    DM_validation_GD = resultingDM(valid_data_letor,
                                   sigma_inv_GradientDescent,
                                   rbf_centers_GradientDescent,
                                   numOfBasisFunction)


    rmse_validation_GD = calculate_error(DM_validation_GD,
                                          weights_GradientDescent,
                                          valid_labels_letor)



    design_matrix_test_GradientDescent = resultingDM(test_data_letor,
                                                       sigma_inv_GradientDescent,
                                                       rbf_centers_GradientDescent,
                                                       numOfBasisFunction)

    rmse_test_GradientDescent = calculate_error(design_matrix_test_GradientDescent,
                                    weights_GradientDescent,
                                    test_labels_letor)

    print ("Best fit Linear Regression Model using Stochastic Gradient Descent(gradientDescent) Performance Metrics: ")
    print ("\n")
    print ("RMSE on Training Set: ", rmse_train_GradientDescent )
    print ("\n")
    print ("RMSE on Validation Set: ", rmse_validation_GD)
    print ("\n")
    print ("RMSE on Test Set: ", rmse_test_GradientDescent)
    print ("\n")
    print ("Weights Vector(w) for the trained Model:\n", weights_GradientDescent)


    #
    # ''' Commented the plots, to handle error on timberlake server, uncomment to plot '''
    # plt.plot(error_iteration_letor_GradientDescent, 'r-', label='Train Error')
    # plt.axis([0, 50, 0.4, 1])
    # plt.ylabel('RMSE in Training Set')
    # plt.xlabel('Number of Epochs')
    # plt.title('Change in Training Error vs Epochs')
    # l = plt.legend()
    # plt.show()
    #
    # plt.plot(learning_rate_changes_GradientDescent, 'g-', label='Learning Rate')
    # plt.axis([0, 100, 0, 1])
    # plt.ylabel('Learning Rate Value')
    # plt.xlabel('Number of Epochs')
    # plt.title('Learning Rate  vs Epochs')
    # l = plt.legend()
    # plt.show()



'''
Task 3 Synthetic data and ClosedForm
'''
def performanceCF_Syn(lamda_values,
                                     error_matrix_syn,
                                     training_data_syn,
                                     training_labels_syn,
                                     valid_data_syn,
                                     valid_labels_syn,
                                     test_data_syn,
                                     test_labels_syn):
    
    ''' After choosing the best hyperparameters, we fix their values and compute the model performance on test set'''
    lamda = 0.1
    numOfBasisFunction = 9
    design_matrix_train_syn_closed, sigma_inv_syn_closed, rbf_centers_syn_closed = priorDM(training_data_syn,
                                                                                                                   training_labels_syn,
                                                                                                                   lamda,
                                                                                                                   numOfBasisFunction)

    weights_syn_closed, rmse_train_syn_closed= cF_weightAdjustment(design_matrix_train_syn_closed,
                                                                                   sigma_inv_syn_closed,
                                                                                   training_labels_syn,
                                                                                   lamda,
                                                                                   numOfBasisFunction)

    design_matrix_validation_syn_closed = resultingDM(valid_data_syn,
                                                                    sigma_inv_syn_closed,
                                                                    rbf_centers_syn_closed,
                                                                    numOfBasisFunction)

    rmse_validation_syn_closed = calculate_error(design_matrix_validation_syn_closed,
                                                 weights_syn_closed,
                                                 valid_labels_syn)

    design_matrix_test_syn_closed = resultingDM(test_data_syn,
                                                              sigma_inv_syn_closed,
                                                              rbf_centers_syn_closed,
                                                              numOfBasisFunction)

    rmse_test_syn_closed = calculate_error(design_matrix_test_syn_closed, weights_syn_closed, test_labels_syn)

    print ("Best fit Linear Regression Model using Closed Form Solution Performance: ")
    print ("\n")
    print ("RMSE on Training Set: ", rmse_train_syn_closed )
    print ("\n")
    print ("RMSE on Validation Set: ", rmse_validation_syn_closed)
    print ("\n")
    print ("RMSE on Test Set: ", rmse_test_syn_closed)
    print ("\n")
    print ("Weights Vector(w) for the trained Model:\n", weights_syn_closed)





'''
Task 4 Synthetic data and GradientDescent
'''
def performanceGD_syn(lamda_values,
                             error_matrix_syn,
                             training_data_syn,
                             training_labels_syn,
                             valid_data_syn,
                             valid_labels_syn,
                             test_data_syn,
                             test_labels_syn):


    ''' After choosing the best hyperparameters, we fix their values and compute the model performance on test set'''
    lamda = 0.1
    numOfBasisFunction = 5

    DM_Training_Task4,\
    InverseSig_Task4, \
    rbf_centers_syn_GradientDescent = priorDM(training_data_syn,
                                              training_labels_syn,
                                              lamda,
                                              numOfBasisFunction)
    '''
    We are testing different gradient descent method here
    '''
    weights_syn_GradientDescent, \
    rmse_train_syn_GradientDescent, \
    learning_rate_changes_syn_GradientDescent, \
    error_iteration_syn_GradientDescent = SGD_sol_momentum(DM_Training_Task4,
                                                  training_labels_syn,
                                                  lamda,
                                                  numOfBasisFunction)



    design_matrix_validation_syn_GradientDescent = resultingDM(valid_data_syn,
                                                                InverseSig_Task4,
                                                                rbf_centers_syn_GradientDescent,
                                                                numOfBasisFunction)

    rmse_validation_syn_GradientDescent = calculate_error(design_matrix_validation_syn_GradientDescent,
                                                          weights_syn_GradientDescent,
                                                          valid_labels_syn)


    design_matrix_test_syn_GradientDescent = resultingDM(test_data_syn,
                                                                       InverseSig_Task4,
                                                                       rbf_centers_syn_GradientDescent,
                                                                       numOfBasisFunction)

    rmse_test_syn_GradientDescent = calculate_error(design_matrix_test_syn_GradientDescent,
                                                    weights_syn_GradientDescent,
                                                    test_labels_syn)

    print ("Synthetic data using Stochastic Gradient Descent(gradientDescent) Performance Metrics: ")
    print ("\n")
    print ("RMSE on Training Set: ", rmse_train_syn_GradientDescent )
    print ("\n")
    print ("RMSE on Validation Set: ", rmse_validation_syn_GradientDescent)
    print ("\n")
    print ("RMSE on Test Set: ", rmse_test_syn_GradientDescent)
    print ("\n")
    print ("Weights Vector(w) for the trained Model:\n", weights_syn_GradientDescent)

    # ''' Commented the plots, to handle error on timberlake server, uncomment to plot '''
    # plt.plot(error_iteration_syn_GradientDescent, 'r-', label='Train Error')
    # plt.axis([0, 50, 0.7, 1.8])
    # plt.ylabel('RMSE Training')
    # plt.xlabel('gradientDescent Iteration')
    # plt.title('Change in Training Error vs gradientDescent Iterations')
    # l = plt.legend()
    # plt.show()
    #
    # plt.plot(learning_rate_changes_syn_GradientDescent, 'g-', label='Learning Rate')
    # plt.axis([0, 100, 0, 1])
    # plt.ylabel('Learning Rate Eta Value')
    # plt.xlabel('gradientDescent Iteration')
    # plt.title('Learning Rate Decay as Model converges')
    # l = plt.legend()
    # plt.show()



def main():
    train_percent = 0.8
    validation_percent = 0.1
    # lamda_values = [0.001, 0.01, 0.1]
    lamda_values = [0.01]



    in_filename_syn = 'data\Querylevelnorm_X.csv'
    out_filename_syn = 'data\Querylevelnorm_t.csv'

    feature_mat_letor, output_labels_letor = readFile(in_filename_syn, out_filename_syn)
    training_data_letor, training_labels_letor, valid_data_letor, valid_labels_letor, test_data_letor, test_labels_letor = partition(feature_mat_letor,
                                                                                                                                               output_labels_letor,
                                                                                                                                               train_percent,
                                                                                                                                               validation_percent)
    print ("Number of samples in Training Data: ", len(training_data_letor))
    print ("Number of samples in Validation Data: ", len(valid_data_letor))
    print ("Number of samples in Test Data: ", len(test_data_letor))

    error_matrix_letor = {'cf':{'train':{}, 'validation':{}},
                          'gradientDescent': {'train':{}, 'validation':{}}}


    if(True):
        if(False):
            print("Tuning Task 1\n")
            performanceTuning1(lamda_values,
                                error_matrix_letor,
                                training_data_letor,
                                training_labels_letor,
                                valid_data_letor,
                                valid_labels_letor,
                                test_data_letor,
                                test_labels_letor)



        if(True):
            print("Tuning Task 2\n")
            performanceTuning2(lamda_values,
                           error_matrix_letor,
                           training_data_letor,
                           training_labels_letor,
                           valid_data_letor,
                           valid_labels_letor,
                           test_data_letor,
                           test_labels_letor)
            a = np.linspace(1, 46, 46)
            xnew = np.linspace(a.min(), a.max(), 46)  # 300 represents number of points to make between T.min and T.max

            # b = error_matrix_letor['cf']['validation'][0.001]
            c = error_matrix_letor['cf']['validation'][0.01]
            # d = error_matrix_letor['cf']['validation'][0.1]
            # e = error_matrix_letor['cf']['validation'][1]

            # pb1 = spline(a, b, xnew)
            pb2 = spline(a, c, xnew)
            # pb3 = spline(a, d, xnew)
            # pb4 = spline(a, e, xnew)

            # plt.plot(xnew, pb1, '-r', label='1')
            plt.plot(xnew, pb2, '-g', label='2')
            # plt.plot(xnew, pb3, '-b', label='3')
            # plt.plot(xnew, pb4, '-c', label='4')
            plt.show()




        if(False):
            print("performance on Task 1\n")
            performanceCF_LETR(lamda_values,
                                        error_matrix_letor,
                                        training_data_letor,
                                        training_labels_letor,
                                        valid_data_letor,
                                        valid_labels_letor,
                                        test_data_letor,
                                        test_labels_letor)

        if (True):
            print("performance on Task 2\n")
            performanceGD_LETR(lamda_values,
                    error_matrix_letor,
                    training_data_letor,
                    training_labels_letor,
                    valid_data_letor,
                    valid_labels_letor,
                    test_data_letor, test_labels_letor)




    in_filename_syn = 'data\input.csv'
    out_filename_syn = 'data\output.csv'
    feature_mat_syn, output_labels_syn = readFile(in_filename_syn, out_filename_syn)



    training_data_syn,\
    training_labels_syn, \
    valid_data_syn, \
    valid_labels_syn, \
    test_data_syn, \
    test_labels_syn = partition(feature_mat_syn,
                                          output_labels_syn,
                                          train_percent,
                                          validation_percent)

    print ("Number of samples in Training Data: ", len(training_data_syn))
    print ("Number of samples in Validation Data: ", len(valid_data_syn))
    print ("Number of samples in Test Data: ", len(test_data_syn))

    error_matrix_syn = {'cf':{'train':{}, 'validation':{}}, 'gradientDescent': {'train':{}, 'validation':{}}}


    if(True):
        if(False):
            '''Tuning Task 3'''
            performanceTuning3(lamda_values,
                           error_matrix_syn,
                           training_data_syn,
                           training_labels_syn,
                           valid_data_syn,
                           valid_labels_syn,
                           test_data_syn,
                           test_labels_syn)

            a = np.linspace(1, 10, 10)
            xnew = np.linspace(a.min(), a.max(), 10)  # 300 represents number of points to make between T.min and T.max


            b = error_matrix_syn['cf']['validation'][0.001]
            c = error_matrix_syn['cf']['validation'][0.01]
            d = error_matrix_syn['cf']['validation'][0.1]
            e = error_matrix_syn['cf']['validation'][1]

            pb1 = spline(a, b, xnew)
            pb2 = spline(a, c,xnew)
            pb3 = spline(a, d, xnew)
            pb4 = spline(a, e, xnew)

            plt.plot(xnew, pb1, '-r', label='1')
            plt.plot(xnew, pb2, '-g', label='2')
            plt.plot(xnew, pb3, '-b', label='3')
            plt.plot(xnew, pb4, '-c', label='4')
            plt.show()




        if(False):
            '''Tuning Task 4'''
            print("GradientDescent on Syn")
            performanceTuning4(lamda_values,
                               error_matrix_syn,
                               training_data_syn,
                               training_labels_syn,
                               valid_data_syn,
                               valid_labels_syn,
                               test_data_syn,
                               test_labels_syn)
            a = np.linspace(1, 10, 10)
            xnew = np.linspace(a.min(), a.max(), 10)  # 300 represents number of points to make between T.min and T.max

            b = error_matrix_syn['gradientDescent']['validation'][0.001]
            c = error_matrix_syn['gradientDescent']['validation'][0.01]
            d = error_matrix_syn['gradientDescent']['validation'][0.1]
            e = error_matrix_syn['gradientDescent']['validation'][1]

            pb1 = spline(a, b, xnew)
            pb2 = spline(a, c, xnew)
            pb3 = spline(a, d, xnew)
            pb4 = spline(a, e, xnew)

            plt.plot(xnew, pb1, '-r', label='1')
            plt.plot(xnew, pb2, '-g', label='2')
            plt.plot(xnew, pb3, '-b', label='3')
            plt.plot(xnew, pb4, '-c', label='4')
            plt.show()


        if(False):
            print("Result on Task 3")
            performanceCF_Syn(lamda_values,
                              error_matrix_syn,
                              training_data_syn,
                              training_labels_syn,
                              valid_data_syn,
                              valid_labels_syn,
                              test_data_syn,test_labels_syn)

        if(False):
            print("Result onTask 4")
            performanceGD_syn(lamda_values,
                                     error_matrix_syn,
                                     training_data_syn,
                                     training_labels_syn,
                                     valid_data_syn,
                                     valid_labels_syn,
                                     test_data_syn,
                                     test_labels_syn)




main()

print("∇")