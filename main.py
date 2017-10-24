
from parameterTuning import *
import matplotlib.pyplot as plt
from scipy.interpolate import spline
from closedForm import*


'''
Task 1 LETOR and closedForm
'''
def performanceCF_LETR(lamda,
                       numOfBasisFunction,
                            errorMat_T1_T2,
                            trainingData_t1_t2,
                            trainingLabel_t1_t2,
                            validationData_t1_t2,
                            validationLabel_t1_t2,
                            testData_t1_t2,
                            testLabel_t1_t2):



    design_matrix_train_cf, sigmaInv_closedForm, rbf_centers_cf = priorDM(trainingData_t1_t2,
                                                                          trainingLabel_t1_t2,
                                                                          lamda,numOfBasisFunction)


    weightClosedForm,\
    rmse_train_cf = cF_weightAdjustment(design_matrix_train_cf,
                                                                    sigmaInv_closedForm,
                                                                    trainingLabel_t1_t2,
                                                                    lamda,
                                                                    numOfBasisFunction)


    design_matrix_validation_cf = resultingDM(validationData_t1_t2,
                                                            sigmaInv_closedForm,
                                                            rbf_centers_cf,
                                                            numOfBasisFunction)


    design_matrix_test_cf = resultingDM(testData_t1_t2,
                                                      sigmaInv_closedForm,
                                                      rbf_centers_cf,
                                                      numOfBasisFunction)


    rmse_validation_cf = calculate_error(design_matrix_validation_cf,
                                         weightClosedForm,
                                         validationLabel_t1_t2)



    rmse_test_cf = calculate_error(design_matrix_test_cf,
                                   weightClosedForm,
                                   testLabel_t1_t2)


    print ("LeTor using ClosedForm Performance Metrics: \n")
    print ("RMSE on Training Set: ", rmse_train_cf)
    print ("RMSE on Validation Set: ", rmse_validation_cf)
    print ("RMSE on Test Set: ", rmse_test_cf)
    print ("Weights Vector :\n", weightClosedForm)
    print("----------------------------------------------------\n")

'''
Task 2 LETOR and gradientDescent
'''
def performanceGD_LETR(lamda,
                       numOfBasisFunction,
                    errorMat_T1_T2,
                    trainingData_t1_t2,
                    trainingLabel_t1_t2,
                    validationData_t1_t2,
                    validationLabel_t1_t2,
                    testData_t1_t2,
                    testLabel_t1_t2):


    design_matrix_train_GradientDescent, \
    sigma_inv_GradientDescent, \
    rbf_centers_GradientDescent = priorDM(trainingData_t1_t2,
                                          trainingLabel_t1_t2,
                                          lamda,
                                          numOfBasisFunction)

    weights_GradientDescent,\
    rmse_train_GradientDescent, \
    learning_rate_changes_GradientDescent, \
    error_iteration_letor_GradientDescent = SGD_sol_momentum(design_matrix_train_GradientDescent,
                                                    trainingLabel_t1_t2,
                                                    lamda,numOfBasisFunction)


    DM_validation_GD = resultingDM(validationData_t1_t2,
                                   sigma_inv_GradientDescent,
                                   rbf_centers_GradientDescent,
                                   numOfBasisFunction)


    rmse_validation_GD = calculate_error(DM_validation_GD,
                                          weights_GradientDescent,
                                          validationLabel_t1_t2)



    design_matrix_test_GradientDescent = resultingDM(testData_t1_t2,
                                                       sigma_inv_GradientDescent,
                                                       rbf_centers_GradientDescent,
                                                       numOfBasisFunction)

    rmse_test_GradientDescent = calculate_error(design_matrix_test_GradientDescent,
                                    weights_GradientDescent,
                                    testLabel_t1_t2)

    print ("LeTor using GradientDescent Performance Metrics: \n")
    print ("RMSE of Training Set: ", rmse_train_GradientDescent )
    print ("RMSE on Validation Set: ", rmse_validation_GD)
    print ("RMSE on Test Set: ", rmse_test_GradientDescent)
    print ("Weights Vector for the Model:\n", weights_GradientDescent)
    print("----------------------------------------------------\n")
'''
Task 3 Synthetic data and ClosedForm
'''
def performanceCF_Syn(lamda,
                       numOfBasisFunction,
                      errorMat_T3_T4,
                      trainingInput_Sync,
                       trainingLabel_Sync,
                       valid_data_syn,
                       valid_labels_syn,
                       test_data_syn,
                        test_labels_syn):

    design_matrix_train_syn_closed, sigma_inv_syn_closed, rbf_centers_syn_closed = priorDM(trainingInput_Sync,
                                                                                                                   trainingLabel_Sync,
                                                                                                                   lamda,
                                                                                                                   numOfBasisFunction)

    weights_syn_closed, rmse_train_syn_closed= cF_weightAdjustment(design_matrix_train_syn_closed,
                                                                                   sigma_inv_syn_closed,
                                                                                   trainingLabel_Sync,
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

    print ("Best Performance of Linear fitting on synthetic data using closedForm solution\n ")
    print ("RMSE on Training Set: ", rmse_train_syn_closed )
    print ("RMSE on Validation Set: ", rmse_validation_syn_closed)
    print ("RMSE on Test Set: ", rmse_test_syn_closed)
    print ("Weights Vector(w) for the trained Model:\n", weights_syn_closed)
    print("----------------------------------------------------\n")
'''
Task 4 Synthetic data and GradientDescent
'''
def performanceGD_syn(lamda,
                       numOfBasisFunction,
                             errorMat_T3_T4,
                             trainingInput_Sync,
                             trainingLabel_Sync,
                             valid_data_syn,
                             valid_labels_syn,
                             test_data_syn,
                             test_labels_syn):


    DM_Training_Task4,\
    InverseSig_Task4, \
    rbf_centers_syn_GradientDescent = priorDM(trainingInput_Sync,
                                              trainingLabel_Sync,
                                              lamda,
                                              numOfBasisFunction)
    '''
    We are testing different gradient descent method here
    '''
    weights_syn_GradientDescent, \
    rmse_train_syn_GradientDescent, \
    learning_rate_changes_syn_GradientDescent, \
    error_iteration_syn_GradientDescent = SGD_sol_momentum(DM_Training_Task4,
                                                  trainingLabel_Sync,
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

    print ("Best Performance of Linear fitting on synthetic data using gradientDescent solution\n")
    print ("RMSE on Training Set: ", rmse_train_syn_GradientDescent )
    print ("RMSE on Validation Set: ", rmse_validation_syn_GradientDescent)
    print ("RMSE on Test Set: ", rmse_test_syn_GradientDescent)
    print ("Weights Vector for the Model:\n", weights_syn_GradientDescent)
    print("----------------------------------------------------\n")

'''
Run
'''
def generateResult():
    train_percent = 0.8
    validation_percent = 0.1
    lamda_values = [0.001, 0.01, 0.1,1]

    errorMat_T1_T2 = {'cf':{'train':{}, 'validation':{}},
                    'gradientDescent': {'train':{}, 'validation':{}}}
    errorMat_T3_T4 = {'cf': {'train': {}, 'validation': {}},
                      'gradientDescent': {'train': {}, 'validation': {}}}

    in_filename_LeTor = 'data\Querylevelnorm_X.csv'
    out_filename_LeTor = 'data\Querylevelnorm_t.csv'

    input_T1_T2, \
    labels_T1_T2 = readFile(in_filename_LeTor, out_filename_LeTor)

    trainingData_t1_t2, \
    trainingLabel_t1_t2, \
    validationData_t1_t2, \
    validationLabel_t1_t2, \
    testData_t1_t2, \
    testLabel_t1_t2 = partition(input_T1_T2, labels_T1_T2,
                                train_percent, validation_percent)

    print ("Samples in Training: ", len(trainingData_t1_t2))

    print ("Samples in Validation: ", len(validationData_t1_t2))

    print ("Samples in Test: ", len(testData_t1_t2))

    in_filename_syn = 'data\input.csv'
    out_filename_syn = 'data\output.csv'
    feature_mat_syn, output_labels_syn = readFile(in_filename_syn, out_filename_syn)

    trainingInput_Sync, \
    trainingLabel_Sync, \
    valid_data_syn, \
    valid_labels_syn, \
    test_data_syn, \
    test_labels_syn = partition(feature_mat_syn,
                                output_labels_syn,
                                train_percent,
                                validation_percent)

    print("Samples in Training ", len(trainingInput_Sync))

    print("Samples in Validation ", len(valid_data_syn))

    print("Samples in Test", len(test_data_syn))
    print("----------------------------------------------------\n")

    StartTuning = True;
    StartTesting = True;


# if you want to Tune, the best parameter, set the bool false to true;
    if(StartTuning):
        if(False):
            print("Tuning Task 1\n")
            performanceTuning1(lamda_values,
                                errorMat_T1_T2,
                                trainingData_t1_t2,
                                trainingLabel_t1_t2,
                                validationData_t1_t2,
                                validationLabel_t1_t2)
        if(False):
            print("Tuning Task 2\n")
            performanceTuning2(lamda_values,
                           errorMat_T1_T2,
                           trainingData_t1_t2,
                           trainingLabel_t1_t2,
                           validationData_t1_t2,
                           validationLabel_t1_t2)
            a = np.linspace(1, 46, 46)
            xnew = np.linspace(a.min(), a.max(), 46)  # 300 represents number of points to make between T.min and T.max

            # b = errorMat_T1_T2['cf']['validation'][0.001]
            c = errorMat_T1_T2['cf']['validation'][0.01]
            # d = errorMat_T1_T2['cf']['validation'][0.1]
            # e = errorMat_T1_T2['cf']['validation'][1]

            # pb1 = spline(a, b, xnew)
            pb2 = spline(a, c, xnew)
            # pb3 = spline(a, d, xnew)
            # pb4 = spline(a, e, xnew)

            # plt.plot(xnew, pb1, '-r', label='1')
            plt.plot(xnew, pb2, '-g', label='2')
            # plt.plot(xnew, pb3, '-b', label='3')
            # plt.plot(xnew, pb4, '-c', label='4')
            plt.show()
        if (False):
            '''Tuning Task 3'''
            performanceTuning3(lamda_values,
                               errorMat_T3_T4,
                               trainingInput_Sync,
                               trainingLabel_Sync,
                               valid_data_syn,
                               valid_labels_syn)

            a = np.linspace(1, 10, 10)
            xnew = np.linspace(a.min(), a.max(), 10)  # 300 represents number of points to make between T.min and T.max

            b = errorMat_T3_T4['cf']['validation'][0.001]
            c = errorMat_T3_T4['cf']['validation'][0.01]
            d = errorMat_T3_T4['cf']['validation'][0.1]
            e = errorMat_T3_T4['cf']['validation'][1]
            plt.autoscale(enable=True, axis='y')
            plt.ylabel('RMSE')
            plt.xlabel('Number of Basis Function')
            plt.title('RMSE vs Number of Basis at different learning rate for Task 3')
            pb1 = spline(a, b, xnew)
            pb2 = spline(a, c, xnew)
            pb3 = spline(a, d, xnew)
            pb4 = spline(a, e, xnew)
            plt.plot(xnew, pb1, label='rate = 0.001')
            plt.plot(xnew, pb2, label='rate = 0.01')
            plt.plot(xnew, pb3, label='rate = 0.1')
            plt.plot(xnew, pb4, label='rate = 1')
            plt.legend()
            plt.grid()
            plt.show()
        if (False):
            '''Tuning Task 4'''
            print("GradientDescent on Syn")
            performanceTuning4(lamda_values,
                               errorMat_T3_T4,
                               trainingInput_Sync,
                               trainingLabel_Sync,
                               valid_data_syn,
                               valid_labels_syn)
            # a = np.linspace(1, 10, 10)
            # xnew = np.linspace(a.min(), a.max(), 10)  # 300 represents number of points to make between T.min and T.max
            #
            # b = errorMat_T3_T4['gradientDescent']['validation'][0.001]
            # c = errorMat_T3_T4['gradientDescent']['validation'][0.01]
            # d = errorMat_T3_T4['gradientDescent']['validation'][0.1]
            # e = errorMat_T3_T4['gradientDescent']['validation'][1]
            # plt.autoscale(enable=True, axis='y')
            # plt.ylabel('RMSE')
            # plt.xlabel('Number of Basis Function')
            # plt.title('RMSE vs Number of Basis at different learning rate for Task 4')
            # pb1 = spline(a, b, xnew)
            # pb2 = spline(a, c, xnew)
            # pb3 = spline(a, d, xnew)
            # pb4 = spline(a, e, xnew)
            # plt.plot(xnew,pb1, label = 'rate = 0.001')
            # plt.plot(xnew,pb2, label = 'rate = 0.01')
            # plt.plot(xnew, pb3, label='rate = 0.1')
            # plt.plot(xnew, pb4, label='rate = 1')
            # plt.legend()
            # plt.grid()
            # plt.show()

# if you want to See results, set the bool false to true;
# all experiments are based on the best parameters chosen from tuning
    if(StartTesting):
        if (True):
            print("performance of Task 1")
            print("Learning rate 0.01, # of Basis 41")

            performanceCF_LETR(0.01,
                               41,
                               errorMat_T1_T2,
                               trainingData_t1_t2,
                               trainingLabel_t1_t2,
                               validationData_t1_t2,
                               validationLabel_t1_t2,
                               testData_t1_t2,
                               testLabel_t1_t2)
        if (True):

            print("performance of Task 2")
            print("Learning rate 0.01, # of Basis 11")
            performanceGD_LETR(0.01,
                               11,
                               errorMat_T1_T2,
                               trainingData_t1_t2,
                               trainingLabel_t1_t2,
                               validationData_t1_t2,
                               validationLabel_t1_t2,
                               testData_t1_t2, testLabel_t1_t2)
        if(True):
            print("Result of Task 3")
            print("Learning rate 0.1, # of Basis 9")
            performanceCF_Syn(0.1,
                              9,
                              errorMat_T3_T4,
                              trainingInput_Sync,
                              trainingLabel_Sync,
                              valid_data_syn,
                              valid_labels_syn,
                              test_data_syn,test_labels_syn)
        if(True):
            print("Result of Task 4")
            print("Learning rate 1, # of Basis 6")
            performanceGD_syn(1,
                              6,
                                     errorMat_T3_T4,
                                     trainingInput_Sync,
                                     trainingLabel_Sync,
                                     valid_data_syn,
                                     valid_labels_syn,
                                     test_data_syn,
                                     test_labels_syn)
    print("∇ -- ∇ ")


if __name__ == '__main__':
    print("Experiment result")
    generateResult()

