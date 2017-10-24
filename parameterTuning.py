
from gradientDescent import *
from closedForm import*

# Used to define the best learning rate and the number of basis function
# called in Task 1,
def Tuning1(lamda_values,
            errorMat_T1_T2,
            trainingData_t1_t2,
            trainingLabel_t1_t2,
            validationData_t1_t2,
            validationLabel_t1_t2):

    for lamda in lamda_values:
        # print ("lamda = ", lamda)
        errorMat_T1_T2['cf']['train'][lamda] = []
        errorMat_T1_T2['cf']['validation'][lamda] = []
        for numOfBasisFunction in range(1, 47,1):
            design_matrix_train_letor, \
            sigma_inv,\
            clusters = priorDM(trainingData_t1_t2,
                               trainingLabel_t1_t2,
                               lamda,
                               numOfBasisFunction)

            weights_t1, \
            rmse_train_t1 = cF_weightAdjustment(design_matrix_train_letor,
                                                   sigma_inv,
                                                   trainingLabel_t1_t2,
                                                   lamda,
                                                   numOfBasisFunction)


            design_matrix_validation_letor = resultingDM(validationData_t1_t2,
                                                         sigma_inv,
                                                         clusters,
                                                         numOfBasisFunction)

            rmse_validation_letor = calculate_error(design_matrix_validation_letor,
                                                    weights_t1,
                                                    validationLabel_t1_t2)

            errorMat_T1_T2['cf']['train'][lamda].append(rmse_train_t1)

            errorMat_T1_T2['cf']['validation'][lamda].append(rmse_validation_letor)

            # print (rmse_validation_letor)
    return errorMat_T1_T2


# Used to define the best learning rate and the number of basis function
# called in Task 2
def Tuning2(lamda_values,
            errorMat_T1_T2,
            trainingData_t1_t2,
            trainingLabel_t1_t2,
            validationData_t1_t2,
            validationLabel_t1_t2):

    for lamda in lamda_values:
        errorMat_T1_T2['gradientDescent']['train'][lamda] = []
        errorMat_T1_T2['gradientDescent']['validation'][lamda] = []
        for numOfBasisFunction in range(1, 47):
            design_matrix_train_letor, \
            sigma_inv, \
            clusters = priorDM(trainingData_t1_t2, trainingLabel_t1_t2, lamda, numOfBasisFunction)

            weights_letor, \
            rmse_train_t2, \
            learning_rate_changes, \
            RMSE_records = SGD_sol(design_matrix_train_letor,
                                   trainingLabel_t1_t2,
                                   lamda,
                                   numOfBasisFunction)

            DM_validation_t2 = resultingDM(validationData_t1_t2,
                                                         sigma_inv,
                                                         clusters,
                                                         numOfBasisFunction)

            rmse_validation_t2 = calculate_error(DM_validation_t2,
                                                    weights_letor,
                                                    validationLabel_t1_t2)

            errorMat_T1_T2['gradientDescent']['train'][lamda].append(rmse_train_t2)

            errorMat_T1_T2['gradientDescent']['validation'][lamda].append(rmse_validation_t2)

            # print (numOfBasisFunction,
            #        rmse_train_t2,
            #        rmse_validation_t2)
    return errorMat_T1_T2



# Used to define the best learning rate and the number of basis function
# called in Task 3
def Tuning3(lamda_values,
            errorMat_T3_T4,
            trainingInput_Sync,
            trainingLabel_Sync,
            valid_data_syn,
            valid_labels_syn):

    for lamda in lamda_values:
        errorMat_T3_T4['cf']['train'][lamda] = []
        errorMat_T3_T4['cf']['validation'][lamda] = []
        for numOfBasisFunction in range(1, 11):


            design_matrix_train_syn,\
            sigma_inv_syn, \
            kCenters= priorDM(trainingInput_Sync,
                                      trainingLabel_Sync,
                                      lamda,
                                      numOfBasisFunction)

            weights_syn,\
            rmse_train_syn = cF_weightAdjustment(design_matrix_train_syn,
                                                 sigma_inv_syn,
                                                 trainingLabel_Sync,
                                                 lamda,
                                                 numOfBasisFunction)

            design_matrix_validation_syn = resultingDM(valid_data_syn,
                                                       sigma_inv_syn,
                                                       kCenters,
                                                       numOfBasisFunction)


            rmse_validation_syn = calculate_error(design_matrix_validation_syn,
                                                  weights_syn,
                                                  valid_labels_syn)

            errorMat_T3_T4['cf']['train'][lamda].append(rmse_train_syn)
            errorMat_T3_T4['cf']['validation'][lamda].append(rmse_validation_syn)
            print (rmse_validation_syn)
    return errorMat_T3_T4


# Used to define the best learning rate and the number of basis function
# called in Task 4

def Tuning4(lamda_values,errorMat_T3_T4,
            trainingInput_Sync,
            trainingLabel_Sync,
            valid_data_syn,
            valid_labels_syn):
    for lamda in lamda_values:
        errorMat_T3_T4['gradientDescent']['train'][lamda] = []
        errorMat_T3_T4['gradientDescent']['validation'][lamda] = []
        for numOfBasisFunction in range(1, 11):
            DM_Training_Task4, \
            InverseSig_Task4, \
            kCenters = priorDM(trainingInput_Sync,
                                                      trainingLabel_Sync,
                                                      lamda,
                                                      numOfBasisFunction)

            weights,\
            error_training_gd, \
            learningRateHistory, RMSE_records = SGD_sol_momentum(DM_Training_Task4,
                                                              trainingLabel_Sync,
                                                              lamda,
                                                              numOfBasisFunction)

            design_matrix_validation_syn_GradientDescent = resultingDM(valid_data_syn, InverseSig_Task4, kCenters, numOfBasisFunction)

            rmse_validation_syn_GradientDescent = calculate_error(design_matrix_validation_syn_GradientDescent, weights, valid_labels_syn)

            errorMat_T3_T4['gradientDescent']['train'][lamda].append(error_training_gd)

            errorMat_T3_T4['gradientDescent']['validation'][lamda].append(rmse_validation_syn_GradientDescent)

            print (numOfBasisFunction, error_training_gd, rmse_validation_syn_GradientDescent)
    return errorMat_T3_T4




def performanceTuning1(lamda_values,
                            errorMat_T1_T2,
                            trainingData_t1_t2,
                            trainingLabel_t1_t2,
                            validationData_t1_t2,
                            validationLabel_t1_t2):

    errorMat_T1_T2 = Tuning1(lamda_values,
                                 errorMat_T1_T2,
                                 trainingData_t1_t2,
                                 trainingLabel_t1_t2,
                                 validationData_t1_t2,
                                 validationLabel_t1_t2)

def performanceTuning2(lamda_values,
                               errorMat_T1_T2,
                               trainingData_t1_t2,
                               trainingLabel_t1_t2,
                               validationData_t1_t2,
                               validationLabel_t1_t2):


    errorMat_T1_T2 = Tuning2(lamda_values,
                                 errorMat_T1_T2,
                                 trainingData_t1_t2,
                                 trainingLabel_t1_t2,
                                 validationData_t1_t2,
                                 validationLabel_t1_t2)

def performanceTuning3(lamda_values,
                         errorMat_T3_T4,
                        trainingInput_Sync,
                        trainingLabel_Sync,
                        valid_data_syn,
                        valid_labels_syn):

    errorMat_T3_T4 = Tuning3(lamda_values,
                               errorMat_T3_T4,
                               trainingInput_Sync,
                               trainingLabel_Sync,
                               valid_data_syn,
                               valid_labels_syn)


def performanceTuning4(lamda_values,
                         errorMat_T3_T4,
                        trainingInput_Sync,
                        trainingLabel_Sync,
                        valid_data_syn,
                        valid_labels_syn):

    errorMat_T3_T4 = Tuning4(lamda_values,
                             errorMat_T3_T4,
                             trainingInput_Sync,
                             trainingLabel_Sync,
                             valid_data_syn,
                             valid_labels_syn)

