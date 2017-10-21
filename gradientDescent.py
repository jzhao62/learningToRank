
import numpy as np
from Solutions import *



'''

w(τ+1) = w(τ) +  −η(τ)* ∇E
w(τ+1) = w(τ) + η(τ) * ((yi − transpose (w(τ)) * ϕ(xi))*ϕ(xi) - λ * w(τ))

'''
def SGD_sol(DM_Training,
            trainingLabels,
            assignedLearningRate,
            num_basis):


    η = assignedLearningRate
    performanceBenchMark = 1
    currWeights = np.random.uniform(-1.0,1.0,size=(1,num_basis))[0]
    ηList = []
    RMSE_records = []
    epochs = 0
    priorError = 0


    # DM_Training, trainin_labels = shuffle(DM_Training,trainingLabels)

    while performanceBenchMark > 0.01:
            DM_Training, trainin_labels = shuffle(DM_Training, trainingLabels)
            for i in range(len(trainingLabels)):
                RMSE_records.append(calculate_error(DM_Training,
                                                    currWeights,
                                                    trainingLabels))

                # (yi − transpose(w(τ))

                temp1 = trainingLabels[i] - np.dot(currWeights, DM_Training[i,:].transpose())

                # −(yi − transpose(w(τ)) * ϕ(xi))
                temp2 = -1 * temp1 * DM_Training[i,:]

                # ∇E = ∇Ed + λ *∇Ew
                temp3 = temp2 + η * currWeights

                # See if learning rate converges
                ηList.append(η)

                # Weight Updates
                new_weights = currWeights - η * temp3

                currWeights = new_weights

            train_ERMS = calculate_error(DM_Training, currWeights, trainingLabels)
            if epochs == 0:
                priorError = train_ERMS
            else:
                performanceBenchMark = abs(priorError - train_ERMS) / priorError
                priorError = train_ERMS
            epochs += 1



    plt.semilogx(RMSE_records, 'b-', label='Train Error(SGD)')
    plt.axis([0, 10000, 0.2, 2.0])
    plt.ylabel('RMSE')
    plt.xlabel('Steps')
    plt.title('Training Error vs GD steps(SGD)')
    plt.grid()
    plt.show()

    plt.plot(ηList , 'c-', label='η')
    plt.axis([0, 50, 0, 1])
    plt.ylabel('Learning Rate')
    plt.xlabel('Steps')
    plt.title('Learning Rate vs steps')
    plt.show()
    return currWeights, train_ERMS, ηList , RMSE_records




def SGD_sol_momentum(DM_Training,
            trainingLabels,
            lamda,
            num_basis):

    # Starting Learning Rate
    η = 1
    boost = 1.5
    degrade= 0.8

    performanceBenchMark = 1
    currWeight = np.random.uniform(-1.0,1.0,size=(1,num_basis))[0]
    ηList = []
    RMSE_records = []
    epochs = 0
    priorError = 0;




    while performanceBenchMark > 0.01:
        # shuffle
        DM_Training, trainin_labels = shuffle(DM_Training, trainingLabels)

        for i in range(len(trainingLabels)):
            RMSE_records.append(calculate_error(DM_Training,
                                                   currWeight,
                                                   trainingLabels))

            # (yi − transpose(w(τ))

            temp1 = trainingLabels[i] - np.dot(currWeight, DM_Training[i,:].transpose())

            # −(yi − transpose(w(τ)) * ϕ(xi))
            temp2 = -1 * temp1 * DM_Training[i,:]

            # ∇E = ∇Ed + λ *∇Ew
            temp3 = temp2 + η* currWeight

            # See if learning rate converges
            ηList.append(η)

            # Weight Updates
            new_weights = currWeight - η * temp3

            new_weight_vec = np.sum(np.square(new_weights))
            old_weight_vec = np.sum(np.square(currWeight))


            errors = np.abs(new_weight_vec - old_weight_vec)

            # Adaptive Learning Rate
            if np.sqrt(errors) < 0.0001:
                η = η * boost
            else:
                η = η * degrade
            currWeight= new_weights

        trainingErros = calculate_error(DM_Training,currWeight, trainingLabels)
        if epochs== 0:
            priorError = trainingErros
        else:
            performanceBenchMark = abs(priorError - trainingErros)/priorError
            priorError = trainingErros
        epochs +=1

    plt.semilogx(RMSE_records, 'b-', label='Train Error')
    plt.xlim([0,10000])
    plt.autoscale(enable=True, axis='y')
    plt.ylabel('RMSE')
    plt.xlabel('Steps')
    plt.title('Training Error vs GD steps(SGD_momentum)')
    plt.grid()
    plt.show()

    plt.plot(ηList, 'c-', label='Learning Rate')
    plt.axis([0, 100, 0, 1])
    plt.ylabel('Learning Rate')
    plt.xlabel('Steps')
    plt.title('Learning Rate vs steps')
    plt.show()




    return currWeight,trainingErros, ηList, RMSE_records


