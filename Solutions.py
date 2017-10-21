import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def readFile(in_filename, out_filename):
    input_matrix = []
    output = []

    lines_in = [line.rstrip('\n') for line in open(in_filename)]
    lines_out = [line.rstrip('\n') for line in open(out_filename)]

    print (len(lines_in), len(lines_out))
    for line in lines_out:
        output.append(int(line))

    for line in lines_in:
        token_list = line.split(',')
        input_vec = []
        for token in token_list:
            input_vec.append(float(token))
        input_matrix.append(input_vec)

    output = np.array(output)
    input_matrix = np.array(input_matrix)
    return input_matrix, output


def calculate_error(input_matrix,
                    weights,
                    output):

    predicted_output = np.dot(weights, input_matrix.transpose())
    squared_error_sum = np.sum(np.square(output - predicted_output))
    rmse = np.sqrt(float(squared_error_sum)/len(output))
    return rmse

def shuffle(input_matrix, output):
    complete_train_data = np.insert(input_matrix, 0, output, axis=1)
    np.random.shuffle(complete_train_data)
    trainingLabels = complete_train_data[:,0]
    input_matrix = np.delete(complete_train_data,0,axis=1)
    return input_matrix, trainingLabels

def plot_data(y_values1, y_values2, lamda, label1, label2, axis_dim):
    plt.plot(y_values1, 'ro',label=label1)
    plt.plot(y_values2, 'b-', label = label2)
    plt.axis(axis_dim)
    plt.ylabel('RMSE')
    plt.xlabel('Model Complexity')
    plt.title('Lambda = ' + str(lamda))
    l = plt.legend()
    plt.show()



def partition(input_matrix,
              output,
              train_percent,
              validation_percent):

    input_matrix, output = shuffle(input_matrix, output)
    trainingSets = []
    trainingLabels = []
    validationSets = []
    validationLabels = []
    testSets = []
    testLabels = []

    train_len = int(np.floor(float(train_percent) * len(input_matrix)))
    for i in range(train_len):
        trainingSets.append(input_matrix[i])
        trainingLabels.append(output[i])

    validation_len = int(np.floor(validation_percent * len(input_matrix)))
    for i in range(train_len, train_len+validation_len):
        validationSets.append(input_matrix[i])
        validationLabels.append(output[i])

    for i in range(train_len+validation_len, len(input_matrix)):
        testSets.append(input_matrix[i])
        testLabels.append(output[i])

    return np.array(trainingSets), np.array(trainingLabels), np.array(validationSets), np.array(validationLabels), np.array(testSets), np.array(testLabels)

def k_means_clusters(train_data,
                     trainingLabels,
                     num_basis):
    try:
        kmeans = KMeans(n_clusters=num_basis, random_state=0).fit(train_data)
        labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_
    except Exception as e:
        print ("Error: ", str(e))
        from kmeans_implement import kmeans
        cluster_centers = kmeans(train_data, k=num_basis)
    return cluster_centers



# could be used to track K means training
# def k_means_clusters(train_data, trainingLabels, num_basis):
#     kmeans = KMeans(n_clusters=num_basis, random_state=0).fit(train_data)
#     labels = kmeans.labels_
#     cluster_centers = kmeans.cluster_centers_
#     clusters = {}
#     for i in range(num_basis):
#         clusters[i] = {'train_data':[], 'output' : [], 'center' : cluster_centers[i]}
#
#         for j in range(len(train_data)):
#             if labels[j] == i:
#                 clusters[i]['train_data'].append(train_data[j])
#                 clusters[i]['output'].append(trainingLabels[j])
#
#     for i in range(num_basis):
#         print(i, len(clusters[i]['train_data']))
#
#     return cluster_centers



'''
Create Closed Form DM based on training data, 
the resulting inverse sigma, and random centers are used to create DM for validation and test sets


'''
def priorDM(train_data,
                                    trainingLabels,
                                    lamda,
                                    num_basis):

    variance = train_data.var(axis=0) 
    sigma = variance * np.identity(len(train_data[0]))
    sigma = sigma + 0.001 * np.identity(len(train_data[0])) # Add a small quantity to avoid 0 values in variance matrix.
    sigma_inv = np.linalg.inv(sigma)

    rand_centers = k_means_clusters(train_data, trainingLabels, num_basis)
    rand_centers = np.array(rand_centers)
    design_matrix=np.zeros((len(train_data),num_basis));

    for i in range(len(train_data)):
        for j in range(num_basis):
            if j==0:
                design_matrix[i][j] = 1;
            else:
                x_Minus_mu = train_data[i]-rand_centers[j]
                x_Minus_mu_trans = x_Minus_mu.transpose()
                temp1 = np.dot(sigma_inv, x_Minus_mu_trans)
                temp2 = np.dot(x_Minus_mu, temp1)
                # Equation (2) in Main
                design_matrix[i][j] = np.exp(((-0.5)*temp2))
                
    return design_matrix, sigma_inv, rand_centers


'''
the resulting inverse sigma, and random centers are used to create DM for validation and test sets

'''


def resultingDM(data,
                sigma_inv,
                rand_centers,
                num_basis):

    design_matrix = np.zeros((len(data),num_basis))
    for i in range(len(data)):
        for j in range(num_basis):
            if j==0:
                design_matrix[i][j] = 1;
            else:
                x_Minus_mu = data[i] - rand_centers[j]
                x_Minus_mu_trans = x_Minus_mu.transpose()
                temp1 = np.dot(sigma_inv,x_Minus_mu_trans)
                temp2 = np.dot(x_Minus_mu, temp1)
                design_matrix[i][j] = np.exp(((-0.5) * temp2))
    return design_matrix



'''
 w∗ = inv((λI + transpose(Φ) * Φ))* transpose(Φ)*y , nothing too complex
'''

def closedForm_weightTraining(Φ,
                              sigma_inv,
                              trainingLabels,
                              reg,
                              num_basis):
    Φ_trans = Φ.transpose()
    λ= reg * np.identity(num_basis)

    firstHalf= np.linalg.inv(λ + np.dot(Φ_trans, Φ))
    secondHalf= np.dot(Φ_trans, trainingLabels)

    weights = np.dot(firstHalf, secondHalf)
    trainingLoss = calculate_error(Φ, weights, trainingLabels)
    return weights, trainingLoss





# def closed_form_solution_validation_phase(validationSets, validationLabels, weights, sigma_inv, rand_centers,num_basis):
#     valid_design_matrix=np.zeros((len(validationSets),num_basis));
#     for i in range(len(validationSets)):
#         for j in range(num_basis):
#             if j==0:
#                 valid_design_matrix[i][j] = 1;
#             else:
#                 x_mu = validationSets[i]-rand_centers[j]
#                 x_mu_trans = x_mu.transpose()
#                 temp1_valid = np.dot(sigma_inv, x_mu_trans)
#                 temp2_valid = np.dot(x_mu, temp1_valid)
#                 valid_design_matrix[i][j] = np.exp(((-0.5)*temp2_valid))
#
#     predicted_output_validation = np.dot(weights, valid_design_matrix.transpose())
#     count = 0
#     for i in range(len(predicted_output_validation)):
#         if np.rint(predicted_output_validation[i]) != int(validationLabels[i]):
#             count +=1
#
#     print ("Error: ", float(count)/len(predicted_output_validation))
#     sq_error_sum_validation = np.sum(np.square(validationLabels - predicted_output_validation))
#
#     # print sq_error_sum
#     validation_error = np.sqrt(float(sq_error_sum_validation)/len(validationSets))
#     return validation_error


