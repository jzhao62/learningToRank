import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def readFile(in_filename, out_filename):
    input_matrix = []
    output_vec = []

    lines_in = [line.rstrip('\n') for line in open(in_filename)]
    lines_out = [line.rstrip('\n') for line in open(out_filename)]

    print (len(lines_in), len(lines_out))
    for line in lines_out:
        output_vec.append(int(line))

    for line in lines_in:
        token_list = line.split(',')
        input_vec = []
        for token in token_list:
            input_vec.append(float(token))
        input_matrix.append(input_vec)

    output_vec = np.array(output_vec)
    input_matrix = np.array(input_matrix)
    return input_matrix, output_vec


def calculate_error(input_matrix,
                    weights,
                    output_vec):

    predicted_output = np.dot(weights, input_matrix.transpose())
    sq_error_sum = np.sum(np.square(output_vec - predicted_output))
    error = np.sqrt(float(sq_error_sum)/len(output_vec))
    return error

def shuffle(input_matrix, output_vec):
    complete_train_data = np.insert(input_matrix, 0, output_vec, axis=1)
    np.random.shuffle(complete_train_data)
    training_labels = complete_train_data[:,0]
    input_matrix = np.delete(complete_train_data,0,axis=1)
    return input_matrix, training_labels

def plot_data(y_values1, y_values2, lamda, label1, label2, axis_dim):
    plt.plot(y_values1, 'ro',label=label1)
    plt.plot(y_values2, 'b-', label = label2)
    plt.axis(axis_dim)
    plt.ylabel('RMSE')
    plt.xlabel('Model Complexity')
    plt.title('Lambda = ' + str(lamda))
    l = plt.legend()
    plt.show()



def split_training_data(input_matrix,
                        output_vec,
                        train_percent,
                        validation_percent):
    input_matrix, output_vec = shuffle(input_matrix, output_vec)
    training_data = []
    training_labels = []
    valid_data = []
    valid_labels = []
    test_data = []
    test_labels = []

    train_len = int(np.floor(float(train_percent) * len(input_matrix)))
    for i in range(train_len):
        training_data.append(input_matrix[i])
        training_labels.append(output_vec[i])

    validation_len = int(np.floor(validation_percent * len(input_matrix)))
    for i in range(train_len, train_len+validation_len):
        valid_data.append(input_matrix[i])
        valid_labels.append(output_vec[i])

    for i in range(train_len+validation_len, len(input_matrix)):
        test_data.append(input_matrix[i])
        test_labels.append(output_vec[i])

    return np.array(training_data), np.array(training_labels), np.array(valid_data), np.array(valid_labels), np.array(test_data), np.array(test_labels)

def k_means_clusters(train_data,
                     training_labels,
                     num_basis):
    try:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_basis, random_state=0).fit(train_data)
        labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_
    except Exception as e:
        print ("Error: ", str(e))
        from kmeans_implement import kmeans
        cluster_centers = kmeans(train_data, k=num_basis)
    return cluster_centers



# could be used to track K means training
# def k_means_clusters(train_data, training_labels, num_basis):
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
#                 clusters[i]['output'].append(training_labels[j])
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
                                    training_labels,
                                    lamda,
                                    num_basis):

    variance = train_data.var(axis=0) 
    sigma = variance * np.identity(len(train_data[0]))
    sigma = sigma + 0.001 * np.identity(len(train_data[0])) # Add a small quantity to avoid 0 values in variance matrix.
    sigma_inv = np.linalg.inv(sigma)

    rand_centers = k_means_clusters(train_data, training_labels, num_basis)
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
                              training_labels,
                              reg,
                              num_basis):
    Φ_trans = Φ.transpose()
    λ= reg * np.identity(num_basis)

    firstHalf= np.linalg.inv(λ + np.dot(Φ_trans, Φ))
    secondHalf= np.dot(Φ_trans, training_labels)

    weights = np.dot(firstHalf, secondHalf)
    trainingLoss = calculate_error(Φ, weights, training_labels)
    return weights, trainingLoss





# def closed_form_solution_validation_phase(valid_data, valid_labels, weights, sigma_inv, rand_centers,num_basis):
#     valid_design_matrix=np.zeros((len(valid_data),num_basis));
#     for i in range(len(valid_data)):
#         for j in range(num_basis):
#             if j==0:
#                 valid_design_matrix[i][j] = 1;
#             else:
#                 x_mu = valid_data[i]-rand_centers[j]
#                 x_mu_trans = x_mu.transpose()
#                 temp1_valid = np.dot(sigma_inv, x_mu_trans)
#                 temp2_valid = np.dot(x_mu, temp1_valid)
#                 valid_design_matrix[i][j] = np.exp(((-0.5)*temp2_valid))
#
#     predicted_output_validation = np.dot(weights, valid_design_matrix.transpose())
#     count = 0
#     for i in range(len(predicted_output_validation)):
#         if np.rint(predicted_output_validation[i]) != int(valid_labels[i]):
#             count +=1
#
#     print ("Error: ", float(count)/len(predicted_output_validation))
#     sq_error_sum_validation = np.sum(np.square(valid_labels - predicted_output_validation))
#
#     # print sq_error_sum
#     validation_error = np.sqrt(float(sq_error_sum_validation)/len(valid_data))
#     return validation_error


