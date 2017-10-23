import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

'''
Read in Files 
'''
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

'''
Calculate RMSE errors
'''
def calculate_error(input_matrix,
                    weights,
                    output):

    predicted_output = np.dot(weights, input_matrix.transpose())
    squared_error_sum = np.sum(np.square(output - predicted_output))
    rmse = np.sqrt(float(squared_error_sum)/len(output))
    return rmse


'''
Random Shuffle data set at the starting of each epoch
'''
def shuffle(input_matrix, output):
    trainingData = np.insert(input_matrix, 0, output, axis=1)
    np.random.shuffle(trainingData)
    trainingLabels = trainingData[:,0]
    input_matrix = np.delete(trainingData,0,axis=1)
    return input_matrix, trainingLabels


''' Partition matrix into 8 : 1 : 1'''
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



''' Generate K clusters with given numOf BasisFunction'''
def generateKclusters(train_data,
                     trainingLabels,
                     numOfBasisFunction):
    kmeans = KMeans(n_clusters=numOfBasisFunction, random_state=0).fit(train_data)
    cluster_centers = kmeans.cluster_centers_
    return cluster_centers



'''
Create Closed Form DM based on training data, 
the resulting inverse sigma, and random centers are used to create DM for validation and test sets
'''
def priorDM(train_data,
            trainingLabels,
            lamda,
            numOfBasisFunction):

    variance = train_data.var(axis=0) 
    sigma = variance * np.identity(len(train_data[0]))
    sigma = sigma + 0.001 * np.identity(len(train_data[0])) # Add a small quantity to avoid 0 values in variance matrix.
    sigma_inv = np.linalg.inv(sigma)

    rand_centers = generateKclusters(train_data,
                                     trainingLabels,
                                     numOfBasisFunction)
    rand_centers = np.array(rand_centers)
    design_matrix=np.zeros((len(train_data),numOfBasisFunction));

    for i in range(len(train_data)):
        for j in range(numOfBasisFunction):
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
                numOfBasisFunction):

    design_matrix = np.zeros((len(data),numOfBasisFunction))
    for i in range(len(data)):
        for j in range(numOfBasisFunction):
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
 w∗ = inv((λI + transpose(Φ) * Φ))* transpose(Φ)*y at closedForm solution, nothing too complex
'''

def cF_weightAdjustment(Φ, sigma_inv,
                        trainingLabels,
                        reg,
                        numOfBasisFunction):
    Φ_trans = Φ.transpose()
    λ= reg * np.identity(numOfBasisFunction)

    firstHalf= np.linalg.inv(λ + np.dot(Φ_trans, Φ))
    secondHalf= np.dot(Φ_trans, trainingLabels)

    weights = np.dot(firstHalf, secondHalf)
    trainingLoss = calculate_error(Φ, weights, trainingLabels)
    return weights, trainingLoss


