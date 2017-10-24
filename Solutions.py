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
def generateKclusters(trainData,
                     trainingLabels,
                     numOfBasis):
    kmeans = KMeans(n_clusters=numOfBasis, random_state=0).fit(trainData)
    cluster_centers = kmeans.cluster_centers_
    return cluster_centers


