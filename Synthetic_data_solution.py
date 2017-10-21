import numpy as np
# from sklearn.cluster import KMeans
#
#
#
#
# #
# # def closedForm_weightTraining(train_data, trainingLabels,lamda, num_basis):
# #     variance = train_data.var(axis=0)
# #     sigma = variance * np.identity(len(train_data[0]))
# #     sigma = sigma + lamda * np.identity(len(train_data[0]))
# #     sigma_inv = np.linalg.inv(sigma)
# #
# #     # rand_indices = np.random.randint(0, len(train_data), size=(1,num_basis))
# #
# #     # rand_centers = []
# #     rand_centers = k_means_clusters(train_data, trainingLabels, num_basis)
# #     # for i in range(len(rand_indices[0])):
# #     #     index = rand_indices[0][i]
# #     #     rand_centers.append(train_data[index])
# #
# #     rand_centers = np.array(rand_centers)
# #     design_matrix=np.zeros((len(train_data),num_basis));
# #
# #     for i in range(len(train_data)):
# #         for j in range(num_basis):
# #             if j==0:
# #                 design_matrix[i][j] = 1;
# #             else:
# #                 x_mu = train_data[i]-rand_centers[j]
# #                 x_mu_trans = x_mu.transpose()
# #                 temp1 = np.dot(sigma_inv, x_mu_trans)
# #                 temp2 = np.dot(x_mu, temp1)
# #                 design_matrix[i][j] = np.exp(((-0.5)*temp2))
# #
# #     design_matrix_trans = design_matrix.transpose()
# #     regularisation_mat = lamda * np.identity(num_basis)
# #     pinv_temp = np.dot(design_matrix_trans, design_matrix) + regularisation_mat
# #     pinv = np.linalg.inv(pinv_temp)
# #     out_temp = np.dot(design_matrix_trans, trainingLabels)
# #     weights = np.dot(pinv, out_temp)
# #
# #     predicted_output = np.dot(weights, design_matrix_trans)
# #     sq_error_sum = np.sum(np.square(trainingLabels - predicted_output))
# #
# #     train_error = np.sqrt(float(sq_error_sum)/len(train_data))
# #     return design_matrix, rand_centers, sigma_inv, weights, train_error
#
# def main():
#     in_filename = 'input.csv'
#     out_filename = 'output.csv'
#     feature_mat, output_labels = read_data_file(in_filename, out_filename)
#     trainingSets, trainingLabels, validationSets, validationLabels, testSets, testLabels = partition(feature_mat, output_labels, 0.8, 0.1)
#     print trainingSets.shape, validationSets.shape, testSets.shape
#     for num_basis in range(2, 11):
#         print "Num Basis Functions: ", num_basis
#         design_matrix, rbf_centers, sigma_inv, weights, rmse_train = closedForm_weightTraining(trainingSets, trainingLabels, 0.3, num_basis)
#         rmse_validation = closed_form_solution_validation_phase(validationSets, validationLabels, weights, sigma_inv, rbf_centers, num_basis)
#         # print "Weights: ", weights
#         print "RMSE Train: ", rmse_train
#         print "RMSE Validation: ", rmse_validation
#         print closed_form_solution_validation_phase(testSets, testLabels, weights, sigma_inv, rbf_centers, num_basis)

#
# main()