
from Solutions import*

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
    sigma = sigma + 0.001 * np.identity(
        len(train_data[0]))  # Add a small quantity to avoid 0 values in variance matrix.
    sigma_inv = np.linalg.inv(sigma)

    rand_centers = generateKclusters(train_data,
                                     trainingLabels,
                                     numOfBasisFunction)
    rand_centers = np.array(rand_centers)
    design_matrix = np.zeros((len(train_data), numOfBasisFunction));

    for i in range(len(train_data)):
        for j in range(numOfBasisFunction):
            if j == 0:
                design_matrix[i][j] = 1;
            else:
                x_Minus_mu = train_data[i] - rand_centers[j]
                x_Minus_mu_trans = x_Minus_mu.transpose()
                temp1 = np.dot(sigma_inv, x_Minus_mu_trans)
                temp2 = np.dot(x_Minus_mu, temp1)
                # Equation (2) in Main
                design_matrix[i][j] = np.exp(((-0.5) * temp2))

    return design_matrix, sigma_inv, rand_centers


'''
the resulting inverse sigma, and random centers are used to create DM for validation and test sets

'''


def resultingDM(data,
                sigma_inv,
                rand_centers,
                numOfBasisFunction):
    design_matrix = np.zeros((len(data), numOfBasisFunction))
    for i in range(len(data)):
        for j in range(numOfBasisFunction):
            if j == 0:
                design_matrix[i][j] = 1;
            else:
                x_Minus_mu = data[i] - rand_centers[j]
                x_Minus_mu_trans = x_Minus_mu.transpose()
                temp1 = np.dot(sigma_inv, x_Minus_mu_trans)
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
    λ = reg * np.identity(numOfBasisFunction)

    firstHalf = np.linalg.inv(λ + np.dot(Φ_trans, Φ))
    secondHalf = np.dot(Φ_trans, trainingLabels)

    weights = np.dot(firstHalf, secondHalf)
    trainingLoss = calculate_error(Φ, weights, trainingLabels)
    return weights, trainingLoss


