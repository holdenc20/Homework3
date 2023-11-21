import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import json
import string

def write_data(samples, path):
    """Writes the sample data to the file path provided.

    Args:
        samples (pandas.DataFrame): the sample data
        path (string): the path to the file containing sample data
    """
    samples.to_csv(path, sep=',', header=True, index=True)


def read_sample_data(file_path):
    """Reads the sample data from the file path provided.

    Args:
        file_path (string): the path to the file containing sample data

    Returns:
        pandas.DataFrame: the sample data
    """
    data = pd.read_csv(file_path, sep=',', header=0, index_col=0)
    return data

def generate_data(dataset, N):
    data = pd.DataFrame(columns=['x1', 'x2', 'x3', 'class'])
    for n in range(1, N + 1):
        if N > 0:
            r = np.random.randint(1, len(dataset)+1)
            dist = dataset.where(dataset['class'] == r).dropna()
            d = np.random.multivariate_normal(dist['mean'].values[0], dist['cov'].values[0])
            data = data._append({'x1': d[0], 'x2': d[1], 'x3': d[2], 'class': r}, ignore_index=True)
            
    return data

def classifier(samples, dataset):
    predictions = []
    for i in range(len(samples)):
        sample = samples.iloc[i]
        pred = predict(sample, dataset)
        predictions.append(pred)
        
        
    samples['prediction'] = predictions
    
    return samples


def predict(sample, dataset):
    x = sample[['x1', 'x2', 'x3']].to_numpy()
    
    discriminants = []
    
    for i in range(len(dataset['mean'].values)):
        mean = dataset['mean'].values[i]
        cov = dataset['cov'].values[i]
        prior = dataset['prior'].values[i]
        
        pdf = multivariate_normal.pdf(x, mean, cov)
        d = np.log(prior) + np.log(pdf)
        discriminants.append(d)
        
    
    i = np.argmax(discriminants)
    return i + 1

def calc_error(pred):
    error = 0
    for i in range(len(pred)):
        if pred['class'].values[i] != pred['prediction'].values[i]:
            error += 1
    return error / len(pred)

def theoretical_classifier(data, dataset):
    pred = classifier(data, dataset)
    #calculate accuracy using min p error
    error = calc_error(pred)
    print("Theoretical minimum p(error) =", error)
    #confusion matrix
    cm = metrics.confusion_matrix(pred['class'], pred['prediction'])
    print("Confusion Matrix:")
    print(cm)
    return pred

def plot_3d_scatter(predictions, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    incorrect = predictions['class'] != predictions['prediction']
    for i in range(1, 5):
        correct = (predictions['class'] == predictions['prediction']) & (predictions['class'] == i)
        
        ax.scatter(predictions[correct]['x1'], predictions[correct]['x2'], predictions[correct]['x3'] ,marker='o', label=predictions[correct]['prediction'].values[0])

    ax.scatter(predictions[incorrect]['x1'], predictions[incorrect]['x2'], predictions[incorrect]['x3'], c='red', marker='x', label='Incorrect')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    ax.set_title('3D Scatter Plot of Predictions for ' + title + ' Samples')
    ax.legend()
    plt.savefig('./homework3/question1/' + title + '.png')

def mlp(numPerc, k, x, labels, numLabels):
    N = len(x)
    numValIters = 10
    y = np.zeros((numLabels, len(x)))

    for i in range(numLabels):
        y[i, :] = (labels == i + 1)

    partSize = N // k
    partInd = np.concatenate((np.arange(0, N, partSize), [len(x)]))

    avgPFE = np.zeros(numPerc)
    
    #M is the number of perceptrons in the hidden layer
    #this is for the model order selection
    for M in range(1, numPerc + 1):
        pFE = np.zeros(k)
        #k-fold cross validation
        #k is the number of folds
        for part in range(k):
            index_val = np.arange(partInd[part], partInd[part + 1])

            index_train = np.setdiff1d(np.arange(N), index_val)
            min_score = 1e6
            for i in range(5):
                net = MLPClassifier(hidden_layer_sizes=(M,), max_iter=10000, activation='relu')
                net.fit(x[index_train], labels[index_train])
                y_val = net.predict(x[index_val])
                score = 1 - accuracy_score(labels[index_val], y_val)
                min_score = min(min_score, score)
                
            pFE[part] = min_score
        #used for model order selection
        avgPFE[M - 1] = np.mean(pFE)

    #minimum cross entropy loss model    
    optM = np.argmin(avgPFE) + 1

    finalnet = MLPClassifier(hidden_layer_sizes=(optM,), max_iter=10000)
    finalnet.fit(x, labels)

    pFEFinal = np.zeros(numValIters)

    for i in range(numValIters):
        y_val = finalnet.predict(x)
        pFEFinal[i] = 1 - accuracy_score(labels, y_val)

    minPFE = np.min(pFEFinal)

    return finalnet, minPFE, optM, {'M': np.arange(1, numPerc + 1), 'mPFE': avgPFE}

if __name__ == '__main__':
    m1 = [1, 2, 5]
    m2 = [1, 5, 6]
    m3 = [7, 3, 9]
    m4 = [2, 7, 2]
    cov1 = [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
    cov2 = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    cov3 = [[6, 0, 0], [0, 6, 0], [0, 0, 6]]
    cov4 = [[12, 0, 0], [0, 6, 0], [0, 0, 3]]
    
    dataset = pd.DataFrame([[1, 0.25, m1, cov1],
                            [2, 0.25, m2, cov2],
                            [3, 0.25, m3, cov3],
                            [4, 0.25, m4, cov4]], 
                           columns=['class', 'prior', 'mean', 'cov'])
    
    #creating MLP Structure - 2 layers
    #P = number of perceptrons in the hidden layer
    #activation function - ISRU, SmoothReLU, or ELU etc
    #output layer - softmax - ensure that all outputs are positive and sum to 1
    #the best number of perceptrons will be found through cross validation
    
    #generate data
    '''
    test_data = generate_data(dataset, 100000)
    write_data(test_data, './homework3/question1/testdata.csv')
    
    d100 = generate_data(dataset, 100)
    write_data(d100, './homework3/question1/d100.csv')
    d500 = generate_data(dataset, 500)
    write_data(d500, './homework3/question1/d500.csv')
    d1000 = generate_data(dataset, 1000)
    write_data(d1000, './homework3/question1/d1000.csv')
    d5000 = generate_data(dataset, 5000)
    write_data(d5000, './homework3/question1/d5000.csv')
    d10000 = generate_data(dataset, 10000)
    write_data(d10000, './homework3/question1/d10000.csv')
    '''
    
    test_data = read_sample_data('./homework3/question1/testdata.csv')
    d100 = read_sample_data('./homework3/question1/d100.csv')
    d500 = read_sample_data('./homework3/question1/d500.csv')
    d1000 = read_sample_data('./homework3/question1/d1000.csv')
    d5000 = read_sample_data('./homework3/question1/d5000.csv')
    d10000 = read_sample_data('./homework3/question1/d10000.csv')
    
    #theoretical optimal classifier
    '''
    predictions = theoretical_classifier(test_data, dataset)
    
    p100 = theoretical_classifier(d100, dataset)
    p500 = theoretical_classifier(d500, dataset)
    p1000 = theoretical_classifier(d1000, dataset)
    p5000 = theoretical_classifier(d5000, dataset)
    p10000 = theoretical_classifier(d10000, dataset)
    
    #plot 3d scatter plot
    plot_3d_scatter(predictions, '100k')
    plot_3d_scatter(p100, '100')
    plot_3d_scatter(p500, '500')
    plot_3d_scatter(p1000, '1k')
    plot_3d_scatter(p5000, '5k')
    plot_3d_scatter(p10000, '10k')
    '''
    '''
    datasets = {100: d100, 500: d500, 1000: d1000, 5000: d5000, 10000:d10000}
    numPerceptrons = 10
    k = 10
    D = pd.DataFrame(columns=['d', 'net', 'minPFE', 'optM', 'stats', 'pFE'])
    valData = {}
    predictions = {}
    
    
    for d, data in datasets.items():
        valData[d] = {}
        
        X_train = data[['x1', 'x2', 'x3']].to_numpy()
        y_train = data['class'].to_numpy()
        net, minPFE, optM, stats = mlp(numPerceptrons, k, X_train, y_train, 4)
        X_val = test_data[['x1', 'x2', 'x3']].to_numpy()
        yVal = net.predict(X_val)
        test_data['decisions'] = yVal
        pFE = np.sum(test_data['decisions'] != test_data['class']) / len(test_data)
        M = stats['M']
        mPFE = stats['mPFE']
        D = D._append({'d': d, 'net': net, 'minPFE': minPFE, 'optM': optM, 'M': M, 'mPFE': mPFE, 'pFE': pFE}, ignore_index=True)
        print(f"NN pFE, N={len(data)}: Error={100 * pFE:.2f}%")
        
        predictions[d] = yVal
    
    write_data(D, './homework3/question1/data.csv')
    '''
    D = read_sample_data('./homework3/question1/data.csv')
    print(D)

    fig, ax = plt.subplots()

    for i, row in D.iterrows():
        print(row['d'])
        print(row['mPFE'])
        M = list(range(1, 11))
        #dealing with weird parsing issue
        row['mPFE'] = [float(value) for value in row['mPFE'].replace('  ', ' ').replace(" ", ',').removeprefix('[').removesuffix(']').split(',') if value]
        ax.plot(M, row['mPFE'], label=row['d'])

    ax.set_xlabel('Number of Perceptrons')
    ax.set_ylabel('Mean Probability of Error')
    ax.set_title('Mean Probability of Error vs Number of Perceptrons')
    ax.legend()
    plt.savefig('./homework3/question1/perceptrons_vs_error.png')
    
    #plot number of samples vs pFE
    fig, ax = plt.subplots()
    ax.plot(D['d'], D['pFE'])
    ax.set_xscale('log')
    #plot the theoretical minimum p(error)
    ax.axhline(y=0.14683, color='r', linestyle='-', label='Theoretical Minimum')
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Probability of Error')
    ax.set_title('Probability of Error vs Number of Samples')
    plt.legend()
    plt.savefig('./homework3/question1/samples_vs_error.png')
    
    #plotting classifier decisions for each dataset
    '''
    for i, row in D.iterrows():
        p = predictions[row['d']]
        test_data['prediction'] = p
        plot_3d_scatter(test_data, str(row['d']) + ' NN')
    '''
    
    #Sample Size vs. Optimal Number of Perceptrons
    fig, ax = plt.subplots()
    ax.plot(D['d'], D['optM'])
    ax.set_xscale('log')
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Optimal Number of Perceptrons')
    ax.set_title('Optimal Number of Perceptrons vs Number of Samples')
    plt.savefig('./homework3/question1/samples_vs_optimalperceptrons.png')