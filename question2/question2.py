import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold

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
    data = pd.DataFrame(columns=['x1', 'x2', 'number'])
    for n in range(1, N + 1):
        if N > 0:
            r = np.random.choice(dataset['number'].values, p=dataset['prior'].values)
            dist = dataset[dataset['number'] == r]
            d = np.random.multivariate_normal(dist['mean'].values[0], dist['cov'].values[0])
            data = data._append({'x1': d[0], 'x2': d[1], 'number': r}, ignore_index=True)
          
    return data


def plot_scatter(data, title):
    plt.figure()
    plt.scatter(data['x1'], data['x2'], marker='o', c=data['number'])
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('2D Scatter Plot for ' + title + ' Samples')
    plt.legend()
    plt.savefig('./homework3/question2/' + title + '.png')

def plot_gaussian_ellipse(gmm, plt):
    for i in range(gmm.n_components):
        mean = gmm.means_[i]
        cov_matrix = gmm.covariances_[i]
        x, y = np.random.multivariate_normal(mean, cov_matrix, 1000).T
        plt.plot(x, y, label=f'Component {i+1}', linestyle='dashed', linewidth=2, alpha=0.7)

if __name__ == '__main__':
    m1 = [1, 2]
    m2 = [3, 2]
    m3 = [15, 6]
    m4 = [7, 10]
    cov1 = [[1, 0], [0, 3]]
    cov2 = [[3, 0], [0, 1]]
    cov3 = [[6, 0], [0, 1]]
    cov4 = [[0.5, 0], [0, 0.5]]
    
    dataset = pd.DataFrame([[1, 0.45, m1, cov1],
                            [2, 0.3, m2, cov2],
                            [3, 0.1, m3, cov3],
                            [4, 0.15, m4, cov4]], 
                           columns=['number', 'prior', 'mean', 'cov'])
    
    d10 = generate_data(dataset, 12)
    write_data(d10, './homework3/question2/d10.csv')
    d100 = generate_data(dataset, 100)
    write_data(d100, './homework3/question2/d100.csv')
    d1000 = generate_data(dataset, 1000)
    write_data(d1000, './homework3/question2/d1000.csv')
    
    d10 = read_sample_data('./homework3/question2/d10.csv')
    d100 = read_sample_data('./homework3/question2/d100.csv')
    d1000 = read_sample_data('./homework3/question2/d1000.csv')
    
    #plot 3d scatter plot
    plot_scatter(d10, '10')
    plot_scatter(d100, '100')
    plot_scatter(d1000, '1000')
   
    #fix for 10 samples - k fold issue
    d = {10: d10, 100: d100, 1000: d1000}
    for d, data in d.items():
        X = data[['x1', 'x2']]
        
        selections = []
        sumavgll = np.zeros(10)
        best_gmm = None
        for i in range(100):
            avgll = []
            kf = KFold(n_splits=10, shuffle=True, random_state=i)
            gmms = []
            for M in range(1, 11):
                ll = []
                for train_index, test_index in kf.split(X):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = data['number'].iloc[train_index], data['number'].iloc[test_index]
                    gmm = GaussianMixture(M)
                    gmm.fit(X_train)
                    ll.append(gmm.score(X_test))
                
                    
                gmms.append(gmm)        
                avgll.append(np.mean(ll))
                sumavgll[M - 1] += np.mean(ll)
            best = np.argmax(avgll) + 1
            selections.append(best)
            best_gmm = gmms[best - 1]
            
        print('For ' + str(d) + ' samples, the best M is ' + str(np.mean(selections)))
        fig = plt.figure()
        plt.hist(selections, bins=range(1, 11), align='left', rwidth=0.8)
        plt.xlabel('Model Order')
        plt.ylabel('Frequency')
        plt.title('Histogram of Best Model Order for ' + str(d) + ' Samples')
        plt.savefig('./homework3/question2/hist' + str(d) + '.png')
        plt.close(fig)
        
        #average ll for each M
        fig = plt.figure()
        plt.plot(range(1, 11), sumavgll)
        plt.xlabel('Model Order')
        plt.ylabel('Average Log Likelihood')
        plt.title('Average Log Likelihood for ' + str(d) + ' Samples')
        plt.savefig('./homework3/question2/avgll' + str(d) + '.png')
        plt.close(fig)
        
        plt.figure()
        plt.scatter(data['x1'], data['x2'], c='b', marker='o', label='Sample')
        plot_gaussian_ellipse(best_gmm, plt)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title(f'Contour Plot for {d} Samples')
        plt.legend()
        plt.savefig(f'./homework3/question2/contour_{d}.png')
        plt.close()
            
        
        
    
    
    
        