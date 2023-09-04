# This script generates the Optimisation imputations for the iris dataset.

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.impute import SimpleImputer
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

np.random.seed(2276282)
def generate_data(ground_truth, y, missing_ratio):
    X_miss = np.copy(ground_truth)
    n_samples, n_features = len(ground_truth), len(ground_truth[0])
    n_missing = int(n_samples * n_features * missing_ratio)
    indices = np.random.choice(n_samples * n_features, size=n_missing, replace=False)
    X_miss.flat[indices] = np.nan
    return(X_miss)

def score(mat, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(mat)
    labels = kmeans.predict(mat)
    score = silhouette_score(mat, labels)
    return(score)

def general_impute(X, action, missing_indices):
    action_mat = np.zeros(X.shape)
    for i in range(len(missing_indices[0])):
        action_mat[missing_indices[0][i]][missing_indices[1][i]] = action[i]
    return(X + action_mat)

def general_optimization(X_miss, n_clusters):
    missing_indices = np.where(np.isnan(X_miss))
    X_ini = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(X_miss)
    opt = lambda action: -score(general_impute(X_ini, action, missing_indices), n_clusters)
    action_opti = minimize(opt, np.zeros(len(missing_indices[1]))).x
    X_opt = general_impute(X_ini, action_opti, missing_indices)
    return(X_opt)

def run_exp(ground_truth, y, missing_ratio, n_centers, save_path):
    X_miss = generate_data(ground_truth, y, missing_ratio)
    X_opt = general_optimization(X_miss, n_centers)
    df = pd.DataFrame(X_opt)
    df.to_csv(save_path, index= False, header=False)

### IRIS DATASET

data_file = "data/datasets/raw/iris.csv"
iris = pd.read_csv(data_file)

ground_truth = iris.iloc[:, :4].values
y = iris.iloc[:, -1].values
for i,flower in enumerate(y):
    if flower=='Setosa':
        y[i]=0
    elif flower=='Versicolor':
        y[i]=1
    else:
        y[i]=2


def run_exps():
    missing_ratios = [0.05, 0.1, 0.15]
    n_centers = 3
    n_samples, n_features = len(ground_truth), len(ground_truth[0])
    random_state = 2276282
    for i, missing_ratio in enumerate(missing_ratios):
        save_path = "data/datasets/real_world_x_opts/iris_x_s"+str(n_samples)+"_f"+str(n_features)+"_c"+str(n_centers)+"_m-r"+str(int(missing_ratio*100))+"_m-nb"+str(int(n_samples*n_features*missing_ratio))+"_rs"+str(random_state)+".csv"
        run_exp(ground_truth, y, missing_ratio, n_centers, save_path)

run_exps()