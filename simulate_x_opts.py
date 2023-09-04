# This script generates the Optimisation imputations for the simulations.

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

np.random.seed(2276282)
def generate_data(n_samples, n_features, n_centers, missing_ratio, random_state=None):

    ground_truth, y = make_blobs(n_samples=n_samples, centers=n_centers, n_features=n_features, random_state=random_state)
    X_miss = np.copy(ground_truth)

    n_missing = int(n_samples * n_features * missing_ratio)
    indices = np.random.choice(n_samples * n_features, size=n_missing, replace=False)
    X_miss.flat[indices] = np.nan

    return(ground_truth, X_miss, y)

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

def run_exp(n_samples, n_features, n_centers, missing_ratio):
    random_state = 2276282
    ground_truth, X_miss, y = generate_data(n_samples, n_features, n_centers, missing_ratio, random_state=random_state)
    missing_indices = np.where(np.isnan(X_miss))
    X_opt = general_optimization(X_miss, n_centers)
    save_path = "data/x_opts/x_s"+str(n_samples)+"_f"+str(n_features)+"_c"+str(n_centers)+"_m-r"+str(int(missing_ratio*100))+"_m-nb"+str(int(n_samples*n_features*missing_ratio))+"_rs"+str(random_state)+".csv"
    df = pd.DataFrame(X_opt)
    df.to_csv(save_path, index= False, header=False)

def run_exps():
    samples = [100, 500, 1000]
    features = [2, 4, 6]
    centers = [2, 3, 5]
    missing_ratios = [0.05, 0.1, 0.15]
    exp_parameters = [[a, b, c, d] for a in samples for b in features for c in centers for d in missing_ratios]
    for i, par in enumerate(exp_parameters):
        n_samples, n_features, n_centers, missing_ratio = par
        run_exp(n_samples=n_samples, n_features=n_features, n_centers=n_centers, missing_ratio=missing_ratio)

run_exps()