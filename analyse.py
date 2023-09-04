# This script generates the figures and tables used in the thesis.

from sklearn.datasets import make_blobs
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
pio.kaleido.scope.mathjax = None
import warnings
warnings.filterwarnings("ignore")

###############################
### DATA GENERATION EXAMPLE ###
###############################

def generate_data(n_samples, n_features, n_centers, missing_ratio, random_state=None):
    ground_truth, y = make_blobs(n_samples=n_samples, centers=n_centers, n_features=n_features, random_state=random_state)
    X_miss = np.copy(ground_truth)
    n_missing = int(n_samples * n_features * missing_ratio)
    indices = np.random.choice(n_samples * n_features, size=n_missing, replace=False)
    X_miss.flat[indices] = np.nan
    return(ground_truth, X_miss, y)

n_samples = 100
n_features = 2
n_centers = 2
missing_ratio = .15
random_state = 2276282
ground_truth, X_miss, y = generate_data(n_samples, n_features, n_centers, missing_ratio, random_state=random_state)

full_data = np.column_stack((ground_truth, y))
df = pd.DataFrame(full_data, columns=['Feature 1', 'Feature 2', 'True Cluster'])
df['True Cluster'] = df['True Cluster'].astype(int)
df['True Cluster'] = df['True Cluster'].astype(str)

fig = px.scatter(df, x="Feature 1", y="Feature 2", color="True Cluster",
                 title="Simulated data", template='none')

pio.write_image(fig, 'figures/generation_data.pdf')

#################################
### SIMULATIONS RMSE ANALYSIS ###
#################################

samples = [100, 500, 1000]
features = [2, 4, 6]
centers = [2, 3, 5]
missing_ratios = [0.05, 0.1, 0.15]
exp_parameters = [[a, b, c, d] for a in samples for b in features for c in centers for d in missing_ratios]
methods = ['Mean', 'Median', 'Most Frequent', 'K-Nearest Neighbours', 'Optimisation']

np.random.seed(2276282)

def load_X_opt(n_samples, n_features, n_centers, missing_ratio):
    random_state = 2276282
    path = "data/x_opts/x_s"+str(n_samples)+"_f"+str(n_features)+"_c"+str(n_centers)+"_m-r"+str(int(missing_ratio*100))+"_m-nb"+str(int(n_samples*n_features*missing_ratio))+"_rs"+str(random_state)+".csv"
    return(pd.read_csv(path, header=None))

def load_data(n_samples, n_features, n_centers, missing_ratio, random_state=2276282):

    ground_truth, y = make_blobs(n_samples=n_samples, centers=n_centers, n_features=n_features, random_state=random_state)
    X_miss = np.copy(ground_truth)

    n_missing = int(n_samples * n_features * missing_ratio)
    indices = np.random.choice(n_samples * n_features, size=n_missing, replace=False)
    X_miss.flat[indices] = np.nan
    X_opt = load_X_opt(n_samples, n_features, n_centers, missing_ratio)
    X_mean = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(X_miss)
    X_med = SimpleImputer(missing_values=np.nan, strategy='median').fit_transform(X_miss)
    X_mf = SimpleImputer(missing_values=np.nan, strategy='most_frequent').fit_transform(X_miss)
    X_knn = KNNImputer(weights="uniform").fit_transform(X_miss)
    Xs = {}
    Xs['Mean'], Xs['Median'], Xs['Most Frequent'], Xs['K-Nearest Neighbours'], Xs['Optimisation'] = X_mean, X_med, X_mf, X_knn, X_opt
    return(ground_truth, X_miss, y, Xs)

def calculate_rmse(x,y):
    return(np.sqrt(mean_squared_error(x, y)))

def calculate_rmses():
    rmses = {}
    for par in exp_parameters:
        rmses[str(par)] = {}
    for par in exp_parameters:
        n_samples, n_features, n_centers, missing_ratio = par
        ground_truth, X_miss, y, Xs = load_data(n_samples, n_features, n_centers, missing_ratio, random_state=2276282)

        for method in methods:
            rmses[str(par)][method] = calculate_rmse(ground_truth, Xs[method])

    return(rmses)

# GRAPH 1 : RMSE count

def rmse_count():
    counts = {}
    for method in methods:
        counts[method] = 0
    rmses = calculate_rmses()
    for par in exp_parameters:
        rmse_dic = rmses[str(par)]
        min_key = min(rmse_dic, key=rmse_dic.get)
        counts[min_key] += 1
    return(counts)

counts_df = pd.DataFrame(list(rmse_count().items()), columns=['Method', 'Count'])
fig = px.histogram(counts_df, x="Method", y = 'Count', title="Count of smallest RMSE with ground truth", template='none', text_auto=True)
fig.update_layout(
    yaxis_title="Count",
)
pio.write_image(fig, 'figures/rmse_count.pdf')

# GRAPH 2 : RMSE by Nb of Samples

rmses = calculate_rmses()
rmses_by_samples = {}
rmses_by_samples['100'], rmses_by_samples['500'], rmses_by_samples['1000'] = [], [], []
for i, key in enumerate(rmses):
    rmse = rmses[key]
    if i<27:
        rmses_by_samples['100'].append(rmse)
    elif i>53:
        rmses_by_samples['1000'].append(rmse)
    else:
        rmses_by_samples['500'].append(rmse)

def sample_df(method):
    sample_dic = {
        '100': [x[method] for x in rmses_by_samples['100']],
        '500': [x[method] for x in rmses_by_samples['500']],
        '1000': [x[method] for x in rmses_by_samples['1000']]
    }
    data_tuples = [(value, key) for key, values in sample_dic.items() for value in values]
    return(pd.DataFrame(data_tuples, columns=['RMSE', 'Number of samples']))
fig = go.Figure(layout = go.Layout(xaxis=dict(type='category')))
fig.add_trace(go.Box(y=sample_df('Mean')['RMSE'], x=sample_df('Mean')['Number of samples'], name='Mean', marker_color='black'))
fig.add_trace(go.Box(y=sample_df('Median')['RMSE'], x=sample_df('Median')['Number of samples'], name='Median', marker_color='brown'))
fig.add_trace(go.Box(y=sample_df('Most Frequent')['RMSE'], x=sample_df('Most Frequent')['Number of samples'], name='Most Frequent', marker_color='red'))
fig.add_trace(go.Box(y=sample_df('K-Nearest Neighbours')['RMSE'], x=sample_df('K-Nearest Neighbours')['Number of samples'], name='K-Nearest Neighbours', marker_color='orange'))
fig.add_trace(go.Box(y=sample_df('Optimisation')['RMSE'], x=sample_df('Optimisation')['Number of samples'], name='Optimisation', marker_color='green'))
fig.update_layout(
    title='RMSE with ground truth by Number of Samples',
    xaxis_title='Number of Samples',
    yaxis_title='RMSE with ground truth',
    boxmode='group',
    template='none',
    legend_title_text='Method'
)
pio.write_image(fig, 'figures/rmse_by_samples.pdf')

# GRAPH 3 : RMSE by Nb of Features

rmses = calculate_rmses()
rmses_by_feat = {}
rmses_by_feat['2'], rmses_by_feat['4'], rmses_by_feat['6'] = [], [], []
for i, key in enumerate(rmses):
    feat = key.strip('[]').split(', ')[1]
    rmses_by_feat[feat].append(rmses[key])

def feat_df(method):
    feat_dic = {
        '2': [x[method] for x in rmses_by_feat['2']],
        '4': [x[method] for x in rmses_by_feat['4']],
        '6': [x[method] for x in rmses_by_feat['6']]
    }
    data_tuples = [(value, key) for key, values in feat_dic.items() for value in values]
    return(pd.DataFrame(data_tuples, columns=['RMSE', 'Number of Features']))
fig = go.Figure()
fig.add_trace(go.Box(y=feat_df('Mean')['RMSE'], x=feat_df('Mean')['Number of Features'], name='Mean', marker_color='black'))
fig.add_trace(go.Box(y=feat_df('Median')['RMSE'], x=feat_df('Median')['Number of Features'], name='Median', marker_color='brown'))
fig.add_trace(go.Box(y=feat_df('Most Frequent')['RMSE'], x=feat_df('Most Frequent')['Number of Features'], name='Most Frequent', marker_color='red'))
fig.add_trace(go.Box(y=feat_df('K-Nearest Neighbours')['RMSE'], x=feat_df('K-Nearest Neighbours')['Number of Features'], name='K-Nearest Neighbours', marker_color='orange'))
fig.add_trace(go.Box(y=feat_df('Optimisation')['RMSE'], x=feat_df('Optimisation')['Number of Features'], name='Optimisation', marker_color='green'))
fig.update_layout(
    title='RMSE with ground truth by Number of Features',
    xaxis_title='Number of Features',
    yaxis_title='RMSE with ground truth',
    boxmode='group',
    template='none',
    legend_title_text='Method'
)
pio.write_image(fig, 'figures/rmse_by_features.pdf')

# GRAPH 4 : RMSE by Nb of true clusters

rmses = calculate_rmses()
rmses_by_tc = {}
rmses_by_tc['2'], rmses_by_tc['3'], rmses_by_tc['5'] = [], [], []
for i, key in enumerate(rmses):
    tc = key.strip('[]').split(', ')[2]
    rmses_by_tc[tc].append(rmses[key])

def tc_df(method):
    feat_dic = {
        '2': [x[method] for x in rmses_by_tc['2']],
        '3': [x[method] for x in rmses_by_tc['3']],
        '5': [x[method] for x in rmses_by_tc['5']]
    }
    data_tuples = [(value, key) for key, values in feat_dic.items() for value in values]
    return(pd.DataFrame(data_tuples, columns=['RMSE', 'Number of True Clusters']))

fig = go.Figure(layout = go.Layout(xaxis=dict(type='category')))
fig.add_trace(go.Box(y=tc_df('Mean')['RMSE'], x=tc_df('Mean')['Number of True Clusters'], name='Mean', marker_color='black'))
fig.add_trace(go.Box(y=tc_df('Median')['RMSE'], x=tc_df('Median')['Number of True Clusters'], name='Median', marker_color='brown'))
fig.add_trace(go.Box(y=tc_df('Most Frequent')['RMSE'], x=tc_df('Most Frequent')['Number of True Clusters'], name='Most Frequent', marker_color='red'))
fig.add_trace(go.Box(y=tc_df('K-Nearest Neighbours')['RMSE'], x=tc_df('K-Nearest Neighbours')['Number of True Clusters'], name='K-Nearest Neighbours', marker_color='orange'))
fig.add_trace(go.Box(y=tc_df('Optimisation')['RMSE'], x=tc_df('Optimisation')['Number of True Clusters'], name='Optimisation', marker_color='green'))
fig.update_layout(
    title='RMSE with ground truth by Number of True Clusters',
    xaxis_title='Number of True Clusters',
    yaxis_title='RMSE with ground truth',
    boxmode='group',
    template='none',
    legend_title_text='Method'
)
pio.write_image(fig, 'figures/rmse_by_clusters.pdf')

# GRAPH 5 : RMSE by Missing ratio

rmses = calculate_rmses()
rmses_by_mr = {}
rmses_by_mr['0.05'], rmses_by_mr['0.1'], rmses_by_mr['0.15'] = [], [], []
for i, key in enumerate(rmses):
    mr = key.strip('[]').split(', ')[3]
    rmses_by_mr[mr].append(rmses[key])

def mr_df(method):
    feat_dic = {
        '0.05': [x[method] for x in rmses_by_mr['0.05']],
        '0.1': [x[method] for x in rmses_by_mr['0.1']],
        '0.15': [x[method] for x in rmses_by_mr['0.15']]
    }
    data_tuples = [(value, key) for key, values in feat_dic.items() for value in values]
    return(pd.DataFrame(data_tuples, columns=['RMSE', 'Missing Ratio']))

fig = go.Figure(layout = go.Layout(xaxis=dict(type='category')))
fig.add_trace(go.Box(y=mr_df('Mean')['RMSE'], x=mr_df('Mean')['Missing Ratio'], name='Mean', marker_color='black'))
fig.add_trace(go.Box(y=mr_df('Median')['RMSE'], x=mr_df('Median')['Missing Ratio'], name='Median', marker_color='brown'))
fig.add_trace(go.Box(y=mr_df('Most Frequent')['RMSE'], x=mr_df('Most Frequent')['Missing Ratio'], name='Most Frequent', marker_color='red'))
fig.add_trace(go.Box(y=mr_df('K-Nearest Neighbours')['RMSE'], x=mr_df('K-Nearest Neighbours')['Missing Ratio'], name='K-Nearest Neighbours', marker_color='orange'))
fig.add_trace(go.Box(y=mr_df('Optimisation')['RMSE'], x=mr_df('Optimisation')['Missing Ratio'], name='Optimisation', marker_color='green'))
fig.update_layout(
    title='RMSE with ground truth by Missing Ratio',
    xaxis_title='Missing Ratio',
    yaxis_title='RMSE with ground truth',
    boxmode='group',
    template='none',
    legend_title_text='Method'
)
pio.write_image(fig, 'figures/rmse_by_mr.pdf')

#######################################
### SIMULATIONS CLUSTERING ANALYSIS ###
#######################################

samples = [100, 500, 1000]
features = [2, 4, 6]
centers = [2, 3, 5]
missing_ratios = [0.05, 0.1, 0.15]
exp_parameters = [[a, b, c, d] for a in samples for b in features for c in centers for d in missing_ratios]
methods = ['Mean', 'Median', 'Most Frequent', 'K-Nearest Neighbours', 'Optimisation', 'Ground Truth']

np.random.seed(2276282)

def load_X_opt(n_samples, n_features, n_centers, missing_ratio):
    random_state = 2276282
    path = "data/x_opts/x_s"+str(n_samples)+"_f"+str(n_features)+"_c"+str(n_centers)+"_m-r"+str(int(missing_ratio*100))+"_m-nb"+str(int(n_samples*n_features*missing_ratio))+"_rs"+str(random_state)+".csv"
    return(pd.read_csv(path, header=None))

def load_data(n_samples, n_features, n_centers, missing_ratio, random_state=2276282):

    ground_truth, y = make_blobs(n_samples=n_samples, centers=n_centers, n_features=n_features, random_state=random_state)
    X_miss = np.copy(ground_truth)

    n_missing = int(n_samples * n_features * missing_ratio)
    indices = np.random.choice(n_samples * n_features, size=n_missing, replace=False)
    X_miss.flat[indices] = np.nan
    X_opt = load_X_opt(n_samples, n_features, n_centers, missing_ratio)
    X_mean = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(X_miss)
    X_med = SimpleImputer(missing_values=np.nan, strategy='median').fit_transform(X_miss)
    X_mf = SimpleImputer(missing_values=np.nan, strategy='most_frequent').fit_transform(X_miss)
    X_knn = KNNImputer(weights="uniform").fit_transform(X_miss)
    Xs = {}
    Xs['Mean'], Xs['Median'], Xs['Most Frequent'], Xs['K-Nearest Neighbours'], Xs['Optimisation'], Xs['Ground Truth'] = X_mean, X_med, X_mf, X_knn, X_opt, ground_truth
    return(ground_truth, X_miss, y, Xs)

# GRAPH 1 : Sil

def calculate_sil(x, nc):
    kmeans = KMeans(n_clusters=nc)
    kmeans.fit(x)
    return(silhouette_score(x, kmeans.labels_))
    

def calculate_sils():
    sils = {}
    for par in exp_parameters:
        sils[str(par)] = {}
    for par in exp_parameters:
        n_samples, n_features, n_centers, missing_ratio = par
        ground_truth, X_miss, y, Xs = load_data(n_samples, n_features, n_centers, missing_ratio, random_state=2276282)

        for method in methods:
            sils[str(par)][method] = calculate_sil(Xs[method], n_centers)

    return(sils)

sils = calculate_sils()

sils_by_mr = {}
sils_by_mr['0.05'], sils_by_mr['0.1'], sils_by_mr['0.15'] = [], [], []
for i, key in enumerate(sils):
    mr = key.strip('[]').split(', ')[3]
    sils_by_mr[mr].append(sils[key])

def mr_df(method):
    feat_dic = {
        '0.05': [x[method] for x in sils_by_mr['0.05']],
        '0.1': [x[method] for x in sils_by_mr['0.1']],
        '0.15': [x[method] for x in sils_by_mr['0.15']]
    }
    data_tuples = [(value, key) for key, values in feat_dic.items() for value in values]
    return(pd.DataFrame(data_tuples, columns=['K-Means Silhouette Score', 'Missing Ratio']))

fig = go.Figure(layout = go.Layout(xaxis=dict(type='category')))
fig.add_trace(go.Box(y=mr_df('Mean')['K-Means Silhouette Score'], x=mr_df('Mean')['Missing Ratio'], name='Mean', marker_color='black'))
fig.add_trace(go.Box(y=mr_df('Median')['K-Means Silhouette Score'], x=mr_df('Median')['Missing Ratio'], name='Median', marker_color='brown'))
fig.add_trace(go.Box(y=mr_df('Most Frequent')['K-Means Silhouette Score'], x=mr_df('Most Frequent')['Missing Ratio'], name='Most Frequent', marker_color='red'))
fig.add_trace(go.Box(y=mr_df('K-Nearest Neighbours')['K-Means Silhouette Score'], x=mr_df('K-Nearest Neighbours')['Missing Ratio'], name='K-Nearest Neighbours', marker_color='orange'))
fig.add_trace(go.Box(y=mr_df('Optimisation')['K-Means Silhouette Score'], x=mr_df('Optimisation')['Missing Ratio'], name='Optimisation', marker_color='green'))
fig.add_trace(go.Box(y=mr_df('Ground Truth')['K-Means Silhouette Score'], x=mr_df('Ground Truth')['Missing Ratio'], name='Ground Truth', marker_color='blue'))
fig.update_layout(
    title='K-Means Silhouette Score by Missing Ratio',
    xaxis_title='Missing Ratio',
    yaxis_title='K-Means Silhouette Score',
    boxmode='group',
    template='none',
    legend_title_text='Method'
)
pio.write_image(fig, 'figures/sil_by_mr.pdf')

# GRAPH 2 : NMI

def calculate_nmi(x, y, nc):
    kmeans = KMeans(n_clusters=nc)
    kmeans.fit(x)
    return(normalized_mutual_info_score(y, kmeans.labels_))
    

def calculate_nmis():
    nmis = {}
    for par in exp_parameters:
        nmis[str(par)] = {}
    for par in exp_parameters:
        n_samples, n_features, n_centers, missing_ratio = par
        ground_truth, X_miss, y, Xs = load_data(n_samples, n_features, n_centers, missing_ratio, random_state=2276282)

        for method in methods:
            nmis[str(par)][method] = calculate_nmi(Xs[method], y, n_centers)

    return(nmis)

nmis = calculate_nmis()

nmis_by_mr = {}
nmis_by_mr['0.05'], nmis_by_mr['0.1'], nmis_by_mr['0.15'] = [], [], []
for i, key in enumerate(nmis):
    mr = key.strip('[]').split(', ')[3]
    nmis_by_mr[mr].append(nmis[key])

def mr_df(method):
    feat_dic = {
        '0.05': [x[method] for x in nmis_by_mr['0.05']],
        '0.1': [x[method] for x in nmis_by_mr['0.1']],
        '0.15': [x[method] for x in nmis_by_mr['0.15']]
    }
    data_tuples = [(value, key) for key, values in feat_dic.items() for value in values]
    return(pd.DataFrame(data_tuples, columns=['K-Means Normalized Mutual Info Score', 'Missing Ratio']))

fig = go.Figure(layout = go.Layout(xaxis=dict(type='category')))
fig.add_trace(go.Box(y=mr_df('Mean')['K-Means Normalized Mutual Info Score'], x=mr_df('Mean')['Missing Ratio'], name='Mean', marker_color='black'))
fig.add_trace(go.Box(y=mr_df('Median')['K-Means Normalized Mutual Info Score'], x=mr_df('Median')['Missing Ratio'], name='Median', marker_color='brown'))
fig.add_trace(go.Box(y=mr_df('Most Frequent')['K-Means Normalized Mutual Info Score'], x=mr_df('Most Frequent')['Missing Ratio'], name='Most Frequent', marker_color='red'))
fig.add_trace(go.Box(y=mr_df('K-Nearest Neighbours')['K-Means Normalized Mutual Info Score'], x=mr_df('K-Nearest Neighbours')['Missing Ratio'], name='K-Nearest Neighbours', marker_color='orange'))
fig.add_trace(go.Box(y=mr_df('Optimisation')['K-Means Normalized Mutual Info Score'], x=mr_df('Optimisation')['Missing Ratio'], name='Optimisation', marker_color='green'))
fig.add_trace(go.Box(y=mr_df('Ground Truth')['K-Means Normalized Mutual Info Score'], x=mr_df('Ground Truth')['Missing Ratio'], name='Ground Truth', marker_color='blue'))
fig.update_layout(
    title='K-Means Normalized Mutual Info Score with true labels by Missing Ratio',
    xaxis_title='Missing Ratio',
    yaxis_title='K-Means Normalized Mutual Info Score with true labels',
    boxmode='group',
    template='none',
    legend_title_text='Method'
)
pio.write_image(fig, 'figures/nmi_by_mr.pdf')

###########################
### REAL WORLD ANALYSIS ###
###########################

missing_ratios = [0.05, 0.1, 0.15]

np.random.seed(2276282)

def generate_data(ground_truth, missing_ratio):
    X_miss = np.copy(ground_truth)
    n_samples, n_features = len(ground_truth), len(ground_truth[0])
    n_missing = int(n_samples * n_features * missing_ratio)
    indices = np.random.choice(n_samples * n_features, size=n_missing, replace=False)
    X_miss.flat[indices] = np.nan
    return(X_miss)

def load_X_opt(n_samples, n_features, n_centers, missing_ratio, dataset):
    random_state = 2276282
    path = "data/datasets/real_world_x_opts/"+dataset+"_x_s"+str(n_samples)+"_f"+str(n_features)+"_c"+str(n_centers)+"_m-r"+str(int(missing_ratio*100))+"_m-nb"+str(int(n_samples*n_features*missing_ratio))+"_rs"+str(random_state)+".csv"
    return(pd.read_csv(path, header=None))

def rmse(x,y):
    return(np.sqrt(mean_squared_error(x, y)))

def sil(x, nc):
    kmeans = KMeans(n_clusters=nc)
    kmeans.fit(x)
    return(silhouette_score(x, kmeans.labels_))

def nmi(x, y, nc):
    kmeans = KMeans(n_clusters=nc)
    kmeans.fit(x)
    return(normalized_mutual_info_score(y, kmeans.labels_))

def rw_analysis(ground_truth, y, n_centers, dataset):
    print(dataset+'\n')
    n_samples, n_features = len(ground_truth), len(ground_truth[0])
    if dataset=='reviews':
        for missing_ratio in missing_ratios:
            print(f'MISSING RATIO = {missing_ratio}')
            print(f'GT RMSE = {rmse(ground_truth,ground_truth)}')
            print(f'GT SIL = {sil(ground_truth, n_centers)}')
            X_miss = generate_data(ground_truth, missing_ratio)
            X_mean = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(X_miss)
            print(f'Mean RMSE = {rmse(ground_truth,X_mean)}')
            print(f'Mean SIL = {sil(X_mean, n_centers)}')
            X_med = SimpleImputer(missing_values=np.nan, strategy='median').fit_transform(X_miss)
            print(f'Median RMSE = {rmse(ground_truth,X_med)}')
            print(f'Median SIL = {sil(X_med, n_centers)}')
            X_mf = SimpleImputer(missing_values=np.nan, strategy='most_frequent').fit_transform(X_miss)
            print(f'Most Frequent RMSE = {rmse(ground_truth,X_mf)}')
            print(f'Most Frequent SIL = {sil(X_mf, n_centers)}')
            X_knn = KNNImputer(weights="uniform").fit_transform(X_miss)
            print(f'KNN RMSE = {rmse(ground_truth,X_knn)}')
            print(f'KNN SIL = {sil(X_knn, n_centers)}')
            X_opt = load_X_opt(n_samples, n_features, n_centers, missing_ratio, dataset)
            print(f'Opt RMSE = {rmse(ground_truth,X_opt)}')
            print(f'Opt SIL = {sil(X_opt, n_centers)}')
    else:
        for missing_ratio in missing_ratios:
            print(f'MISSING RATIO = {missing_ratio}')
            print(f'GT RMSE = {rmse(ground_truth,ground_truth)}')
            print(f'GT SIL = {sil(ground_truth, n_centers)}')
            print(f'GT NMI = {nmi(ground_truth, y, n_centers)}')
            X_miss = generate_data(ground_truth, missing_ratio)
            X_mean = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(X_miss)
            print(f'Mean RMSE = {rmse(ground_truth,X_mean)}')
            print(f'Mean SIL = {sil(X_mean, n_centers)}')
            print(f'Mean NMI = {nmi(X_mean, y, n_centers)}')
            X_med = SimpleImputer(missing_values=np.nan, strategy='median').fit_transform(X_miss)
            print(f'Median RMSE = {rmse(ground_truth,X_med)}')
            print(f'Median SIL = {sil(X_med, n_centers)}')
            print(f'Median NMI = {nmi(X_med, y, n_centers)}')
            X_mf = SimpleImputer(missing_values=np.nan, strategy='most_frequent').fit_transform(X_miss)
            print(f'Most Frequent RMSE = {rmse(ground_truth,X_mf)}')
            print(f'Most Frequent SIL = {sil(X_mf, n_centers)}')
            print(f'Most Frequent NMI = {nmi(X_mf, y, n_centers)}')
            X_knn = KNNImputer(weights="uniform").fit_transform(X_miss)
            print(f'KNN RMSE = {rmse(ground_truth,X_knn)}')
            print(f'KNN SIL = {sil(X_knn, n_centers)}')
            print(f'KNN NMI = {nmi(X_knn, y, n_centers)}')
            X_opt = load_X_opt(n_samples, n_features, n_centers, missing_ratio, dataset)
            print(f'Opt RMSE = {rmse(ground_truth,X_opt)}')
            print(f'Opt SIL = {sil(X_opt, n_centers)}')
            print(f'Opt NMI = {nmi(X_opt, y, n_centers)}\n')

### IRIS ANALYSIS

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
print(len(ground_truth))
print(len(ground_truth[0]))
pca = PCA(n_components=2)
principal_components = pca.fit_transform(ground_truth)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
fig = px.scatter(pca_df, x="PC1", y="PC2", color=y, template='none')
fig.update_layout(
    title="PCA plot of the Iris dataset",
    legend_title_text='True labels'
)
pio.write_image(fig, 'figures/iris_pca.pdf')

n_centers = 3
rw_analysis(ground_truth, y, n_centers, "iris")

### SEEDS ANALYSIS

data_file = "data/datasets/raw/seeds_dataset.txt"
seeds = pd.read_csv(data_file, sep='\t', header=None)
ground_truth = seeds.iloc[:, :7].values
y = seeds.iloc[:, -1].values
for i,s in enumerate(y):
    if s==1:
        y[i]=0
    elif s==2:
        y[i]=1
    else:
        y[i]=2

y = [str(x) for x in y]
print(len(ground_truth))
print(len(ground_truth[0]))
pca = PCA(n_components=2)
principal_components = pca.fit_transform(ground_truth)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
fig = px.scatter(pca_df, x="PC1", y="PC2", color=y, template='none')
fig.update_layout(
    title="PCA plot of the Seeds dataset",
    legend_title_text='True labels'
)
pio.write_image(fig, 'figures/seeds_pca.pdf')

n_centers = 3
rw_analysis(ground_truth, y, n_centers, "seeds")