# %%
%load_ext autoreload
%autoreload 2

# %% load feature_interactions_ckubmeg1.npy
import numpy as np
import pandas as pd
feature_interactions = np.load('feature_interactions_ckubmeg1.npy')
print(feature_interactions.shape)

baseline_interactions = np.load('activation_baseline_similarity.npy')
print(baseline_interactions.shape)

# %%

from sklearn.cluster import AffinityPropagation, DBSCAN, OPTICS, HDBSCAN, KMeans

# load the explanations file
explanations_file = '/workspace/crosscoders-feature-interactions/sleepers/sleepers/autointerp/autointerp_data/explanations_ckubmeg1.csv'
explanations = pd.read_csv(explanations_file)
print(explanations.shape)
# get list of all feature ids
feature_ids = explanations['feature_id'].unique()
# change to list of ints
feature_ids = [int(id) for id in feature_ids]
print(len(feature_ids))

# Filter the interaction matrices to only include features that have explanations
feature_interactions = feature_interactions[feature_ids][:, feature_ids]
baseline_interactions = baseline_interactions[feature_ids][:, feature_ids]

# Now perform clustering on the filtered matrices
similarity_matrix = (feature_interactions + feature_interactions.T)/2.0
np.fill_diagonal(similarity_matrix, 1.0)
model = AffinityPropagation()
labels = model.fit_predict(similarity_matrix)

# get the labels for the baseline interactions
baseline_labels = model.fit_predict(baseline_interactions)


# %%
# map each cluster label to a list of explanation ids
cluster_active_fts = {}
cluster_explanations = {}
for i, label in enumerate(labels):
    if label not in cluster_active_fts:
        cluster_active_fts[label] = []
    cluster_active_fts[label].append(feature_ids[i])
    if label not in cluster_explanations:
        cluster_explanations[label] = []
    cluster_explanations[label].append(explanations.loc[explanations['feature_id'] == feature_ids[i], 'explanation'].values[0])

print('IM')
print('Number of clusters:', len(cluster_explanations))
print('Average number of explanations per cluster:', round(np.mean([len(v) for v in cluster_explanations.values()]), 2))
print('Variation in number of explanations per cluster:', round(np.max([len(v) for v in cluster_explanations.values()]) - np.min([len(v) for v in cluster_explanations.values()]), 2))

baseline_cluster_active_fts = {}
baseline_cluster_explanations = {}
for i, label in enumerate(baseline_labels):
    if label not in baseline_cluster_active_fts:
        baseline_cluster_active_fts[label] = []
    baseline_cluster_active_fts[label].append(feature_ids[i])
    if label not in baseline_cluster_explanations:
        baseline_cluster_explanations[label] = []
    baseline_cluster_explanations[label].append(explanations.loc[explanations['feature_id'] == feature_ids[i], 'explanation'].values[0])

print('Baseline')
print('Number of clusters:', len(baseline_cluster_explanations))
print('Average number of explanations per cluster:', round(np.mean([len(v) for v in baseline_cluster_explanations.values()]), 2))

print('Variation in number of explanations per cluster:', round(np.max([len(v) for v in baseline_cluster_explanations.values()]) - np.min([len(v) for v in baseline_cluster_explanations.values()]), 2))


# %%
# print some examples of the explanations
for label in range(3):
    print(cluster_explanations[label][0])
    print(cluster_explanations[label][1])
    print(cluster_explanations[label][2])
    break


# %%
import matplotlib.pyplot as plt
# plot distribution of cluster sizes
plt.hist([len(v) for v in cluster_explanations.values()], bins=100)
plt.show()

# plot distribution of baseline cluster sizes
plt.hist([len(v) for v in baseline_cluster_explanations.values()], bins=100)
plt.show()

# %%
import json
# for all clusters with size < 25 dump {cluster_label: [explanation1, explanation2, ...]} to a file containing all clusters
cluster_explanations_small = {int(k): v for k, v in cluster_explanations.items() if len(v) < 25 and len(v) > 5}
with open('cluster_explanations.json', 'w') as f:
    json.dump(cluster_explanations_small, f)

baseline_cluster_explanations_small = {int(k): v for k, v in baseline_cluster_explanations.items() if len(v) < 25 and len(v) > 5}
with open('baseline_cluster_explanations.json', 'w') as f:
    json.dump(baseline_cluster_explanations_small, f)

# %%

from sleepers.autointerp.util.llm_autointerp import azureAutointerp

autointerp = azureAutointerp()

# %%
import random
# load cluster_explanations.json
with open('cluster_explanations.json', 'r') as f:
    im_cluster_explanations = json.load(f)

# load baseline_cluster_explanations.json
with open('baseline_cluster_explanations.json', 'r') as f:
    baseline_cluster_explanations = json.load(f)

correct_count = 0
total_count = 0

for version in ['Baseline', 'IM']:
    cluster_explanations = baseline_cluster_explanations if version == 'Baseline' else im_cluster_explanations
    all_cluster_labels = list(cluster_explanations.keys())
    for cluster_label, explanations in list(cluster_explanations.items()):
        for i in range(5):
            sampled_explanations = random.sample(explanations, 6)
            examples = sampled_explanations[:5]
            test_explanation = sampled_explanations[5]
            # select random explanations from different clusters
            random_cluster_labels = random.sample(all_cluster_labels, 4)
            # check none of the random cluster labels are the same as the cluster label
            while cluster_label in random_cluster_labels:
                random_cluster_labels = random.sample(all_cluster_labels, 4)

            random_explanations = [cluster_explanations[label][random.randint(0, len(cluster_explanations[label])-1)] for label in random_cluster_labels]
            insert_position = random.randint(0, 4)
            test_explanations = random_explanations[:insert_position] + [test_explanation] + random_explanations[insert_position:]
            prompt = autointerp.format_interaction_evaluator_prompt(examples, test_explanations)
            # api call
            response = autointerp.generate_autointerp(prompt)
            #print(insert_position)
            #print(response)
            try:
                if insert_position == int(response):
                    correct_count += 1
                total_count += 1
            except:
               # print(response)
                pass

    print(f'{version} Accuracy: {correct_count/total_count}')

# %%



