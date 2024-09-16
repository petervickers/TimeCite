import os
import random
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

CITE_RATE = 0.3


# Function to find threshold using dev set
def clf_threshold_finder(dev_df):
    dev_df_valid_cos_sim = dev_df[dev_df['sim'] != -1]
    x, y = np.array(dev_df_valid_cos_sim['sim']).reshape(-1, 1), \
           np.array([1 if y_i == 1 else 0 for y_i in dev_df_valid_cos_sim['cite']])
    clf = LogisticRegression()
    clf.fit(x, y)
    implied_threshold = -clf.intercept_ / clf.coef_.item()
    return implied_threshold.item()


# Function to read and process TimeCite similarities
def process_similarity_data(file_path):
    # Read the data
    df = pd.read_csv(file_path, sep='\t')
    return df


# Read TimeCite data
TimeCite = pd.read_csv('TimeCite.tsv', sep='\t')

# Get all TSV files in the data directory
data_files = [f for f in os.listdir('preds') if f.endswith('.tsv')]

# Dictionary to store accuracy per model and bin
model_accuracies = {}

# Iterate over each file in the data directory
for file in data_files:
    model_name = file.split('.')[0]
    file_path = os.path.join('preds', file)
    
    # Process similarity data
    sims_df = process_similarity_data(file_path)
    
    # Build Evaluation dfs
    timecite_sims_test_df = TimeCite.loc[TimeCite['is_test'] == 1].merge(sims_df, on=['id0', 'id1'], how='left')
    timecite_sims_dev_df = TimeCite.loc[TimeCite['is_test'] == 0].merge(sims_df, on=['id0', 'id1'], how='left')

    bin_wise_accuracies = {}  # Dictionary to store accuracy per bin for current model

    # Iterate over bins
    for bin_name, bin_df_test in timecite_sims_test_df.groupby('max_bin'):  # Use 'max_bin' for grouping
        bin_df_val = timecite_sims_dev_df[timecite_sims_dev_df['max_bin'] == bin_name]

        # Find threshold using validation set
        threshold = clf_threshold_finder(bin_df_val)

        # Predict citations for test set
        bin_df_test['predicted_citation'] = (bin_df_test['sim'] >= threshold)

        # For rows where 'sim' == -1, set 30% of 'predicted_citation' to 1 for both cite == 1 and cite == 0
        missing_values_df = bin_df_test[bin_df_test['sim'] == -1]

        missing_values_cited_indices = missing_values_df[missing_values_df['cite'] == 1].index
        missing_values_not_cited_indices = missing_values_df[missing_values_df['cite'] == 0].index

        num_cited_to_set = int(CITE_RATE * len(missing_values_cited_indices))
        num_not_cited_to_set = int(CITE_RATE * len(missing_values_not_cited_indices))

        cited_indices_to_set = random.sample(list(missing_values_cited_indices), num_cited_to_set)
        not_cited_indices_to_set = random.sample(list(missing_values_not_cited_indices), num_not_cited_to_set)

        bin_df_test.loc[cited_indices_to_set, 'predicted_citation'] = True
        bin_df_test.loc[not_cited_indices_to_set, 'predicted_citation'] = True

        bin_df_test.loc[missing_values_df.index.difference(cited_indices_to_set).difference(not_cited_indices_to_set), 'predicted_citation'] = False

        # Calculate accuracy
        accuracy = (bin_df_test['predicted_citation'] == bin_df_test['cite']).mean()
        bin_wise_accuracies[bin_name] = accuracy

    model_accuracies[model_name] = bin_wise_accuracies

# Convert the model accuracies dictionary to a DataFrame
results = pd.DataFrame.from_dict(model_accuracies, orient='index').transpose()

# Add the bin names as a column
results.insert(0, 'bin', results.index)

# Save the results to 'results.tsv'
results.to_csv('results.tsv', sep='\t', index=False)

print("Results saved to 'results.tsv'")
