import argparse
import pickle
import os
import numpy as np


class KeyValueAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        for value in values:
            key, val = value.split('=', 1)
            setattr(namespace, key, val)

parser = argparse.ArgumentParser(description='Process some parameters in the format param=value.')
parser.add_argument('params', nargs='*', action=KeyValueAction, help='Parameters in the format param=value')

# Parse the command-line arguments
args = parser.parse_args()

# Convert the parsed args to a dictionary
params = vars(args)

# Define default values
default_params = {
    'use_api': 'False',
    'model': 'gpt3',
}

# Update the parameters dictionary with default values if they are not provided
for key, value in default_params.items():
    if key not in params:
        params[key] = value

# Print the parameters dictionary
print(params)

file = "distances_" + params['model']

distances1 = pickle.load(open(os.path.join("checkpoints","table3",file + ".pkl"),"rb"))
distances2 = pickle.load(open(os.path.join("checkpoints","table3",file + "_ablated.pkl"),"rb"))

dists1, dists2 = np.array(distances1), np.array(distances2)
# # print("all mean 1 = ",np.mean(dists))
# # print("all mean 2 = ",np.mean(dists,axis=0))
# # print("all mean 3 = ",np.mean(dists,axis=1))
# data1 = dists2.flatten().tolist()
data1 = dists2[:,0].tolist()
# data = np.mean(dist2,axis=1)
data2 = dists1[:,0].tolist()
# data2 = dists1.flatten().tolist()
res = []

# for i,num in enumerate(data2):
#     percentile_rank = stats.percentileofscore(dists2[i,:].tolist(), num )
#     res.append(percentile_rank)
#
# print("res = ",sum(res)/len(res))
print("mean 1 = ",sum(data1)/len(data1))
print("mean 2 = ",sum(data2)/len(data2))