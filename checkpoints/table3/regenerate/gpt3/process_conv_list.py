import pickle
import pandas as pd
import openai
import copy
import json
import time
import itertools
from openai import OpenAI,AsyncOpenAI
import os
import asyncio
import csv
import sys
import my_utils
import traceback
import random
import my_prompts

def process_conv(conv):
    res = ""
    conv_parts = conv.split("[CUSTOMER LEAVING THE CHAT]")
    res += conv_parts[0] + "[CUSTOMER LEAVING THE CHAT]\n\n"
    res += "AGENT: [AGENT LEAVING THE CHAT]\n"
    return res

df = pd.read_csv(os.path.join("stage0","full_conversations_gpt4_gpt3_2.csv"),encoding='utf-8')
conv_list = df['conv'].values.tolist()
conv_list1 = [process_conv(conv) for conv in conv_list]

df['conv'] = conv_list1
df.to_csv("full_convs_gpt4_gpt3_2.csv",header=True,index=None)

fc_res = []
cols = df.columns.values.tolist()
for index, row in df.iterrows():
    d = {k:row[k] for k in cols}
    fc_res.append(d)
pickle.dump(fc_res,open("full_convs_gpt4_gpt3_2.pkl","wb"))
print("Done")