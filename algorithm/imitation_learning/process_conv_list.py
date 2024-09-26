import pickle
import pandas as pd
import os


def process_conv(conv):
    res = ""
    conv_parts = conv.split("[CUSTOMER LEAVING THE CHAT]")
    res += conv_parts[0] + "[CUSTOMER LEAVING THE CHAT]\n\n"
    res += "AGENT: [AGENT LEAVING THE CHAT]\n"
    return res

df = pd.read_csv(os.path.join("stage3","full_conversations.csv"),encoding='utf-8')
conv_list = df['conv'].values.tolist()
conv_list1 = [process_conv(conv) for conv in conv_list]

df['conv'] = conv_list1
df.to_csv("full_conv_l2_7b_trained.csv",header=True,index=None)

fc_res = []
cols = df.columns.values.tolist()
for index, row in df.iterrows():
    d = {k:row[k] for k in cols}
    fc_res.append(d)
pickle.dump(fc_res,open("full_conv_l2_7b_trained.pkl","wb"))
print("Done")