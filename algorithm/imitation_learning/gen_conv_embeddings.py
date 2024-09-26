import argparse
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
   'stage': '0',
}

# Update the parameters dictionary with default values if they are not provided
for key, value in default_params.items():
   if key not in params:
      params[key] = value

   # Print the parameters dictionary
print(params)

stage_no = params['stage']
# stage_no = "3"

curr_knowledge = pickle.load(open(os.path.join("stage" + str(int(stage_no)-1),"conv_guidelines.pkl"),"rb"))
convs = curr_knowledge['sub_conv']

client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

async def get_embedding(client, text, model="text-embedding-ada-002"):
   #print('genererating the embeddings of subconv...')
   text = text.replace("\n", " ")
   response = await client.embeddings.create(input = [text], model=model)
   #print("done generating embeddings")
   return response.data[0].embedding


async def main():
   print("[db] generating embeddings...")
   op_file = os.path.join("stage" + str(int(stage_no)-1), "sub_conv_embeds.pkl")
   tasks = []
   for t in convs:
      task = asyncio.create_task(get_embedding(client,t))
      tasks.append(task)
   embed_list  = await asyncio.gather(*tasks)

   print("[db] done generating embeddings...")
   pickle.dump(embed_list, open(op_file, "wb"))

if __name__ == "__main__":
   asyncio.run(main())

