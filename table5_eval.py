import asyncio
import pickle
import argparse
import os
import pandas as pd
import openai
from openai import AsyncOpenAI
import my_prompts

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
    'method': 'tf',
    'prompt': 'nodemanding',
    # 'eval_model': 'gpt-3.5-turbo-0125'
    # 'eval_model': 'meta-llama/Meta-Llama-3.1-8B-Instruct'
}




# Update the parameters dictionary with default values if they are not provided
for key, value in default_params.items():
    if key not in params:
        params[key] = value

# Print the parameters dictionary
print(params)

# file_to_eval = "l3_70b_eval_" + params['model'] + "_" + params['notrained'] + "_"
file_to_eval = "gpt4_eval_{model}_{prompt}.pkl".format(
    model=params['model'],
    prompt=params['prompt']
)



async def eval_conv(_curr_conv, gpt_client, customer_agent):
    curr_conv = _curr_conv

    sample_conv1 = my_prompts.GPT4_EXCELLENT_CONV_DEMANDING1
    sample_conv2 = my_prompts.GPT4_EXCELLENT_CONV_DEMANDING2

    if 'nodemanding' in params['prompt']:
        sample_conv1 = my_prompts.GPT4_EXCELLENT_CONV_NODEMANDING1
        sample_conv2 = my_prompts.GPT4_EXCELLENT_CONV_NODEMANDING2

    sys_txt = my_prompts.GPT_EVAL_PROMPTS_2SAMPLE.format(
        excellent_conv1='"""' + sample_conv1 + '"""',
        excellent_conv2='"""' + sample_conv2 + '"""',
    )
    msg_list = [{'role':'system', 'content':sys_txt}]
    msg_list.append({'role':'user','content':curr_conv})
    # print("called API")
    response = await gpt_client.chat.completions.create(
        # model="ft:gpt-3.5-turbo-0613:yale-university::87p0vJIx",
        model=customer_agent,
        # model="gpt-4",
        messages=msg_list,
        temperature=0,
        max_tokens=200,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0)
    response1 = response.choices[0].message.content
    return response1

async def api_main(eval_file,gpt_client,customer_agent):

    model_to_eval = eval_file
    # full_conv_df1 = pd.read_csv(os.path.join("{model}.csv").
    #                             format(model=model_to_eval),encoding='utf-8')

    conv_list1 = eval_file['conv'].values.tolist()


    tasks = []
    for i in range(len(conv_list1)):
        # if i != 19:
        #     continue
        task = asyncio.create_task(eval_conv(conv_list1[i],gpt_client,customer_agent))
        tasks.append(task)
    results = await asyncio.gather(*tasks)
    total_score = 0
    rating_res = []
    for res in results:
        print("Evaluating conversation ", i, ":", res)
        rating = res.split("\n")[0].lower()
        if "rating: excellent" in rating:
            total_score += 5
            rating_res.append(5)
        elif "rating: very good" in rating:
            total_score += 4
            rating_res.append(4)
        elif "rating: good" in rating:
            total_score += 3
            rating_res.append(3)
        elif "rating: acceptable" in rating:
            total_score += 2
            rating_res.append(2)
        elif "rating: unacceptable" in rating:
            total_score += 1
            rating_res.append(1)
    print("---------summary--------")
    print("total_score = ",total_score,total_score/len(conv_list1))
    print("scores = ", rating_res)
    print("ratio of 5: ", rating_res.count(5) / len(rating_res))
    print("avg = ", sum(rating_res) / len(rating_res))

    print("Done evaluating conv..")

def main():

    file_to_eval = "gpt4_eval_{model}_{prompt}.pkl".format(
        model = params['model'],
        prompt= params['prompt']
    )
    # print("file_to_eval = ",file_to_eval)
    results = pickle.load(open(os.path.join("checkpoints","table5",file_to_eval),'rb'))

    total_score = 0
    rating_res = []
    if params['use_api'] == 'True':
        openai.api_key = os.environ["OPENAI_API_KEY"]
        gpt_client = AsyncOpenAI()
        deepinfra_client = AsyncOpenAI(
            api_key=os.environ["DEEPINFRA_KEY"],
            base_url="https://api.deepinfra.com/v1/openai")
        # customer_agent = "gpt-3.5-turbo-0125"
        # customer_agent = "gpt-4"
        customer_agent = params['eval_model']
        client = gpt_client
        if 'llama' in customer_agent:
            client = deepinfra_client
        file_to_eval = "full_conv_{model}_tf.csv".format(
            model=params['model'],
            method=params['method'],
            prompt=params['prompt']
        )
        conv_file = pd.read_csv(os.path.join("conversations","{model}").
                                    format(model=file_to_eval),encoding='utf-8')
        # conv_file = pickle.load(open(os.path.join("conversations", file_to_eval), 'rb'))
        # api_main(conv_file,gpt_client,customer_agent)
        asyncio.run(api_main(conv_file,client,customer_agent))
        return

    print("Ready to evaluate conversations renerated by {}, {}, {}:".
            format(params['model'],params['method'],params['prompt']))
    for i,res in enumerate(results):
        print("Evaluating conversation ",i,":",res)
        rating = res.split("\n")[0].lower()
        if "rating: excellent" in rating:
            total_score += 5
            rating_res.append(5)
        elif "rating: very good" in rating:
            total_score += 4
            rating_res.append(4)
        elif "rating: good" in rating:
            total_score += 3
            rating_res.append(3)
        elif "rating: acceptable" in rating:
            total_score += 2
            rating_res.append(2)
        elif "rating: unacceptable" in rating:
            total_score += 1
            rating_res.append(1)
    print("---------summary--------")
    print("total_score = ", total_score)
    print("scores = ", rating_res)
    print("ratio of 5: ", rating_res.count(5) / len(rating_res))
    print("avg = ", sum(rating_res) / len(rating_res))


if __name__ == '__main__':
    main()
