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
import os
import replicate
import argparse

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
        'model': 'meta/llama-2-7b-chat',
        'mode' : 'real',
        'train': 'True'
}

    # Update the parameters dictionary with default values if they are not provided
for key, value in default_params.items():
    if key not in params:
        params[key] = value

    # Print the parameters dictionary
print(params)


mode = params['mode']


n_size = 32

if mode == 'test':
    n_size = 2
# stage_no = "3"



# Please make sure you have set OPENAI_API_KEY or DEEP_INFRA_KEY environment variable
gpt_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
embedding_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


model_stg_prob_mapping = {0:0,1:0,2:0,3:0,4:0}
if params['train'] != 'True':
    model_stg_prob_mapping = {0:0,1:0,2:0,3:0,4:0}
response_mapping = {}
student_agent = params['model']

# teacher_agent = "gpt-4"
# teacher_agent = "gpt-3.5-turbo-0125"
customer_agent = "gpt-3.5-turbo-0125"
customer_client = gpt_client
teacher_client = gpt_client
student_client = gpt_client

def convert_msg(d1, sys_msg, to="agent", usr_first_msg="Hello, how may I help you?"):
    d2 = {"messages": [sys_msg]}
    if to == 'agent':
        d2['messages'].append({'role': 'user', 'content': 'Hello'})
        d2['messages'].append({"role": "assistant", "content": 'Hello! How may I help you?'})

        for i, msg in enumerate(d1['messages'][2:]):
            role = 'assistant' if i % 2 else 'user'
            msg = {"role": role, "content": msg['content']}
            d2['messages'].append(msg)

    if to == 'user':
        d2['messages'].append({'role':'user','content':usr_first_msg})
        for i, msg in enumerate(d1['messages'][3:]):
            role = 'user' if i % 2 else 'assistant'
            msg = {"role": role, "content": msg['content']}
            d2['messages'].append(msg)
    return d2



customer_styles = [
"analytical",
"amiable",
"expressive",
"driver",
]

customer_emotions = [
"concerned because you realized you have booked a restricted ticket, so you are worried whether you can successfully cancel the ticket",
"frustrated because you have tried cancelling the ticket yourself on the app and website but failed to do that, then you realized the ticket is non-changeable and non-refundable",
"calm",
"confused about the cancellation policy and you don't know how to cancel the ticket",
]

problems = [
"flight cancellation without penalty",
]

problems_explain = {
"flight cancellation without penalty": """You booked a non-refundable and nonchangeable flight ticket 1 day ago. Now you call the agent to request to cancel it without penalty and requested a full refund.""",
"seat upgrade": """You purchased a economy ticket and now you call to upgrade to business class.""",
"requesting compensation for lost luggage": """You lost your baggage at the airport and you call the airline company to ask for possible $500 compensation.""",
}

customer_styles_explain = {
"""analytical""": """You prefer to have more profound knowledge about the subject before getting convinced on a particular matter. You verify each piece of information and focus more on the brand’s features to ensure its quality and efficiency. You use most of your logical thinking rather than your emotional side when making decisions.""",
"""amiable""": """You are respectful, friendly, and trustworthy. You’re good at listening to and forming relationships with others. Unlike analytical thinkers, ami- able people care more about building rapport and establishing trust with other pro- fessionals. You’re more interested in conducting business transactions with people who meet their buying expectations. Your decision relies on how the company val- ues their interest in relationship-building.""",
"""expressive""": """You are willing to voice your opinions, share your feelings and emotions and talk about your personal situations. You’d prefer to share your perspective when presented with facts rather than ask for additional information. You are a fast decision maker.""",
"""driver""": """You are primarily self-centered and opinionated. You find pleasure in manipulating a pitch that identifies you as reasonable and authoritative. You expect information to be delivered in the quickest way possible because you’re goal-oriented. You’re commanding in nature and motivated to achieve your objectives. You want immediate answers and solutions. You also value competence as much as you love expertise and preparation. Drivers are fast decision makers.""",
}

is_demanding = ["demanding", ""]


comb_list = [customer_styles, customer_emotions, problems, is_demanding]
combs = list(itertools.product(*comb_list))

# n = 200
# for i in range(2):


comb_sample = []
for i,c in enumerate(combs):
    list_c = list(c)
    # list_c.append(solve_or_not[0])
    comb_sample.append(list_c)


pickle.dump(comb_sample, open("comb_sample", "wb"))

async def gen_conv(cust_comb,_i):
    print("db generating conv...")
    print("comb = ",cust_comb)
    c_p, c_e, problem, is_demanding = cust_comb[0], cust_comb[1], cust_comb[2], cust_comb[3]


    conv_txt = ""
    conv = {}
    retry = 0
    while retry < 4:
        try:
            agent_sys_txt = my_prompts.AGENT_PROMPT_TICKET
            cust_sys_txt = my_prompts.CUSTOMER_PROMPT
            cust_sys_txt = cust_sys_txt.format(
                problem=problems_explain[problem],
                customer_personality=c_p,
                customer_personality_explain=customer_styles_explain[c_p],
                customer_emotion=c_e,
                difficulty=is_demanding
            )

            # print("agent_sys_txt = ",agent_sys_txt)
            # print("cust_sys_txt = ", cust_sys_txt)

            agent_sys_msg = {"role": "system", "content": agent_sys_txt}
            cust_sys_msg = {"role": "system", "content": cust_sys_txt}
            cust_first_msg = {"role": "user", "content": "Hello, how may I help you?"}

            d_cust = {"messages": [cust_sys_msg, cust_first_msg]}
            repeat_time = 8
            conv = {}
            conv_txt = ""
            for _j in range(repeat_time):
                response = await gpt_client.chat.completions.create(
                    # model="ft:gpt-3.5-turbo-0613:yale-university::87p0vJIx",
                    model=customer_agent,
                    # model="gpt-4",
                    messages=d_cust['messages'],
                    temperature=0.3,
                    max_tokens=200,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0)
                response1 = response.choices[0].message.content
                # print("user response = ", response)
                conv_txt += "CUSTOMER: " + response1 + "\n\n"
                agent_sys_txt = my_prompts.AGENT_PROMPT_TICKET

                usr_msg = {"role": "assistant", "content": response1}
                d_cust['messages'].append(usr_msg)
                # time.sleep(1.01)
                agent_sys_msg['content'] = agent_sys_txt
                d_agent = convert_msg(d_cust, agent_sys_msg, to="agent")
                last_txt = d_agent['messages'][-1]['content']
                last_txt_org = last_txt

                model_name = student_agent
                client = student_client


                response_mapping[(_i, _j)] = model_name
                agent_sys_txt += my_prompts.AIRLINE_POLICY_TICKET
                d_agent['messages'][0]['content'] = agent_sys_txt

                if 'gpt' in model_name.lower():
                    response = await client.chat.completions.create(
                        # model="gpt-4",
                        model=model_name,
                        # model="meta-llama/Llama-2-7b-chat-hf",
                        messages=d_agent['messages'],
                        temperature=0.3,
                        max_tokens=300,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                    )
                    response2 = response.choices[0].message.content
                elif 'llama' in model_name.lower():
                    prompt = my_utils.convert_msg_to_prompt(d_agent['messages'])
                    output = replicate.run(
                        student_agent,
                        input = {"prompt": prompt,
                                 "temperature": 0.3,
                                 "max_new_tokens": 300,
                                 "stop_sequences": "</s>"}
                    )
                    response1 = ''.join(output)
                    response2 = response1.split("[INST]")[0]
                    response2 = response2.replace("AGENT:","")
                    response2 = response2.strip()
                    # if '[INST]' in response1:
                    #     print("found problem")
                conv_txt += "AGENT: " + response2 + "\n\n"
                # print("agent response = ", response)
                d_agent['messages'][-1]['content'] = last_txt_org
                d_agent['messages'].append({"role": "assistant", "content": response2})
                d_cust = convert_msg(d_agent, cust_sys_msg, to="user")
                if 'goodbye' in response1.lower() or 'CUSTOMER LEAVING THE CHAT' \
                        in response1 or 'have a great day' in response1.lower() \
                        or len(response2.strip()) < 3:
                    break

            if len(response2.strip()) < 3 and 'CUSTOMER LEAVING THE CHAT' not in response1:
                print("[db] agent gave empty response, retrying...")
                retry += 1
                continue

            conv['agent_prompt'] = agent_sys_txt
            conv['customer_prompt'] = cust_sys_txt
            conv['personality'] = c_p
            conv['emotion'] = c_e
            conv['difficulty'] = is_demanding
            conv['conv'] = conv_txt
            break
        except Exception as e:
            print("Exception occured: ",e)
            print(traceback.format_exc())
            retry += 1
            print("retrying...", retry)
            continue

    print("Finished generating convs")
    return conv, conv_txt


async def my_main():
    tasks = []
    for _i, comb in enumerate(comb_sample[:n_size]):
        task = asyncio.create_task(gen_conv(comb,_i))
        tasks.append(task)
    results = await asyncio.gather(*tasks)
    # sys.exit()

    conv_res = [e[0] for e in results]


    fields = ["agent_prompt", "customer_prompt", "personality", "emotion", "difficulty", "conv"]
    with open(os.path.join("full_conversations_notrain.csv"), 'w', encoding="utf-8", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for conv in conv_res:
            writer.writerow({
                "agent_prompt": conv['agent_prompt'],
                "customer_prompt": conv['customer_prompt'],
                "personality": conv['personality'],
                "emotion": conv['emotion'],
                "difficulty": conv['difficulty'],
                "conv": conv['conv'],
            })

    print("Done generating conv..")

if __name__ == "__main__":
    asyncio.run(my_main())