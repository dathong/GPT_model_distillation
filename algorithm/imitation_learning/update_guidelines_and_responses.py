import argparse
import pickle
import traceback

import pandas as pd
import replicate
from openai import OpenAI,AsyncOpenAI
import tiktoken
import asyncio
import timeit
import os
import re
import sys
import my_prompts
import my_utils

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
        'stage': '2',
        'model': 'meta/llama-2-7b-chat',
        'mode' : 'real',
}

    # Update the parameters dictionary with default values if they are not provided
for key, value in default_params.items():
    if key not in params:
        params[key] = value

    # Print the parameters dictionary
print(params)

gpt_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
gpt_syn_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

client = gpt_client
TA_model = "gpt-3.5-turbo-0125"
student_model = params['model']

async def prompt_learning(df,max_iter = 5):
    # previous_guidelines,conv_list,student_responses,teacher_responses
    start = timeit.default_timer()
    conv_list = df['sub_conv'].values
    student_responses = df['student_agent'].values
    teacher_responses = df['teacher_agent'].values
    tasks = []
    for i,conv in enumerate(conv_list):
        task = asyncio.create_task(learn_instance_prompt(conv,student_responses[i],teacher_responses[i],max_iter))
        tasks.append(task)
    results = await asyncio.gather(*tasks) # multiple do lists and dont lists
    guidelines = [e[0] for e in results]
    scores = [e[1] for e in results]

    end = timeit.default_timer()
    total_time = end - start
    print(f'Total time {total_time:.4f}s')
    conv_guidelines = {'sub_conv':df['sub_conv'].values.tolist(), 'guidelines': guidelines, "scores:": scores}
    pickle.dump(conv_guidelines,open(os.path.join("stage{_stage_no}".format(_stage_no=stage_no),"conv_guidelines.pkl"),"wb"))


def num_tokens_from_string(string, encoding_name = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

async def learn_instance_prompt(conv,student, teacher, max_iter = 5):
    print("db learning instance prompt...")
    prev_guidline = ''
    best_guideline = ''
    best_score = -1
    iter_no = 0
    count = 0
    score = 0
    student_response = student
    eval_scores = []
    while True:
        print("iter_no = ",iter_no,"...")
        new_guideline = await get_textual_updates(prev_guidline, conv, student_response, teacher)
        new_response = await generate_responses(conv, new_guideline)
        new_score = await TA_evaluator(conv, new_response,teacher)
        eval_scores.append(new_score)
        student_response = new_response
        if new_score > best_score :
            best_guideline = new_guideline
            best_score = new_score
            if new_score == 5:
                print("db finished learning instance prompt...")
                return (best_guideline, eval_scores)
        if new_score < score:
            print("db finished learning instance prompt...")
            return (best_guideline, eval_scores)
        iter_no += 1
        if iter_no >= max_iter:
            break
        score = new_score
        prev_guidline = new_guideline
    print("db finished learning instance prompt...")
    return (best_guideline, eval_scores)

def extract_first_number(text):
    # Using regular expression to find the first number in the string
    match = re.search(r'\d+', text)
    if match:
        return int(match.group())
    else:
        return None

async def TA_evaluator(conv, student, teacher):
    usr_txt = """
    Here is the on-going conversation between a customer and a customer support agent in an airline company (in triple quotes):

    \"\"\"
    {conv}
    \"\"\"

    Here is the next response from a good customer support agent (also in triple quotes).  Let's call him GOOD AGENT.

    \"\"\"
    {good_agent}
    \"\"\"

    Now act as the customer, you are provided with a response from a new agent for the same conversation. Let's call him NEW AGENT. Your job is to evaluate how similar the NEW AGENT's response to the GOOD AGENT's response in terms of customer's satisfaction. Rank the response a score from scale one to five. One means very different from GOOD AGENT's response and five means very similar to the GOOD AGENT's response. Only show the score in your output. Refrain from providing any additional output. If you cannot tell, output 'Hard to tell'.

    NEW AGENT's response:
    \"\"\"
    {new_agent}
    \"\"\"

    """
    usr_txt = usr_txt.format(conv= conv,
                                new_agent=student,
                                good_agent=teacher
                                )
    msg_list = [{"role": 'system', "content": "You are a customer for an airline company."}]
    usr_msg = {"role": 'user', "content": usr_txt}
    msg_list.append(usr_msg)
    total_token_no = num_tokens_from_string(usr_txt)
    res_max_token = 4000 - total_token_no
    try:
        response = await gpt_client.chat.completions.create(
                    # model="gpt-4",
                    model=TA_model,
                    messages=msg_list,
                    temperature=0,
                    max_tokens=res_max_token,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
        )
        gpt_evaluation = response.choices[0].message.content
        score = extract_first_number(gpt_evaluation)
    except Exception as e:
        print("Exception occur: ",e)
        return "None"
    return score


async def get_textual_updates(prev_guideline_i, conv, student_response_i,  teacher_response_i ):
    SYS_TXT_MNG= my_prompts.MANAGER_PROMPT
    msg_list = [{"role": 'system', "content": SYS_TXT_MNG}]
    total_token_no = num_tokens_from_string(SYS_TXT_MNG)
    user_msg = conv + '\n\n'
    user_msg += '+AGENT (Good):\n' + teacher_response_i + '\n\n'
    user_msg += '+AGENT (Bad):\n' + student_response_i + '\n\"\"\"\n\n'    # print("user_msg = ",user_msg)
    new_msg = {"role": 'user', "content": user_msg}
    msg_list.append(new_msg)
    conv_token_no = num_tokens_from_string(user_msg)
    total_token_no += conv_token_no
    res_max_token = 4000 - total_token_no
    try:
        response = await gpt_client.chat.completions.create(
                    # model="ft:gpt-3.5-turbo-0613:yale-university::87p0vJIx",
                    model=TA_model,
                    # model="gpt-4",
                    messages=msg_list,
                    temperature=0.3,
                    max_tokens=res_max_token,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
        )
        update = response.choices[0].message.content
        new_guideline = prev_guideline_i + '\n\n\n\n'+update+'\n'#consolidate_guidelines(previous_guidelines[i],response2)
        new_guideline = await consolidate_all_lists(new_guideline)
    except Exception as e:
        new_guideline = prev_guideline_i
    return new_guideline

async def consolidate_all_lists(guideline):
    sys_txt = my_prompts.MERGE_PROMPT
    msg_list = [{"role": 'system', "content": sys_txt}]
    total_token_no = num_tokens_from_string(sys_txt)
    user_msg = guideline 
    new_msg = {"role": 'user', "content": user_msg}
    msg_list.append(new_msg)
    total_token_no += num_tokens_from_string(user_msg)
    res_max_token = 4000 - total_token_no
    # print("max token allowed = ", res_max_token)
    try:
        response = await gpt_client.chat.completions.create(
            # model="ft:gpt-3.5-turbo-0613:yale-university::87p0vJIx",
            model=TA_model,
            # model="gpt-4",
            messages=msg_list,
            temperature=0.3,
            max_tokens=res_max_token,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0)
        new_response = response.choices[0].message.content
    except Exception as e:
        print("Exception occured: ", e)
        new_response = guideline
    return new_response


######### generate new responses ##############
async def generate_responses(conv, guideline):

    sys_txt = my_prompts.AGENT_PROMPT_TICKET
    if len(guideline) > 2:
        sys_txt += my_prompts.AGENT_GUIDELINE_PROMPT1

    sys_txt += my_prompts.AIRLINE_POLICY_TICKET

    sys_txt = sys_txt[:8000]
    msg_list = [{"role": 'system', "content": sys_txt}]
    total_token_no = num_tokens_from_string(sys_txt)
    for j, msg in enumerate([conv]):
        role = 'assistant' if j % 2 else 'user'
        msg_token_no = num_tokens_from_string(msg)
        new_msg = {"role": role, "content": msg}
        msg_list.append(new_msg)
        total_token_no += msg_token_no

    if len(guideline) > 2:

        last_txt = msg_list[-1]['content']
        last_txt += "[" + my_prompts.AGENT_GUIDELINE_PROMPT2
        last_txt += guideline + "]"
        msg_list[-1]['content'] = last_txt
        total_token_no += num_tokens_from_string(last_txt)

    res_max_token = 3800 - total_token_no
    response1 = ""
    try: # replace with Llama 2 agent later
        # print("max token allowed = ",res_max_token,total_token_no)
        if 'gpt' in student_model.lower():
            response = await client.chat.completions.create(
                # model="gpt-4",
                model=student_model,
                # model="meta-llama/Llama-2-7b-chat-hf",
                messages=msg_list,
                temperature=0.3,
                max_tokens=300,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            response1 = response.choices[0].message.content
        elif 'llama' in student_model.lower():
            prompt = my_utils.convert_msg_to_prompt(msg_list)
            output = replicate.run(
                student_model,
                input={"prompt": prompt,
                       "temperature": 0.3,
                       "max_new_tokens": 300,
                       "stop_sequences": "</s>"}
            )
            response1 = ''.join(output)
#         print('the new response is\n {}'.format(response1))
    except Exception as e:
        print("Exception occured: ",e)
        print(traceback.format_exc())
        response1 = ''
    return response1


if __name__ == "__main__":
    stage_no = params['stage']
    # stage_no = "3"
    df = pd.read_csv(os.path.join("stage{_stage_no}".format(_stage_no=stage_no), "conv_df.csv"))
    if params['mode'] == 'test':
        df = df[:1]
    asyncio.run(prompt_learning(df))