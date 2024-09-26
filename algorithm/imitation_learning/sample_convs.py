import argparse
import itertools
import pickle
import sys
import traceback

import replicate

import my_utils
import os
import pandas as pd
from openai import OpenAI,AsyncOpenAI
from my_utils import *
import asyncio
import time
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
        'stage': '0',
        'student_model': params['student_model'],
        'teacher_model': params['teacher_model'],
        'mode' : 'real',
}

    # Update the parameters dictionary with default values if they are not provided
for key, value in default_params.items():
    if key not in params:
        params[key] = value

    # Print the parameters dictionary
print(params)


stage_no = params['stage']
# stage_no = '0'
response_mapping = pickle.load(open(os.path.join("stage" + stage_no,"response_mapping.pkl"),"rb"))

gpt_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
embedding_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

guidelines, sub_conv_embeds = [],[]
if int(stage_no) > 0:
    for i in range(1,int(stage_no)+1):
        conv_guidelines = pickle.load(open(os.path.join("stage" + str(int(i)-1),"conv_guidelines.pkl"),"rb"))
        guidelines.extend(conv_guidelines['guidelines'])
        sub_conv_embeds.extend(pickle.load(open(os.path.join("stage" + str(int(i)-1),"sub_conv_embeds.pkl"),"rb")))

# model_stg_prob_mapping = {0:1,1:0.75,2:0.5,3:0.25,4:0}

student_agent = params['student_model']
# student_agent = "gpt-3.5-turbo-0125"
teacher_agent = params['teacher_model']
# teacher_agent = "gpt-4"
teacher_client = gpt_client
student_client = gpt_client

# sample 100 sub_convs but need to extract full sub_convs from each conv
def sample_conv(conv_list):
    conv_res, response_res = [], []
    conv_res.append("\n\n".join(conv_list[:1]))
    response_res.append(conv_list[1])
    # [user: hi]
    # [agent: how ccan i help you?]
    # [user: i have problem...]
    # GPT3 or llama2: [agent: i can help that.]
    if len(conv_list) >= 3:
        conv_res.append("\n\n".join(conv_list[:3]))
        response_res.append(conv_list[3])
    if len(conv_list) >= 5:
        conv_res.append("\n\n".join(conv_list[:5]))
        response_res.append(conv_list[5])
    if len(conv_list) >= 7:
        conv_res.append("\n\n".join(conv_list[:7]))
        response_res.append(conv_list[7])
#    if len(conv_list) >= 9:
#        res.append("\n\n".join(conv_list[:9]))
    return conv_res, response_res

def convert_conv_to_list(conv):
    res = []
    str_list = conv.split("\n")
    curr_str = ""
    for s in str_list:
        # if s.strip() == "":
        #     curr_str += "\n"
        if s.startswith('CUSTOMER:') or s.startswith('AGENT:'):
            res.append(curr_str)
            curr_str = ""
        curr_str += s + "\n"
    res.append(curr_str)
    return res[1:]
    
async def gen_new_response(conv, sys_msg_agt, conv_id):
    sys_txt = sys_msg_agt

    msg_list = [{"role": 'system', "content": sys_txt}]
    total_token_no = num_tokens_from_string(sys_txt)
    curr_agent_id = -1
    for _j, msg in enumerate(conv):
        role = 'assistant' if _j % 2 else 'user'
        msg_token_no = num_tokens_from_string(msg)
        new_msg = {"role": role, "content": msg}
        msg_list.append(new_msg)
        total_token_no += msg_token_no
        if not _j % 2:
            curr_agent_id += 1
    res_max_token = 3800 - total_token_no
    retry = 0
    curr_model = response_mapping[(conv_id,curr_agent_id)]
    model_name = student_agent
    client = student_client
    if curr_model == student_agent:
        model_name = teacher_agent
        client = teacher_client

    # if int(stage_no) > 0 and model_name == student_agent:
    #     sys_txt += "In addition, please follow these DO List and the DONT List when interacting with the customer. The 'DO List' outlines what you must follow, and the 'DONT List' details what you should avoid. Ensure you provide a complete answer to the customer. Always try to say something to the customer. Do not leave the answer blank. Do not just ask the customer to wait and then stop the conversation. Note that you are a customer support agent now, so only talk like an agent.\n\n "
    #     conv_txt = "\n".join(conv)
    #     conv_embed = await my_utils.get_embedding_async(conv_txt, embedding_client)
    #     closest_idx = my_utils.find_k_closest_embedding(conv_embed, sub_conv_embeds)
    #     # print("db closest idx, guidelibe len, sub_conv_embeds len: ", closest_idx, len(guidelines),
    #     #      len(sub_conv_embeds))
    #     best_guideline = guidelines[closest_idx]
    #     # list of 128 best training guidelines & conversations embedding
    #     best_guideline = guidelines[closest_idx]
    #     sys_txt += best_guideline

    if int(stage_no) > 0 and model_name == student_agent:
        sys_txt += my_prompts.AGENT_GUIDELINE_PROMPT1
        last_txt = msg_list[-1]['content']
        last_txt_org = last_txt
        conv_txt = "\n".join(conv)
        # print("db embedding_client = ",embedding_client)
        conv_embed = await my_utils.get_embedding_async(conv_txt, embedding_client)
        # print("db conv_embed = ",conv_embed)
        closest_idx, distance = my_utils.find_k_closest_embedding(conv_embed, sub_conv_embeds)
        # print("db closest idx, guidelibe len, sub_conv_embeds len: ", closest_idx, len(guidelines),
        #      len(sub_conv_embeds))
        best_guideline = guidelines[closest_idx]
        # agent_sys_txt += GUIDELINE_FOLLOW_PROMPT
        # agent_sys_txt += best_guideline
        last_txt += "\n\n[" + my_prompts.AGENT_GUIDELINE_PROMPT2.strip()
        last_txt += best_guideline + "]"
        msg_list[-1]['content'] = last_txt
    sys_txt += my_prompts.AIRLINE_POLICY_TICKET
    msg_list[0]['content'] = sys_txt
    while retry < 4:
        try:
            print("[db] generating response...")
        # print("max token allowed = ",res_max_token)
            if 'gpt' in model_name.lower():
                response = await client.chat.completions.create(
                    # model="gpt-4",
                    model=model_name,
                    # model="meta-llama/Llama-2-7b-chat-hf",
                    messages=msg_list,
                    temperature=0.3,
                    max_tokens=300,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                response2 = response.choices[0].message.content
            elif 'llama' in model_name.lower():
                prompt = my_utils.convert_msg_to_prompt(msg_list)
                output = replicate.run(
                    student_agent,
                    input={"prompt": prompt,
                           "temperature": 0.3,
                           "max_new_tokens": 400,
                           "stop_sequences": "</s>"}
                )
                response2 = ''.join(output)
        #     response1 = "[dummy answers]"
            if len(response2.strip()) < 3:
                print("[db] agent gave empty response, retrying...")
                retry += 1
                continue
            return response2, model_name
        except Exception as e:
            print("Exception occured: ", e)
            print(traceback.format_exc())
            retry += 1
            print("retrying...", retry)
            time.sleep(2)
            continue
    return "[no response]", model_name

async def main():

    # stage_no = "3"

    op_file = os.path.join("stage{_stage_no}".format(_stage_no=stage_no),"teacher_responses.pkl")
    #full_conversation_list = pickle.load(open(os.path.join("stage{_stage_no}".format(_stage_no=stage_no),"all_data_llama2.pkl"),"rb"))
    full_conversations_df = pd.read_csv(os.path.join("stage" + stage_no,"full_conversations.csv"),encoding='utf-8')
    subconv_list, curr_responses, curr_conv_id = [], [], []

    for _i,conv in enumerate(full_conversations_df['conv'].values.tolist()):
        conv_list = convert_conv_to_list(conv)
        conv_sample, responses = sample_conv(conv_list)
        curr_conv_id.extend([_i for n in range(len(conv_sample))])
        subconv_list.extend(conv_sample)
        curr_responses.extend(responses)



    conv_list1 = []
    for c in subconv_list:
        conv_sl = my_utils.convert_conv_to_list(c)
        conv_list1.append(conv_sl)

    sys_txt = my_prompts.AGENT_PROMPT_TICKET
    tasks = []
    for _i,conv in enumerate(conv_list1):
        task = asyncio.create_task(gen_new_response(conv, sys_txt, curr_conv_id[_i]))
        tasks.append(task)
    next_vals = await asyncio.gather(*tasks)
    student_responses, teacher_responses = [],[]
    for i,val in enumerate(next_vals):
        response, model_name = val[0], val[1]
        if model_name == student_agent:
            student_responses.append(response)
            teacher_responses.append(curr_responses[i])
        else:
            teacher_responses.append(response)
            student_responses.append(curr_responses[i])

    pickle.dump(teacher_responses,open(op_file,"wb"))
        
    df = pd.DataFrame()
    df['sub_conv'] = subconv_list
    df['student_agent'] = student_responses
    df['teacher_agent'] = teacher_responses
    df.to_csv(os.path.join("stage{_stage_no}".format(_stage_no=stage_no),"conv_df.csv"),header=True,index=False)

    print("Done")


if __name__ == "__main__":
    asyncio.run(main())

