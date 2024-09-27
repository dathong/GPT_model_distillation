import itertools
import pickle
import sys
import traceback

import my_utils
import os
import pandas as pd
from openai import OpenAI,AsyncOpenAI
from my_utils import *
import asyncio
import time
import my_prompts

stage_no = "3"
response_mapping = pickle.load(open(os.path.join("stage" + stage_no,"response_mapping.pkl"),"rb"))

gpt_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
embedding_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])


guidelines, sub_conv_embeds = [],[]
if int(stage_no) > 0:
    for i in range(int(stage_no)):
        conv_guidelines = pickle.load(open(os.path.join("stage" + str(int(i)),"conv_guidelines.pkl"),"rb"))
        guidelines.extend(conv_guidelines['guidelines'])
        sub_conv_embeds.extend(pickle.load(open(os.path.join("stage" + str(int(i)),"sub_conv_embeds.pkl"),"rb")))


# model_stg_prob_mapping = {0:1,1:0.75,2:0.5,3:0.25,4:0}

student_agent = "meta-llama/Llama-2-7b-chat-hf"
# teacher_agent = "gpt-3.5-turbo-0125"
teacher_agent = "gpt-4"
teacher_client = gpt_client
student_client = llama2_client

distances = []
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
    if len(conv_list) >= 9:
        conv_res.append("\n\n".join(conv_list[:9]))
        response_res.append(conv_list[9])

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
    print("gen new response...")
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

    if int(stage_no) > 0:
        sys_txt += my_prompts.AGENT_GUIDELINE_PROMPT1
        last_txt = msg_list[-1]['content']
        last_txt_org = last_txt
        conv_txt = "\n".join(conv)
        # print("db embedding_client = ",embedding_client)
        conv_embed = await my_utils.get_embedding_async(conv_txt, embedding_client)
        # print("db conv_embed = ",conv_embed)
        closest_idx, dists = my_utils.find_k_closest_embedding_all(conv_embed, sub_conv_embeds)
        # print("db closest idx, guidelibe len, sub_conv_embeds len: ", closest_idx, len(guidelines),
        #      len(sub_conv_embeds))
        distances.append(dists)
        best_guideline = guidelines[closest_idx]
        # agent_sys_txt += GUIDELINE_FOLLOW_PROMPT
        # agent_sys_txt += best_guideline
        last_txt += "\n\n[" + my_prompts.AGENT_GUIDELINE_PROMPT2.strip()
        last_txt += best_guideline + "]"
        msg_list[-1]['content'] = last_txt

    print("done generating new response...")

async def main():

    # stage_no = "3"

    op_file = os.path.join("stage{_stage_no}".format(_stage_no=stage_no),"teacher_responses.pkl")
    #full_conversation_list = pickle.load(open(os.path.join("stage{_stage_no}".format(_stage_no=stage_no),"all_data_llama2.pkl"),"rb"))
    full_conversations_df = pd.read_csv(os.path.join("full_conv_gpt3_trained_gpt3.csv"),encoding='utf-8')
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

    # print("avg distances = ",sum(distances)/len(distances))
    pickle.dump(distances, open("distances_gpt3_ablated.pkl", "wb"))

    print("Done")


if __name__ == "__main__":
    asyncio.run(main())

