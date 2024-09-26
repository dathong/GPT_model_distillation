## Introduction: 
Please make sure you have installed OpenAI APIs and relevant packages (NumPy, Pandas, etc. ) before running the code. The project sample configure file is provided.

## Quick Start: 
For convenience, there are some .py files provided with the same names as the tables and figures in the paper. To reproduce the paper results, please just run the files with the corresponding names. For instance, to reproduce the results for Table 1 with GPT-3 model, Strategy-Imitation method, and No-demanding prompt, run "table1.py" with the following parameters:

```
python table1_eval.py model=gpt3 method=bl prompt=nodemanding
```

The output should look like something:

>Ready to evaluate conversations renerated by gpt3, bl, demanding:
>
>Evaluating conversation  0 : +Rating: Excellent.
>
>+Explanation: The agent was very professional, patient, and understanding throughout the conversation. They clearly explained the company's policy on non-refundable and non-changeable tickets and offered ...
>
>Evaluating conversation  31 : +Rating: Excellent.
>
>+Explanation: The agent was very professional and empathetic in handling the customer's request. They clearly explained the company's policy and offered to check for any exceptions. Even though the customer's >request could not be fulfilled, the agent managed to maintain a positive interaction throughout the conversation.
>
>---------summary--------
>
>total_score =  157
>
>scores =  [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 5, 5, 5, 5, 5, 5, 5]
>
>ratio of 5:  0.96875
>avg =  4.90625

### Here is the parameter description:

TABLE1
| Parameter | Possible Values | Description
| --- | --- | --- |
| model | gpt3, llama2_7b, llama2_13b, llama2_70b | Model languages used
| method | bl, si, ft, gg | Base Language Model (bl), Strategy Imitation (si), Fine-tuning (ft), Global Guidelines (gg) |
| prompt | demanding, nodemanding | Use sample conversations between GPT4 agent vs demanding or non-demanding customer |



TABLE3
| Parameter | Possible Values | Description
| --- | --- | --- |
| model | gpt3, llama2_7b, llama2_13b, llama2_70b | Model languages used

Sample command:
```
python table3_eval.py model=gpt3
```
Sample output:
>mean 1 =  0.17427864963495032
>
>mean 2 =  0.16965571404850113

For Table 3, you can re-generate the embeddings of the conversation contexts + responses by re-running sample_convs_dist_all.py for each model in "regenerate" folder. However, this requires providing OpenAI API key in the environment variables OPEN_AI_KEY. For example, after obtaining the key, you can re-generate the embeddings and the checkpoints of GPT-3 results by running regenerate -> gpt3 -> sample_convs_dist_all.py.

TABLE4
| Parameter | Possible Values | Description
| --- | --- | --- |
| model | gpt3, llama2_7b, llama2_13b, llama2_70b | Model languages used
| prompt | demanding, nodemanding | Use sample conversations between GPT4 agent vs demanding or non-demanding customer 

Sample command:
```
python table4_eval.py model=gpt3 prompt=demanding 
```

Sample output:
>---------summary--------
>total_score =  140
>scores =  [5, 5, 4, 5, 5, 3, 5, 5, 5, 5, 5, 3, 4, 5, 5, 5, 5, 5, 1, 5, 5, 5, 5, 3, 5, 3, 3, 5, 3, 5, 3, 5]
>ratio of 5:  0.6875
>avg =  4.375

TABLE5
| Parameter | Possible Values | Description
| --- | --- | --- |
| model | gpt3, llama2_13b, llama2_70b | Model languages used
| prompt | demanding, nodemanding | Use sample conversations between GPT4 agent vs demanding or non-demanding customer 

Sample command:
```
python table5_eval.py model=gpt3 prompt=demanding 
```

Sample output:
>---------summary--------
>total_score =  160
>scores =  [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
>ratio of 5:  1.0
>avg =  5.0

TABLE6
| Parameter | Possible Values | Description
| --- | --- | --- |
| model | gpt3, llama2_13b, llama2_70b | Model languages used
| method | bl, si, ft, gg | Base Language Model (bl), Strategy Imitation (si), Fine-tuning (ft), Global Guidelines (gg) |
| prompt | demanding, nodemanding | Use sample conversations between GPT4 agent vs demanding or non-demanding customer 

Sample command:
```
python table6_eval.py model=gpt3 method=si prompt=demanding 
```

Sample output:
>---------summary--------
>total_score =  38
>scores =  [5, 5, 5, 5, 5, 3, 5, 5]
>ratio of 5:  0.875
>avg =  4.75

# Re-run and re-evaluation

## Prepare the environment
The above scripts displays the results the authors ran. If you want to train or evaluate your own model, you can follow the instructions below.

This paper uses a variety of cloud service APIs to call LLMs such as OpenAI, Deepinfra, HuggingFace, Replicate,... so you need to register a (billed) account with those cloud services.
For example, to re evaluate Table 1 with API call, you need to have OPENAI_API_KEY environment variable; to reevaluate Table 2 with API call, you need DEEPINFRA_KEY (we use Deepinfra to call Llama 3). More instructions to obtain Replicate API can be found here https://replicate.com/docs/get-started/python, and DeepInfra API is here https://deepinfra.com/docs/getting-started.  The instructions on how to get the API keys are from these platform website. For example, to set up OpenAI API key, please follow this link https://platform.openai.com/docs/quickstart. Since we use API to query model outputs, we need to specify the keys for different models. To use ChatGPT, we need to specify the OpenAI key. To use LlaMa models, we used the Replicate and DeepInfra platform. Below is an example of adding the keys  to the environment variables: open a terminal and type the following:

```
export REPLICATE_API_TOKEN = your_replicate_api_token
export OPEN_AI_Key = your_open_ai_key
export DEEPINFRA_KEY = your_deep_infra_key
```
You also need to install the package replicate. For Mac users, try 
```
pip install replicate
```
After adding the API keys as the environment variable,  you can start running the "main_script.py" to start running the algorithm. After finishing, the final conversation list will be generated in the "stage{k}" folder, where k is the number of total stages.



## Re-evaluate with API calls:

 After adding the key as the environment variable, you can run the same evaluation code like Part 1, but add one more parameter `use_api=True` and `eval_model=[model_to_evaluate]`. The "eval_model" can be any models available on OpenAI and Deep Infra platforms, for example: `gpt-3.5-turbo-0125`, `gpt-4`, `meta-llama/Meta-Llama-3.1-70B-Instruct`,... For example, to reproduce the Table 1 result for GPT3 with an API call using `gpt-3.5-turbo-0125` model, run:

```
python table1_eval.py model=gpt3 method=bl prompt=nodemanding use_api=True eval_model=gpt-3.5-turbo-0125
```

You should get similar results to Table 1, model GPT3, base model, and no-demanding prompt. Please note: due to the probabilistic nature of LLM, sometimes you will get a slightly different number from the paper (even with temperature=0). However, the difference should be small and does not change the claims made in the paper.


Different environments might have different configurations. LLMs require some complex environment setup before they can be used. If you have any issues running the code or have any questions, please contact me Dat Hong at hongtiendat@gmail.com

## Run the algorithms to regenerate the conversations:

In case you want to re-generate the conversations using the algorithm from scratch, we provide the code in the "algorithm" folder. To run the algorithm, in addition to OpenAI key, you need the Replicate API key. 



For example, to run the code with gpt3 model with 3 stages, run:

```
python main_script.py student_model=gpt-3.5-turbo-0125 teacher_model=gpt-4o-mini max_stage=3
```

The code needs some time to run. 
You will see the following output files in the stage{i} folder.

| File name | Description | 
| --- | --- |
| conv_guidelines.pkl | Contains the intermediate guidelines and sample responses for each iteration | 
| response_mapping.pkl | Temporary file: indicate which models are used for each responses in the conversations | 
| conv_df.pkl | Temporary file: the student responses and teacher response, given a conversational context | 
| teacher_responses.pkl | Temporary file: the teacher responses to the conversations| 
| sub_conv_embeds.pkl | Temporary file: embedding vectors of the conversations | 
| full_conversations.csv | Result file: the output conversations | 

If you would like to see intermediate results on other stages, you can check the corresponding folders, such as "stage1".

Please note that the code will consume some amount of money because it will call remote API. You can change the student and teacher model in the parameters `student_model` and 'teacher_model`.

Here are the possible parameters:

| Parameter | Possible Values | Description
| --- | --- | --- |
| student_model | gpt-3.5-turbo-0125, meta/llama-2-7b-chat, meta/llama-2-13b-chat, meta/llama-2-70b-chat | Model languages used
| teacher_model | gpt-3.5-turbo-0125, gpt-4, gpt-4o-mini | Model languages used that provided by OpenAI
| max_stage | Integer, default 3, recommend less than 10 | How many stages you want the algorithm to run
| mode | test, train | If mode is "test", we only generate 2 conversations for testing. If the mode is "train", we can generate 32 conversations


The output conversation will be generated in folder "stage3/full_conversations.csv".

## Evaluate the new conversation with API calls:

After generating the new conversation, you can evaluate this conversation but running:

```
python eval_conv.py conv_file=[path to the file] prompt=nodemanding eval_model=gpt-3.5-turbo-0125
```

Here are the possible parameters:

| Parameter | Possible Values | Description
| --- | --- | --- |
| conv_file | for example default: algorithm\imitation_learning\stage3\full_conversations.csv | The path to your file
| eval_model | gpt-3.5-turbo-0125, gpt-4, gpt-4o-mini, meta-llama/Meta-Llama-3.1-70B-Instruct,... | Model languages used that provided by OpenAI and DeepInfra
| max_stage | Integer, default 3, recommend less than 10 | How many stages you want the algorithm to run
| prompt | demanding, nodemanding | Use sample conversations between GPT4 agent vs demanding or non-demanding customer 


Different environments might have different configurations. LLMs require some complex environment setup before they can be used. If you have any issues running the code or have any questions, please contact me Dat Hong at hongtiendat@gmail.com
