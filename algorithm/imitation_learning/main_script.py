import argparse
import os


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
        'max_stage': '3',
        # 'model': 'gpt-3.5-turbo-0125',
        'student_model': 'meta/llama-2-7b-chat',
        'teacher_model': 'gpt-3.5-turbo-0125',
        'mode' : 'test'
}

    # Update the parameters dictionary with default values if they are not provided
for key, value in default_params.items():
    if key not in params:
        params[key] = value

    # Print the parameters dictionary
print(params)


# for stage in range(1,4):
for stage in range(int(params['max_stage']) + 1):
    print("STAGE: ",stage,"...")
    if not os.path.exists("stage" + str(stage)):
        os.makedirs("stage" + str(stage))
    if stage > 0:
        os.system("python gen_conv_embeddings.py stage=" + str(stage))
    os.system("python conv_generation.py stage=" + str(stage)
              + " student_model=" + params['student_model']
              + " teacher_model=" + params['teacher_model']
              + " mode=" + params['mode'])
    os.system("python sample_convs.py stage=" + str(stage)
              + " student_model=" + params['student_model']
              + " teacher_model=" + params['teacher_model'])
    os.system("python update_guidelines_and_responses.py stage=" + str(stage) + " model=" + params['model'])