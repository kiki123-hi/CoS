from openai import OpenAI
import httpx
from icecream import ic as print  
import re
from tqdm import tqdm
import json
import argparse
import math
import os
import time  


def read_jsonl(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def write_jsonl(path, data):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

# def load_inputs(file_path):
#     with open(file_path, 'r') as f:
#         return json.load(f)
def load_inputs(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            print(f"JSON anlysis failed： {e.lineno},  {e.colno}")
            print(f"Erro：{e.doc}")
            raise

def extract_events(input_text):
    events = []
    event_pattern = r'Event Id: (\d+), start time: (.*), end time: (.*), utility score: ([0-9.]+)'
    matches = re.findall(event_pattern, input_text)
    for match in matches:
        events.append({
            'id': match[0],
            'start_time': match[1],
            'end_time': match[2],
            'utility': float(match[3])
        })
    return events


api_key = ""
deepseek_api_key = ""
qwen_api_key = ""

# deepseeek api
client = OpenAI(
    api_key=deepseek_api_key, 
    base_url="https://api.deepseek.com"
)
# client = OpenAI(
#     api_key=qwen_api_key, 
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
# )


parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, required=True)
parser.add_argument("--block_num", type=int, required=True)
parser.add_argument("--block_index", type=int, required=True)
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--folder_name", type=str, required=True)
parser.add_argument("--per_proc_max_generate", type=int, required=True)
args = parser.parse_args()

folder_name = args.folder_name
data = load_inputs(args.input_path)

block_len = math.ceil(len(data) / args.block_num)
prefix_len = args.block_index * block_len

if args.block_index != (args.block_num - 1):
    data = data[args.block_index * block_len:(args.block_index + 1) * block_len]
else:
    data = data[args.block_index * block_len:]

save_file_path = os.path.join(folder_name, args.save_path)
try:
    pre_len = len(read_jsonl(save_file_path))
except:
    pre_len = 0

total_generation_time = 0
processed_count = 0

for idx, example in tqdm(enumerate(data)):
    if idx < pre_len:
        continue
    if idx >= args.per_proc_max_generate:
        break
    
    events = extract_events(example['question'])
    
    events_str = "\n".join([
        f"Event Id: {e['id']}, start time: {e['start_time']}, end time: {e['end_time']}, utility score: {e['utility']}"
        for e in events
    ])
    
    prompt = """Please generate an optimal event schedule by following these steps:
    1. Understand the Problem:
    - Objective: Maximize the total utility score of selected events.
    - Constraints: 
        No two events can overlap in time.
        All events must use their exact given time slots.
    2. Analyze Event Information:
    - Here are the available events:
        {events_str}
    - For each event, note:
        Event Id: Unique identifier
        Time Window: [Start Time → End Time]
        Utility Score: Value gained if selected
    3. Solution Steps
    Step 1: Generate All Possible Conflict-Free Sequences
    Use a backtracking algorithm to enumerate all possible non-overlapping event combinations.
    For each potential combination:
    a. Verify that all event pairs satisfy:
    EventA.end <EventB.start OR EventB.end> EventA.start
    b. Retain all combinations that pass this check.
    Step 2: Calculate Total Utility for Each Sequence
    For each Conflict-Free sequence, compute the total utility.
    Step 3: Select the Optimal Sequence
    Choose the sequence with the highest total utility from all Conflict-Free sequences.
    4.Output the Solution:
    Must provide your final answer in this exact format:
    Selected Events: [Event Ids in selection order]
    """.format(events_str=events_str)
#     prompt = """Please generate an event schedule based on the information provided below, with the objective of maximizing the sum of utility values of all events in the generated schedule.
# Key requirements:
# 1. Non-Overlapping: Events cannot occur at the same time. Ensure that each event ends before the next one begins.
# 2. Time Constraints: Use the exact start and end times provided for each event.

# Event information:
# {events_str}

# The event information includes:
# - Event Id
# - Start Time
# - End Time
# - Utility score

# Constraints:
# - Ensure that all selected events do not overlap in time.
# - Confirm that the start and end times strictly match the provided event information.

# Please provide the solution in the format:
# Selected Events: [Event Ids]
# """.format(events_str=events_str)

    messages = [
        {"role": "system", "content": "You are a problem solver."},
        {"role": "user", "content": prompt}
    ]
     # deepseek API
    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=messages,
            stream=False,
            temperature=0,
            top_p=1,
            n=1
        )

    # try:
    #     start_time = time.time()
        
    #     response = client.chat.completions.create(
    #         model="qwen-plus",
    #         messages=messages,
    #         stream=False,
    #         temperature=0,
    #         top_p=1,
    #         n=1,
    #     )

        end_time = time.time()
        generation_time = end_time - start_time
        total_generation_time += generation_time
        processed_count += 1
        
        content = response.choices[0].message.content
        example['schedule'] = content
        example['generation_time'] = generation_time

        write_jsonl(save_file_path, example)

    except Exception as e:
        print(e)
        error_idx = prefix_len + idx


if processed_count > 0:
    print(f"\n\n=== Time statistics ===")
    print(f" {total_generation_time:.2f}s")
    print(f" {processed_count}")
else:
    print("None.")