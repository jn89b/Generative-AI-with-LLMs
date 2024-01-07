from datasets import load_dataset
import openai
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import GenerationConfig

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)

huggingface_dataset_name = "knkarthick/dialogsum"
dataset = load_dataset("knkarthick/dialogsum")

example_indices = [40, 200]

dash_line = '-'.join('' for x in range(100))

for i, index in enumerate(example_indices):
    print(dash_line)
    print('Example ', i + 1)
    print(dash_line)
    print('INPUT DIALOGUE:')
    print(dataset['test'][index]['dialogue'])
    print(dash_line)
    print('BASELINE HUMAN SUMMARY:')
    print(dataset['test'][index]['summary'])
    print(dash_line)
    print()
    
#load the model
USE_LLAMA2 = True

model_name='google/flan-t5-base'
# if USE_LLAMA2:
#     model_name = "NousResearch/Llama-2-7b-chat-hf"
#     model = 
# model_name = "meta-llama/Llama-2-7b-chat-hf"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

#load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

#test the tokenizer encoding and decoding
sentence = 'I love to play football'
sentence_encoded = tokenizer(sentence, return_tensors='pt')
sentence_decoded = tokenizer.decode(sentence_encoded['input_ids'][0],
                                    skip_special_tokens=True)
print('ENCODED SENTENCE:', sentence_encoded)
print('DECODED SENTENCE:', sentence_decoded)

# Testing the Model
for i, index in enumerate(example_indices):
    dialogue = dataset['test'][index]['dialogue']
    summary = dataset['test'][index]['summary']
    
    inputs = tokenizer(dialogue, return_tensors='pt')
    output = tokenizer.decode(
        model.generate(
            inputs["input_ids"], 
            max_new_tokens=50,
        )[0], 
        skip_special_tokens=True
    )
    
    print(dash_line)
    print('Example ', i + 1)
    print(dash_line)
    print(f'INPUT PROMPT:\n{dialogue}')
    print(dash_line)
    print(f'BASELINE HUMAN SUMMARY:\n{summary}')
    print(dash_line)
    print(f'MODEL GENERATION - WITHOUT PROMPT ENGINEERING:\n{output}\n')


#%% Prompt Engineering

# Zero shot with Instruction Prompt
for i, index in enumerate(example_indices):
    dialogue = dataset['test'][index]['dialogue']
    summary = dataset['test'][index]['summary']

    prompt = f"""
Summarize the following conversation.
{dialogue}

Summary:
    """

    # Input constructed prompt instead of the dialogue.
    inputs = tokenizer(prompt, return_tensors='pt')
    output = tokenizer.decode(
        model.generate(
            inputs["input_ids"], 
            max_new_tokens=50,
        )[0], 
        skip_special_tokens=True
    )
    
    print(dash_line)
    print('Example ', i + 1)
    print(dash_line)
    print(f'INPUT PROMPT:\n{prompt}')
    print(dash_line)
    print(f'BASELINE HUMAN SUMMARY:\n{summary}')
    print(dash_line)    
    print(f'MODEL GENERATION - ZERO SHOT:\n{output}\n')


#prompt it to do math
some_math = '2 + 2 ='
prompt = f"""
Here is a simple addition problem:
3 + 5 = 8
Now Calculate the following:
{some_math}
"""

inputs = tokenizer(prompt, return_tensors='pt')
output = tokenizer.decode(
    model.generate(
        inputs["input_ids"], 
        max_new_tokens=50,
    )[0], 
    skip_special_tokens=True
)

print(dash_line)
print(f'INPUT PROMPT:\n{prompt}')
print(dash_line)
print(f'MODEL GENERATION - ZERO SHOT:\n{output}\n')
