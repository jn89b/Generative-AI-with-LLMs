# Course Introduction

- LLM
- General Purpose technology that can be utilized for various purposes/applications

# Week 1 Introduction 

## Introduction
- Refer to paper "Attention is all you need"
- Attention is powerful because you do it parallel
- Provide a framework on how to build generative AI product life cycle
- Smaller models can be very useful for specific applications or tasks:
  - Think about OODA
    - Observe: What is the current situation? What is the reason you want to change? How bad do you want to change?
    - Orient: Where are you currenly at relative to where you want to go? How far is your destination?
    - Decide: What is the exact path you are going to take? How are you going to handle challenges and set backs?
    - Act: What's the approach and method you will take to implement the decisions? What is your action plan?

## Generative AI and LLMS

- Generative AI:
  - Subset of Machine Learning
  - LLMS have been trained on trillion of words with large compute power, demonstrate emergent behaviors, these are known as base models
  - Some base models are :
    - GPT
    - BERT
    - BLOOM
    - LLaMa
    - FLAN-T5
    - PaLM
- Utilizing LLMS are totally different to how you deal with traditional deep learning and programming paradims
- The process is as follows:
  - Prompt: Text you pass to LLM
  - Context: Space or window of memory
  - Predict: The model predicts the next words, and if you have a question
  - Completion: The output of model, act of using the model to generat text is known as inference

## LLM use cases and tasks 

- Use external apis to make decision in real time/live applications 

## Text generation before Transfromers

- RNN Recurrent Neural Networks, can't deal with context/ does so poorly
- Transformers introduce the concept of attention
  - Give weight of attention with other words
  - Known as **self-attention**
- How it works:
  - Encoder: Inputs("Prompts") with contextual understanding and produces one vector per input token
  - Decoder: Accepts input tokens and generates new tokens
  - Before your feed it as inputs you must first **tokenize** the word
    - Convert words to numbers, where each word correlates to a number utilizing a hash
    - Can use Token IDS to:
      - Match complete words
      - Represent parts of words
    - Must have tokenizer consistent from input to output
  - Embedding:
    - Map into vector based on position:
    - Token
  - Self-attention layers, will update weights to figure out attention
    - This is done parallel, thus known as multi-headed attention
    - This allows each head to have different attention parameters, thus each head can emphasize on properties sucha s rhyme,relation, etc to other words.
  - Feed-forward network
    - Get a vector which consists of probability scores
  - Softmax layer
    - Normalize values

## Generating text with transformers
- Encoder Only Models
  - Used for classification tasks
  - BERT 
- Encoder Decoder models
  - Input and output can be different length
  - BART, T5
- Decoder only models
  - GPT
  - BLOOM
  - LLama
  - Generative models

# Prompt Engineering

- **Prompt Engineering** - Have model improve on specific tasks by formulating better questions
- **In-Context Learning/Zero Shot Inference** - Can do it without example
- **One Shot Inference** - Provide, example with prompt can improve performance
- **Few Shot Inference** - Provide, multiple examples with prompt can improve performance
- Remember Context Window - Limited to few thousand models
- If its not performing well, check out finetuning

# Generative Configuration

- Top-k: Choose the top-k words, which can provide some randomness but not to the point where replies are not too off-topic
- Top-p: Cumulative propoerty <= p_desired 
- Temperature: Parameter influences the shape of probability distrubution, the higher the temperature the higher the randomness, the lower the temperature the lower the randomness
  - Tighter peakers with lower temperature
  - Spread out with higher temperature

# Generative AI Life Cycle

- Scope
  - Define the use case
  - What do I want the model to do exactly??
- Select
    - Choose an existing model or pretrain your own
- Adapt and Align Model - Assess Performance
  - Parallel:
    - Prompt Engineering
      - Initially try in-context learning
      - Utilize zero/one/few-shot learning to evaluate how well the model works
      - If this doesn't work check out fine-tuning
    - Fine-tuning
      - Learn more about this in week 2
    - Align with human feedback
      - Reinforcement learning with human feedback
    - This requires you to evaluate the performance model
  - Evaluate
- Application Integration
  - Optimize and deploy model for inference
    - Ensure best compute performances for model
  - Augment model and build LLM-powered applications
    - Consider additional infrastructure
    - Limitations:
      - Hallucination
      - Complex reasoning with mathematics
    - There are ways to overcome this later on in the weeks

