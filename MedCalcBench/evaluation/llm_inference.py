__author__ = "guangzhi"
'''
Adapted from https://github.com/Teddy-XiongGZ/MedRAG/blob/main/src/medrag.py
'''
import os
import re
import json
import tqdm
import torch
import time
import asyncio
import argparse
import transformers
from transformers import AutoTokenizer
import openai
from transformers import StoppingCriteria, StoppingCriteriaList
import tiktoken
import openai
import sys
from huggingface_hub import login
from mcp_agent.core.fastagent import FastAgent
import medcalc.calculator as mc

login(token=os.getenv("HUGGINGFACE_TOKEN"))

openai.api_key = os.getenv("OPENAI_API_KEY") 


class LLMInference:

    def __init__(self, llm_name="OpenAI/gpt-3.5-turbo", cache_dir="../../huggingface/hub"):
        self.llm_name = llm_name
        self.cache_dir = cache_dir
        if self.llm_name.split('/')[0].lower() == "openai":
            self.model = self.llm_name.split('/')[-1]
            if "gpt-3.5" in self.model or "gpt-35" in self.model:
                self.max_length = 4096
            elif "gpt-4" in self.model:
                self.max_length = 8192
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        else:
            self.type = torch.bfloat16
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name, cache_dir=self.cache_dir, legacy=False)
            if "mixtral" in llm_name.lower() or "mistral" in llm_name.lower():
                self.tokenizer.chat_template = open('../templates/mistral-instruct.jinja').read().replace('    ', '').replace('\n', '')
                self.max_length = 32768
            elif "llama-2" in llm_name.lower():
                self.max_length = 4096
                self.type = torch.float16
            elif "llama-3" in llm_name.lower():
                self.max_length = 8192
            elif "meditron-70b" in llm_name.lower():
                self.tokenizer.chat_template = open('../templates/meditron.jinja').read().replace('    ', '').replace('\n', '')
                self.max_length = 4096
            elif "pmc_llama" in llm_name.lower():
                self.tokenizer.chat_template = open('../templates/pmc_llama.jinja').read().replace('    ', '').replace('\n', '')
                self.max_length = 2048
            self.model = transformers.pipeline(
                "text-generation",
                model=self.llm_name,
                torch_dtype=self.type,
                device_map="auto",
                model_kwargs={"cache_dir":self.cache_dir},
            )

    def answer(self, messages):
        # generate answers

        ans = self.generate(messages)
        ans = re.sub("\s+", " ", ans)
        return ans

    def custom_stop(self, stop_str, input_len=0):
        stopping_criteria = StoppingCriteriaList([CustomStoppingCriteria(stop_str, self.tokenizer, input_len)])
        return stopping_criteria
    

    def get_result(self, instruction, text, model, mcp):
        if mcp:
            fast = FastAgent("fast-agent example")

            @fast.agent(instruction=instruction, model=model, servers=["medcalc"])
            async def agent_function():
                async with fast.run() as agent:
                    result = await agent(text)
                    return result

            loop = asyncio.get_event_loop()
            return loop.run_until_complete(agent_function())

        else:
            fast = FastAgent("fast-agent example")

            @fast.agent(instruction=instruction, model=model, servers=[])
            async def agent_function():
                async with fast.run() as agent:
                    result = await agent(text)
                    return result

            loop = asyncio.get_event_loop()
            return loop.run_until_complete(agent_function())


    def generate(self, messages, prompt=None):
        '''
        generate response given messages
        '''
        if "mcp" in self.llm_name.lower():
            model = None
            mcp = False
            if "haiku" in self.llm_name.lower():
                model = 'haiku'
            if "sonnet" in self.llm_name.lower():
                model = 'sonnet'
                # time.sleep(30)
            if "gpt-4.1" in self.llm_name.lower():
                model = 'gpt-4.1-mini'
            if "o1" in self.llm_name.lower():
                model = 'o1'
            if "o1-mini" in self.llm_name.lower():
                model = 'o1-mini'
            if "o3-mini" in self.llm_name.lower():
                model = 'o3-mini'
            
            if "true" in self.llm_name.lower():
                mcp = True

            if model is None:
                raise ValueError("Model not found")
            instruction = messages[0]["content"]
            text = messages[1]["content"]
        
            ans = self.get_result(instruction, text, model, mcp)

        elif "tools" in self.llm_name.lower():
            if "openai" in self.llm_name.lower():
                client = openai.OpenAI()
                
                # Define your tools/functions
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "egfr_epi",
                            "description": "Estimated Glomerular Filtration Rate (eGFR) using the EPI formula (version 2021)",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "scr": {
                                        "type": "number",
                                        "description": "serum creatinine level in mg/dL"
                                    },
                                    "age": {
                                        "type": "integer",
                                        "description": "Age in years"
                                    },
                                    "male": {
                                        "type": "boolean",
                                        "description": "true if Male"
                                    },
                                },
                                "required": ["scr", "age", "male"]
                            }
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "crcl_cockcroft_gault",
                            "description": "Calculate Creatinine Clearance using the Cockcroft-Gault formula",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "age": {
                                        "type": "integer",
                                        "description": "Age in years"
                                    },
                                    "weight": {
                                        "type": "number",
                                        "description": "Actual body weight in kg"
                                    },
                                    "height": {
                                        "type": "number",
                                        "description": "Height in inches"
                                    },
                                    "scr": {
                                        "type": "number",
                                        "description": "serum creatinine level in mg/dL"
                                    },
                                    "sex": {
                                        "type": "string",
                                        "description": "Gender ('male' or 'female')"
                                    },
                                },
                                "required": ["age", "weight", "height", "scr", "sex"]
                            }
                        }
                    }
                ]
                
                # Initialize conversation with user's messages
                conversation = messages.copy()
                max_iterations = 10
                iterations = 0
                
                while iterations < max_iterations:
                    # Make API request
                    response = client.chat.completions.create(
                        model='gpt-4.1-mini',
                        messages=conversation,
                        tools=tools,
                        tool_choice="auto"
                    )
                    
                    message = response.choices[0].message
                    
                    # Important: Store the COMPLETE message object to preserve tool_calls
                    assistant_message = {
                        "role": "assistant",
                        "content": message.content or ""
                    }
                    
                    # If there are tool calls, add them to the message
                    if message.tool_calls:
                        assistant_message["tool_calls"] = [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments
                                }
                            }
                            for tool_call in message.tool_calls
                        ]
                    
                    # Add the complete assistant message to conversation
                    conversation.append(assistant_message)
                    
                    # If no tool calls, we're done
                    if not message.tool_calls:
                        ans = message.content
                        break
                        
                    # Process tool calls
                    tool_responses = []
                    for tool_call in message.tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)
                        
                        if function_name == "egfr_epi":
                            result = mc.egfr_epi(
                                scr=function_args["scr"], 
                                age=function_args["age"], 
                                male=function_args["male"]
                            )
                        elif function_name == "crcl_cockcroft_gault":
                            result = mc.crcl_cockcroft_gault(
                                age=function_args["age"],
                                weight=function_args["weight"],
                                height=function_args["height"],
                                scr=function_args["scr"],
                                sex=function_args["sex"]
                            )
                        else:
                            result = f"Error: Function {function_name} not implemented"
                            
                        # Create proper tool response
                        tool_responses.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": str(result)
                        })
                    
                    # Add all tool responses to conversation
                    conversation.extend(tool_responses)
                    
                    iterations += 1
                
                # If we hit max iterations, use the last response
                if iterations == max_iterations:
                    last_assistant_messages = [msg for msg in reversed(conversation) if msg.get("role") == "assistant"]
                    ans = last_assistant_messages[0]["content"] if last_assistant_messages else "Max iterations reached without resolution"

            else:
                stopping_criteria = None
                if prompt is None:
                    prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                if "meditron" in self.llm_name.lower():
                    stopping_criteria = self.custom_stop(["###", "User:", "\n\n\n"], input_len=len(self.tokenizer.encode(prompt, add_special_tokens=True)))
                if "llama-3" in self.llm_name.lower():
                    response = self.model(
                        prompt,
                        do_sample=False,
                        eos_token_id=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                        pad_token_id=self.tokenizer.eos_token_id,
                        max_length=min(self.max_length, len(self.tokenizer.encode(prompt, add_special_tokens=True)) + 4096),
                        truncation=True,
                        stopping_criteria=stopping_criteria,
                        temperature=0.0
                    )
                else:
                    response = self.model(
                        prompt,
                        do_sample=False,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.eos_token_id,
                        max_length=min(self.max_length, len(self.tokenizer.encode(prompt, add_special_tokens=True)) + 4096),
                        truncation=True,
                        stopping_criteria=stopping_criteria,
                        temperature=0.0
                    )
                ans = response[0]["generated_text"]


        else:
            if "openai" in self.llm_name.lower():
                client = openai.OpenAI()

                response = client.chat.completions.create(
                        model=self.model,
                        messages=messages
                )

                ans = response.choices[0].message.content

            else:
                stopping_criteria = None
                if prompt is None:
                    prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                if "meditron" in self.llm_name.lower():
                    stopping_criteria = self.custom_stop(["###", "User:", "\n\n\n"], input_len=len(self.tokenizer.encode(prompt, add_special_tokens=True)))
                if "llama-3" in self.llm_name.lower():
                    response = self.model(
                        prompt,
                        do_sample=False,
                        eos_token_id=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                        pad_token_id=self.tokenizer.eos_token_id,
                        max_length=min(self.max_length, len(self.tokenizer.encode(prompt, add_special_tokens=True)) + 4096),
                        truncation=True,
                        stopping_criteria=stopping_criteria,
                        temperature=0.0
                    )
                else:
                    response = self.model(
                        prompt,
                        do_sample=False,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.eos_token_id,
                        max_length=min(self.max_length, len(self.tokenizer.encode(prompt, add_special_tokens=True)) + 4096),
                        truncation=True,
                        stopping_criteria=stopping_criteria,
                        temperature=0.0
                    )
                ans = response[0]["generated_text"]
        return ans


class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_words, tokenizer, input_len=0):
        super().__init__()
        self.tokenizer = tokenizer
        self.stops_words = stop_words
        self.input_len = input_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        tokens = self.tokenizer.decode(input_ids[0][self.input_len:])
        return any(stop in tokens for stop in self.stops_words)    