"""Generate answers with GPT-4

Usage:
python3 gen_api_answer.py --model gpt-3.5-turbo
"""
import argparse
import json
import os
import time
import concurrent.futures

import openai
import shortuuid
import tqdm
from transformers import AutoTokenizer
import google.generativeai as google_genai

from fastchat.llm_judge.common import (
    load_questions,
    temperature_config,
    chat_completion_openai,
    chat_completion_anthropic,
    chat_completion_palm,
    db_inference_deployment,
)
from fastchat.llm_judge.gen_model_answer import reorg_answer_file
from fastchat.model.model_adapter import get_conversation_template, ANTHROPIC_MODEL_LIST
from mistralai.client import MistralClient
from mistralai.exceptions import MistralAPIException

import requests
import logging
import time

log = logging.getLogger(__name__)

class _MistralClient:
    client = None
    
    @staticmethod
    def get():
        return _MistralClient.client
    
    @staticmethod
    def set(mistral_client):
        _MistralClient.client = mistral_client

def block_until_ready(base_url: str, max_minutes: int = 45):
    """Block until the endpoint is ready."""
    sleep_s = 5
    timout_s = max_minutes * 60  # At max, wait 5 minutes

    ping_url = f'{base_url}/ping'

    waited_s = 0
    while True:
        try:
            requests.get(ping_url)
            print (f'Endpoint {ping_url} is ready')
            break
        except requests.exceptions.ConnectionError:
            print (f'Endpoint {ping_url} not ready yet. Sleeping {sleep_s} seconds')
            time.sleep(sleep_s)
            waited_s += sleep_s

        if waited_s >= timout_s:
            raise TimeoutError(
                f'Endpoint {ping_url} did not become read after {waited_s:,} seconds, exiting'
            )


def get_answer(
    question: dict, model: str, tokenizer, num_choices: int, max_tokens: int, answer_file: str
):
    assert (
        args.force_temperature is not None and "required_temperature" in question.keys()
    ) == False
    if args.force_temperature is not None:
        temperature = args.force_temperature
    elif "required_temperature" in question.keys():
        temperature = question["required_temperature"]
    elif question["category"] in temperature_config:
        temperature = temperature_config[question["category"]]
    else:
        temperature = 0.7

    choices = []
    chat_state = None  # for palm-2 model
    question_id = question["question_id"]
    print(f"Starting question #{question_id}")
    for i in range(num_choices):
        conv = get_conversation_template(model)

        turns = []
        for j in range(len(question["turns"])):
            conv.append_message(conv.roles[0], question["turns"][j])
            conv.append_message(conv.roles[1], None)

            if model in ANTHROPIC_MODEL_LIST:
                output = chat_completion_anthropic(model, conv, temperature, max_tokens)
            elif model == "palm-2-chat-bison-001":
                chat_state, output = chat_completion_palm(
                    chat_state, model, conv, temperature, max_tokens
                )
            elif model.startswith('databricks'):
                api_key = os.environ["DATABRICKS_REAL_TOKEN"]
                output = db_inference_deployment(f'https://e2-dogfood.staging.cloud.databricks.com/serving-endpoints/{model}/invocations', tokenizer, conv, temperature, max_tokens, api_key=api_key, api_args={
                    'model': model,
                    'use_raw_prompt': None,
                    'prompt': None,
                    'messages': conv.to_openai_api_messages()
                })
            elif model == 'mistral-large':              
                if not _MistralClient.get():
                    _MistralClient.set(MistralClient(
                        endpoint=os.environ["MISTRAL_URL"],
                        api_key=os.environ["MISTRAL_API_KEY"],
                    ))
                    
                def retry_request(retry=5):
                    if retry == 0:
                        return None

                    try:
                        return _MistralClient.get().chat(
                            messages=conv.to_openai_api_messages(),
                            model='azureai',
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                    except MistralAPIException as e:
                        time.sleep(3 * retry)
                        _MistralClient.set()
                        
                        return retry_request(retry - 1)
                  
                chat_response = retry_request()
                output = chat_response.choices[0].message.content
            elif model.startswith("mistral-medium"):
                if not _MistralClient.get():
                    _MistralClient.set(MistralClient(
                        api_key=os.environ["MISTRAL_MEDIUM_API_KEY"],
                    ))
                    
                def retry_request(retry=5):
                    if retry == 0:
                        return None

                    try:
                        return _MistralClient.get().chat(
                            messages=conv.to_openai_api_messages(),
                            model=model,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                    except MistralAPIException as e:
                        time.sleep(3 * retry)
                        _MistralClient.set()
                        
                        return retry_request(retry - 1)
                  
                chat_response = retry_request()
                output = chat_response.choices[0].message.content
            elif model == 'gemini-1.0-pro-latest':
                google_genai.configure(api_key=os.environ['GEMINI_API_KEY'])
                google_model = google_genai.GenerativeModel(model)
                ignore = [
                    google_genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    google_genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    google_genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    google_genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                ]
                safety_settings = {
                    category: google_genai.types.HarmBlockThreshold.BLOCK_NONE
                    for category in ignore
                }
                
                generation_config=google_genai.types.GenerationConfig(
                    candidate_count=1,
                    max_output_tokens=max_tokens,
                    temperature=temperature
                )
                
                result = google_model.generate_content(
                    [ { "role": p["role"], "parts": [ {"text": p["content"] } ] }  for p in conv.to_openai_api_messages() ],
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )  
                
                print(result)
                output = result.result.candidates[0].content.parts[0].text
                
            elif model == 'meta-llama/Llama-2-70b-chat-hf':
                output = chat_completion_openai(model, conv, temperature, max_tokens, api_dict={
                    'api_base': 'https://95f8e758a80a.ngrok.app/v1',
                    'api_key': 'free'
                }) 
            elif "https://" in model or "http://" in model:
                block_until_ready(model)
                api_key = os.environ.get("MOSAICML_API_KEY", None)
                output = db_inference_deployment(model, tokenizer, conv, temperature, max_tokens, api_key=api_key)
            else:
                output = chat_completion_openai(model, conv, temperature, max_tokens)

            conv.update_last_message(output)
            turns.append(output)

        choices.append({"index": i, "turns": turns})

    question_id = question["question_id"]
    print(f"Completed question #{question_id}")
    # Dump answers
    ans = {
        "question_id": question["question_id"],
        "answer_id": shortuuid.uuid(),
        "model_id": model,
        "choices": choices,
        "tstamp": time.time(),
    }

    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    with open(answer_file, "a") as fout:
        fout.write(json.dumps(ans) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument("--model_name", type=str, default="model")
    parser.add_argument(
        "--force-temperature", type=float, help="Forcibly set a sampling temperature."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument(
        "--parallel", type=int, default=1, help="The number of concurrent API calls."
    )
    parser.add_argument("--openai-api-base", type=str, default=None)
    args = parser.parse_args()

    if args.openai_api_base is not None:
        openai.api_base = args.openai_api_base

    question_file = f"data/{args.bench_name}/question.jsonl"
    questions = load_questions(question_file, args.question_begin, args.question_end)

    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_name}.jsonl"
    print(f"Output to {answer_file}")

    tokenizer = AutoTokenizer.from_pretrained("rajammanabrolu/gpt-4-chat", trust_remote_code=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = []
        for question in questions:
            future = executor.submit(
                get_answer,
                question,
                args.model,
                tokenizer,
                args.num_choices,
                args.max_tokens,
                answer_file,
            )
            futures.append(future)

        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()

    reorg_answer_file(answer_file)
