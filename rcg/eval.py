from datetime import datetime
import json
from pathlib import Path
import re
from typing import Optional
from rcg.prompt import FUNCTION_SCHEMAS, SYSTEM_PROMPT_CODE, SYSTEM_PROMPT_EVAL, SYSTEM_PROMPT_NGRAM, SYSTEM_PROMPT_PLAN, SYSTEM_PROMPT_TOPK, get_system_prompt_code, get_system_prompt_eval
from rcg.llm import OpenAILLM, LocalLLM, LoRALLM
from rcg.val import prompt_to_code_general
# from rcg.macro import learn_macros
from rcg.macro_stitch import sequence_to_expr, learn_macros, build_function_schema
from distutils.util import strtobool
from tqdm import tqdm
import tiktoken
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Evaluation Prompts
prompts = [
    # "Show me all objects in the scene",
    # "Move to banana 3",
    # "Move to position [1, 2, 1]",
    # "Grasp the object*",
    # "Release the object**",
    "Pick up the object, move to the right by 30cm, then release the object*",
    "Move to banana 3 and pick it up",
    "Move to the left by 30cm, up by 10cm, then to the right by 20cm",
    "Move the gripper down and to the right by 20cm, down and to the left by 20cm, up and to the left by 20cm, and up and to the right by 20cm",
    "Is there a banana in this scene? If so, move to it. Otherwise, move to the left by 25cm",
    "Move the leftmost banana so that it is now the middle banana",
    "Move the rightmost banana forward by 10cm",
    "Move banana 1 to the left of banana 3",
    "Move banana 2 10cm closer to me",
    "Pick up the closest banana to the camera",
    "Move the gripper in a circle",
    "Move to all 3 bananas in sequence",
    "Clear all bananas off the table",
    "Move all the bananas next to each other",
    "Put all bananas in a line",
    # "Pick up leftmost banana"
]

# Macro Learning Test Prompts
# prompts = [
#     "Move to and grasp Banana 1",
#     "Move to and grasp Banana 2",
#     "Move to and grasp Banana 3",
#     "Move to and grasp the leftmost banana",
#     "Move to and grasp the rightmost banana",
#     "Move to and grasp the middle banana",
# ]

# Macro Modification Test Prompts
# prompts = [
#     "Move to and grasp Banana 1",
#     "Move to and grasp Banana 1",
#     "Move to and grasp Banana 1",
#     "Move to and grasp Banana 1",
#     "Move to and grasp Banana 2",
#     "Move to and grasp Banana 2",
# ]

num_reps = 1
log_file = "eval_log.txt"


def ordered_subsets(lst, min_len=2, max_len=10):
    n = len(lst)
    max_len = min(max_len, n)
    
    for i in range(n):
        for j in range(i + min_len, min(i + max_len, n) + 1):
            yield lst[i:j]

def write_log(message: str):
    """Write message to log file."""
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
    except Exception as e:
        print(f"[Warning] Failed to write log: {e}")

def add_info(model, prompt: str):
    try:
        # Get index of last get_info call
        last_index = len(model.conversation_history) - 1 - model.conversation_history[::-1].index({"role": "user", "content": "Get the current scene information"})
        
        # Remove last get_info call and response from history
        model.conversation_history.pop(last_index)
        model.conversation_history.pop(last_index)
    except ValueError:
        pass
    
    # Load pre-generated get_info data from file
    with open(Path(__file__).parent / "data/get_info.json", 'r', encoding='utf-8') as f:
        get_info = json.load(f)
    i = prompt.count("*")
    
    # Add current get_info data to history
    model.conversation_history.append({"role": "user", "content": "Get the current scene information"})
    model.conversation_history.append({"role": "assistant", "content": json.dumps(get_info[i], ensure_ascii=False)})

def parse_manual_function_call(text: str) -> Optional[tuple]:
    """
    Manually parse JSON-formatted function calls from LLM output.
    Returns: (function_name, function_args) or None
    """
    found_functions = []

    # Remove <think> tag content
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    # Find JSON-formatted function calls
    # Pattern matches: {"function": "xxx", "args": {...}}
    json_pattern = r'\{["\']function["\']\s*:\s*["\'](\w+)["\']\s*,\s*["\']args["\']\s*:\s*(\{[^}]*\})\s*\}'
    matches = re.findall(json_pattern, text)
    for match in matches:
        function_name = match[0]
        args_str = match[1]
        found_functions.append((function_name, args_str))

    # Also try matching JSON in code blocks
    code_block_pattern = r'```json\s*\n\s*\{["\']function["\']\s*:\s*["\'](\w+)["\']\s*,\s*["\']args["\']\s*:\s*(\{[^}]*\})\s*\}\s*\n\s*```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    for match in matches:
        function_name = match[0]
        args_str = match[1]
        if (function_name, args_str) not in found_functions:
            found_functions.append((function_name, args_str))

    return found_functions

def static_evaluation(code: str, macros: list = []) -> tuple[bool, str]:
    """Static evaluation of generated code."""
    
    # Parse function calls from code
    try:
        functions = parse_manual_function_call(code)
    except:
        return (False, f"Code is not in valid JSON format")
        
    result = True
    reasoning = ""
    
    for function in functions:
        function_name, function_args_str = function
        try:
            function_args = json.loads(function_args_str)
        except:
            result = False
            reasoning += f"Arguments for function '{function_name}' are not valid JSON.\n"
            function_args = {}
        
        for schema in FUNCTION_SCHEMAS + macros:
            # Check if function exists
            if schema["name"] == function_name:
                # Check required arguments
                for req_arg in schema["parameters"]["required"]:
                    if req_arg not in function_args:
                        result = False
                        reasoning += f"Missing required argument '{req_arg}' for function '{function_name}'.\n"
                
                # Check provided arguments
                for arg_name, arg_value in function_args.items():
                    if arg_name not in schema["parameters"]["properties"]:
                        result = False
                        reasoning += f"Unexpected argument '{arg_name}' for function '{function_name}'.\n"
                    
                    else:
                        # Check argument type
                        expected_type = schema["parameters"]["properties"][arg_name]["type"]
                        if expected_type == "string" and not isinstance(arg_value, str):
                            result = False
                            reasoning += f"Argument '{arg_name}' for function '{function_name}' should be a string.\n"
                        elif expected_type == "number" and not isinstance(arg_value, (int, float)):
                            result = False
                            reasoning += f"Argument '{arg_name}' for function '{function_name}' should be a number.\n"
                        elif expected_type == "boolean" and not isinstance(arg_value, bool):
                            result = False
                            reasoning += f"Argument '{arg_name}' for function '{function_name}' should be a boolean.\n"
                break
        else:
            result = False
            reasoning += f"Function '{function_name}' is not a valid function.\n"
    
    return result, f"{result}\n{reasoning}"

def functions_equal(functions1, functions2):
    """Compare two parsed functions to see if they're equal."""

    # Baseline check to ensure same number of functions
    if len(functions1) != len(functions2):
        return False

    for f1, f2 in zip(functions1, functions2):
        f1_name, f1_args_str = f1
        f2_name, f2_args_str = f2

        # Make sure functions are the same
        if f1_name != f2_name:
            return False
        
        try:
            f1_args = json.loads(f1_args_str)
            f2_args = json.loads(f2_args_str)
        except:
            return False
        
        # Baseline check to ensure same number of arguments
        if len(f1_args) != len(f2_args):
            return False
        
        # Make sure argument names and values are the same
        for (a1n, a1v), (a2n, a2v) in zip(f1_args.items(), f2_args.items()):
            if a1n != a2n or a1v != a2v:
                return False
    
    return True

def geneval(code_model, eval_model, user_input, include_input_in_eval, macros = [], eval_attempts=5, code_message=None, name=""):
    """Generate and evaluate the code with the specified evaluation model."""
    correct = False
    eval_counter = 0
    eval_message = ""
    
    # Ensure generated code is correct
    while not correct and eval_counter < eval_attempts:
        # Call code model
        if code_message is None or eval_counter > 0:
            code_message = code_model.generate(user_input if not eval_message else eval_message)
            write_log(f"CODER RESULT: {code_message}\n")

        if eval_model is None:
            # Evaluate code with static evaluator
            correct, eval_message = static_evaluation(code_message, macros)
        else:
            # Evaluate code with LLM
            eval_prompt = f"User Request: {user_input}\nCode: {code_message}" if include_input_in_eval else f"Code: {code_message}"
            eval_message = eval_model.generate(eval_prompt)
            found = re.search(r"\b(True|False)\b", eval_message, re.IGNORECASE)
            if found:
                correct = strtobool(found.group(1))

        eval_counter += 1
        write_log(f"{name} EVALUATOR RESULT: {eval_message}\n")

    if eval_model is not None:
        eval_model.reset()
    if eval_counter >= eval_attempts and not correct:
        raise RuntimeError(f'Generated code failed {name} evaluation')

    return code_message

def topk_tokens(prompt, code, k=5):
    enc = tiktoken.get_encoding("cl100k_base")

    # -------------- tokenizer (use cl100k_base for OpenAI models) --------------
    enc = tiktoken.get_encoding("cl100k_base")

    # token ids
    src_ids = enc.encode(prompt)
    tgt_ids = enc.encode(json.dumps(code))

    # convert each token id back to readable string for labels
    # note: decoding single token gives string possibly with leading spaces — that's fine for labels
    src_tokens = [enc.decode([tid]) for tid in src_ids]
    tgt_tokens = [enc.decode([tid]) for tid in tgt_ids]

    # -------------- get embeddings for each token string --------------
    # choose embeddings model (text-embedding-3-small or similar)
    EMB_MODEL = "text-embedding-3-small"

    def get_embeddings_for_token_list(tokens):
        # The API supports batch inputs; we pass the list of token strings
        # we call the embeddings endpoint and unpack the vectors.
        resp = openai.embeddings.create(model=EMB_MODEL, input=tokens)
        # resp.data is a list aligned with input tokens
        embeddings = np.array([item.embedding for item in resp.data], dtype=np.float32)
        return embeddings

    # For each token we ask for an embedding of that token's string
    # (Note: tokens often include leading space; that's OK and preserves token semantics)
    src_emb = get_embeddings_for_token_list(src_tokens)   # shape (src_len, dim)
    tgt_emb = get_embeddings_for_token_list(tgt_tokens)   # shape (tgt_len, dim)

    # src_tokens, src_emb = merge_tokens_to_words(src_tokens, src_emb)
    # tgt_tokens, tgt_emb = merge_tokens_to_words(tgt_tokens, tgt_emb)

    # -------------- compute similarity matrix --------------
    # We'll compute cosine similarity between each target token and every source token
    # result shape: (tgt_len, src_len)
    sim = cosine_similarity(tgt_emb, src_emb)  # rows: tgt tokens, cols: src tokens

    # Normalize per-target so rows sum to 1 (acts like attention distribution)
    sim_rownorm = sim / (sim.sum(axis=1, keepdims=True) + 1e-12)

    sim_max = sim_rownorm.max(axis=0)

    topk_idx = np.argsort(sim_max)[::-1]
    topk_tokens = [src_tokens[idx] for idx in topk_idx if src_tokens[idx].strip()][:k]
    return topk_tokens

if __name__ == "__main__":

    # Initialize database
    try:
        with open('rcg/data/traces.json', 'r', encoding='utf-8') as f:
            traces = json.load(f)
    except FileNotFoundError:
        traces = []
    try:
        with open('rcg/data/ngrams.json', 'r', encoding='utf-8') as f:
            ngrams = json.load(f)
    except FileNotFoundError:
        ngrams = {}
    macros = []

    try:

        # Initialize models
        if not OpenAILLM.API_KEY:
            raise ValueError("OpenAI API key not set")
        # LoRALLM.init_pipeline()
        OpenAILLM.init_pipeline()
        code_model = OpenAILLM(system_prompt=SYSTEM_PROMPT_CODE, temperature=1.0, reasoning="medium")
        eval_model = OpenAILLM(system_prompt=SYSTEM_PROMPT_EVAL, temperature=0.7)
        ngram_model = OpenAILLM(system_prompt=SYSTEM_PROMPT_NGRAM, temperature=0.1)

        # Initialize logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(__file__).parent.parent / "log"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"eval_{timestamp}.log"
        write_log(f"=== LLM Evaluation Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
        write_log(f"Code Model: {code_model.MODEL}")
        write_log(f"Eval Model: {eval_model.MODEL}")
        write_log("")

        # Evaluate each prompt multiple times
        for prompt in tqdm(prompts):
            total_time = 0
            successes = 0
            for i in range(num_reps):
                write_log(f"=== Iteration {i+1} of Prompt: {prompt} ===")
                start_time = datetime.now()
                success = True
                
                # Call get_info first, removing previous call
                add_info(code_model, prompt)
                add_info(eval_model, prompt)
                user_input = prompt.replace("*", "")

                try:
                    # Functional evaluation
                    code_message = geneval(code_model, eval_model, user_input, include_input_in_eval=True, macros=macros, name="Function")
                    # Static evaluation
                    code_message = geneval(code_model, None, user_input, include_input_in_eval=False, code_message=code_message, macros=macros, name="Syntax")
                    # Final generated code
                    parsed_code = parse_manual_function_call(code_message)
                    write_log(f"Parsed functions: {parsed_code}\n")
                except Exception as e:
                    write_log(f"Error during evaluation: {e}\n")
                    continue
                finally:
                    code_model.reset()
                    eval_model.reset()
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                total_time += duration
                successes += 1
                # if functions_equal(final_valid_code, final_code):
                #     successes += 1
                #     success = True

                traces.append({
                    "timestamp": datetime.now().isoformat(),
                    "prompt": user_input,
                    "final_code": parsed_code,
                    "success": success,
                    "duration_seconds": duration,
                })

                lambda_expr = sequence_to_expr(parsed_code)
                write_log(f"Lambda expression: {lambda_expr}\n")

                abstractions, macros = learn_macros(traces)
                write_log(f"Learned abstractions: {abstractions}\n")

                if macros:
                    write_log(f"MACROS:")
                    for schema in macros:
                        write_log(json.dumps(schema, indent=4) + "\n")         

                # ngram_prompt = f"User Request: {prompt}\nCode: {json.dumps(parsed_code, ensure_ascii=False)}"
                # ngram_response = ngram_model.generate(ngram_prompt, memory=False)
                # write_log(f"Ngram response for generated code:\n{ngram_response}\n")
                # selected_ngrams = [
                #     [parsed_code[int(i.strip()) - 1] for i in line.strip().split(",") if i.strip()]
                #     for line in ngram_response.split("\n") if line.strip()
                # ]

                # for ngram in selected_ngrams:
                #     if len(ngram) < 2:
                #         continue
                    
                #     ngram_key = json.dumps([n[0] for n in ngram], ensure_ascii=False)
                #     if ngram_key not in ngrams:
                #         ngrams[ngram_key] = {"traces": []}

                #     # topk = topk_tokens(prompt, ngram)
                #     topk_prompt = f"User Request: {prompt}\nCode: {json.dumps(ngram, ensure_ascii=False)}"
                #     topk_response = topk_model.generate(topk_prompt, memory=False)
                #     write_log(f"Top-k response for ngram {ngram_key}: {topk_response}\n")
                #     topk = [t.strip().lower() for t in topk_response.split(",") if t.strip()]

                #     ngrams[ngram_key]["traces"].append({
                #         "prompt": prompt,
                #         "code": [{"name": n[0], "args": json.loads(n[1])} for n in ngram],
                #         "topk_tokens": topk
                #     })

                # old_names, learned_ngrams, new_macros = learn_macros(ngrams)
                # for old_name, ngram, macro in zip(old_names, learned_ngrams, new_macros):                    
                #     for f in range(len(FUNCTION_SCHEMAS)):
                #         if FUNCTION_SCHEMAS[f]["name"] == old_name:
                #             FUNCTION_SCHEMAS[f] = macro
                #             write_log(f"MODIFIED MACRO for {ngram}: {old_name} is now {macro['name']}\nDescription: {macro['description']}\nParameters: {json.dumps(macro['parameters'], indent=4)}\n")
                #             break
                #     else:
                #         FUNCTION_SCHEMAS.append(macro)
                #         write_log(f"LEARNED NEW MACRO for {ngram}: {macro['name']}\nDescription: {macro['description']}\nParameters: {json.dumps(macro['parameters'], indent=4)}\n")

                code_model.reset(get_system_prompt_code(macros))
                eval_model.reset(get_system_prompt_eval(macros))

            avg_time = total_time / successes if successes > 0 else float('inf')
            write_log(f"Success Rate for Prompt '{prompt}': {successes / num_reps}")
            write_log(f"Average Duration for Prompt '{prompt}': {avg_time:.1f} seconds\n\n")

    except KeyboardInterrupt:
        pass
    finally:
        # with open('rcg/data/traces.json', 'w', encoding='utf-8') as f:
        #     json.dump(traces, f, ensure_ascii=False, indent=4)
        with open('rcg/data/ngrams.json', 'w', encoding='utf-8') as f:
            json.dump(ngrams, f, ensure_ascii=False, indent=4)
        with open('rcg/data/schemas.json', 'w', encoding='utf-8') as f:
            json.dump(FUNCTION_SCHEMAS + macros, f, ensure_ascii=False, indent=4)
