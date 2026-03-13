from datetime import datetime
import json
from pathlib import Path
import re
from typing import Optional
from rcg.prompt import FUNCTION_SCHEMAS, SYSTEM_PROMPT_CODE, SYSTEM_PROMPT_EVAL, SYSTEM_PROMPT_PLAN, SYSTEM_PROMPT_CODEPLAN
from rcg.llm import OpenAILLM, LocalLLM, LoRALLM
from rcg.val import prompt_to_code_general
from distutils.util import strtobool
from tqdm import tqdm

prompts = [
    "Show me all objects in the scene",
    "Move to banana 3",
    "Move to position [1, 2, 1]",
    "Grasp the object*",
    "Release the object**",
    "Pick up the object, more to the right by 30cm, then release the object*",
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
]

num_reps = 5
log_file = "eval_log.txt"


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

def static_evaluation(code: str) -> tuple[bool, str]:
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
        
        for schema in FUNCTION_SCHEMAS:
            # Check if function exists
            if schema["name"] == function_name:
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

def geneval(code_model, eval_model, user_input, include_input_in_eval, eval_attempts=5, code_message=None, name=""):
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
            correct, eval_message = static_evaluation(code_message)
        else:
            # Evaluate code with LLM
            eval_prompt = f"User Request: {user_input}\nCode: {code_message}" if include_input_in_eval else f"Code: {code_message}"
            eval_message = eval_model.generate(eval_prompt)
            found = re.search(r"\b(True|False)\b", eval_message, re.IGNORECASE)
            if found:
                correct = strtobool(found.group(1))

        write_log(f"{name} EVALUATOR RESULT: {eval_message}\n")
        eval_counter += 1

    if eval_model is not None:
        eval_model.reset()
    if eval_counter >= eval_attempts and not correct:
        raise RuntimeError(f'Generated code failed {name} evaluation')

    return code_message

if __name__ == "__main__":

    # Initialize models
    if not OpenAILLM.API_KEY:
        raise ValueError("OpenAI API key not set")
    LoRALLM.init_pipeline()
    OpenAILLM.init_pipeline()
    gen_model = OpenAILLM(system_prompt=SYSTEM_PROMPT_CODEPLAN, temperature=1.0, reasoning="medium")
    eval_model = OpenAILLM(system_prompt=SYSTEM_PROMPT_EVAL, temperature=0.7)

    # Initialize logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(__file__).parent.parent / "log"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"eval_{timestamp}.log"
    write_log(f"=== LLM Evaluation Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    write_log(f"Generator Model: {gen_model.MODEL}")
    write_log(f"Evaluator Model: {eval_model.MODEL}")
    write_log("")

    # Evaluate each prompt multiple times
    for prompt in tqdm(prompts):
        total_time = 0
        successes = 0
        
        for i in range(num_reps):
            write_log(f"=== Iteration {i+1} of Prompt: {prompt} ===")
            start_time = datetime.now()
            
            # Call get_info first, removing previous call
            add_info(gen_model, prompt)
            add_info(eval_model, prompt)
            user_input = prompt.replace("*", "")

            try:
                # Functional evaluation
                code_message = geneval(gen_model, eval_model, user_input, include_input_in_eval=True, name="Function")
                # Static evaluation
                code_message = geneval(gen_model, None, user_input, include_input_in_eval=False, code_message=code_message, name="Syntax")
                # Final generated code
                parsed_code = parse_manual_function_call(code_message)
                write_log(f"Parsed Function:\n{parsed_code}\n")
            except Exception as e:
                write_log(f"Error during evaluation: {e}\n")
                continue
            finally:
                gen_model.reset()
                eval_model.reset()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            total_time += duration
            successes += 1

        avg_time = total_time / successes if successes > 0 else float('inf')
        write_log(f"Success Rate for Prompt '{prompt}': {successes / num_reps}")
        write_log(f"Average Duration for Prompt '{prompt}': {avg_time:.1f} seconds\n\n")
