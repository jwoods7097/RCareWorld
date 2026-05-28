import re
from distutils.util import strtobool
from datetime import datetime
from pathlib import Path
import json
from rcg.prompt import FUNCTION_SCHEMAS

LOG_FILE = ""

def start_logging(file_prefix: str):
    """Configure logging settings."""
    global LOG_FILE

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(__file__).parent.parent / "log"
    log_dir.mkdir(exist_ok=True)
    LOG_FILE = log_dir / f"{file_prefix}_{timestamp}.log"

def write_log(message: str):
    """Write message to log file."""
    if not LOG_FILE:
        return
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
    except Exception as e:
        print(f"[Warning] Failed to write log: {e}")

def parse_manual_function_call(text: str) -> list[tuple[str, str]]:
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
        print(f"Parsed functions: {functions}")
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

def geneval(code_model, eval_model, user_input, include_input_in_eval, macros = [], eval_attempts=5, code_message=None):
    """Generate and evaluate the code with the specified evaluation model."""
    correct = False
    eval_counter = 0
    eval_message = ""
    
    # Ensure generated code is correct
    passed_function = False
    while not correct and eval_counter < eval_attempts:
        # Call code model
        if code_message is None or eval_counter > 0:
            code_message = code_model.generate(user_input if not eval_message else eval_message)
            write_log(f"CODER RESULT: {code_message}\n")

        # Evaluate code with LLM
        eval_prompt = f"User Request: {user_input}\nCode: {code_message}" if include_input_in_eval else f"Code: {code_message}"
        eval_message = eval_model.generate(eval_prompt)
        write_log(f"FUNCTION EVALUATOR RESULT: {eval_message}\n")
        found = re.search(r"\b(True|False)\b", eval_message, re.IGNORECASE)
        if found:
            correct = strtobool(found.group(1))

        # Evaluate code with static analysis
        if correct:
            passed_function = True
            correct, eval_message = static_evaluation(code_message, macros)
            write_log(f"SYNTAX EVALUATOR RESULT: {eval_message}\n")

        eval_counter += 1

    if eval_model is not None:
        eval_model.reset()
    if eval_counter >= eval_attempts and not correct:
        name = "SYNTAX" if passed_function else "FUNCTION"
        raise RuntimeError(f'Generated code failed {name} evaluation')
    
    return code_message