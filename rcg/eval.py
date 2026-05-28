from datetime import datetime
import json
from pathlib import Path
from rcg.utils import start_logging, write_log, parse_manual_function_call, geneval
from rcg.prompt import FUNCTION_SCHEMAS, SYSTEM_PROMPT_CODE, SYSTEM_PROMPT_EVAL, get_system_prompt_code, get_system_prompt_eval
from rcg.llm import OpenAILLM
# from rcg.macro import learn_macros
from rcg.macro_stitch import rewrite_json_calls, sequence_to_expr, learn_macros
from tqdm import tqdm

# Evaluation Prompts
prompts = [
    "Show me all objects in the scene",
    "Move to banana 3",
    "Move to position [1, 2, 1]",
    "Grasp the object*",
    "Release the object**",
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
prompts = [
    "Move to and grasp Banana 1",
    "Move to and grasp Banana 1",
    "Move to and grasp Banana 1",
    "Move to and grasp Banana 1",
    "Move to and grasp Banana 2",
    "Move to and grasp Banana 2",
]

num_reps = 1

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

if __name__ == "__main__":

    # Initialize database
    try:
        with open('rcg/data/traces.json', 'r', encoding='utf-8') as f:
            traces = json.load(f)
    except FileNotFoundError:
        traces = []
    macros = []

    try:

        # Initialize models
        if not OpenAILLM.API_KEY:
            raise ValueError("OpenAI API key not set")
        # LoRALLM.init_pipeline()
        OpenAILLM.init_pipeline()
        code_model = OpenAILLM(system_prompt=SYSTEM_PROMPT_CODE, temperature=1.0, reasoning="medium")
        eval_model = OpenAILLM(system_prompt=SYSTEM_PROMPT_EVAL, temperature=0.7)

        # Initialize logging
        start_logging("eval")
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
                    # Generate and evaluate code
                    code_message = geneval(code_model, eval_model, user_input, include_input_in_eval=True, macros=macros)
                    code_message = rewrite_json_calls(code_message)

                    # Try to parse manual function call from text
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
                        write_log(json.dumps(schema, indent=4, ensure_ascii=False) + "\n")

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
        with open('rcg/data/schemas.json', 'w', encoding='utf-8') as f:
            json.dump(FUNCTION_SCHEMAS + macros, f, ensure_ascii=False, indent=4)
