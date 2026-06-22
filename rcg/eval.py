from datetime import datetime
import json
import os
import os
from pathlib import Path
from rcg.utils import start_logging, write_log, parse_manual_function_call, geneval
from rcg.prompt import FUNCTION_SCHEMAS, SYSTEM_PROMPT_CODE, SYSTEM_PROMPT_EVAL, get_system_prompt_code, get_system_prompt_eval
from rcg.llm import OpenAILLM
# from rcg.macro import learn_macros
from rcg.macro_stitch import rewrite_json_calls, sequence_to_expr, learn_macros
from tqdm import tqdm
import argparse
import csv

# Evaluation Prompts
prompts_objects = [
    "Grab the green cube",
    "Move 20cm above the blue can",
    "Grab the medium-sized cube",
    "Grab a yellow fruit",
    "Grab the cylindrical object",
    "Put the crackers on the other side of the table",
    "Move rightmost banana to the left of the can",
    "Move leftmost fruit 15cm closer to me",
    "Pick up closest food item to the red cube",
    "Move the orange to the center of the table",
    "Move leftmost banana 10cm to the right, then move rightmost banana 5cm to the left",
    "Pick up the apple and throw it on the ground",
    "Pick up the sphere, move it to the right 50cm, then release it",
    "Move to the smallest cube, then move to the largest cube",
    "Move the gripper down and to the right by 20cm, down and to the left by 20cm, up and to the left by 20cm, and up and to the right by 20cm",
    "Clear the table",
    "Remove the non-food items on the table",
    "Sort objects by shape",
    "Stack all the cubes",
    "Put warm-colored objects next to each other",
]

prompts_feeding = [
    "Give a strawberry to the user",
    "Hand the human something to drink",
    "Move the dangerous item away from the human",
    "Place the peach on the plate",
    "Put the fork up to the user's mouth",
    "Pick up the spoon and hand it to the user",
    "Swap the fork with the knife",
    "Move the glass to the left of the plate",
    "Feed the strawberry closest to the ham to the user",
    "Put the largest fruit onto the plate",
    "Collect all strawberries onto the plate",
    "Put the apple on the plate and use the knife to cut it",
    "Feed the apple first and then the orange to the user",
    "Assemble a ham sandwich on the plate",
    "Grasp the napkin and wipe the user's mouth",
    "Feed all the food on the table to the user",
    "Put all yellow fruits onto the plate",
    "Make a small snack by placing bread, ham, and a strawberry on the plate",
    "Put the peach and banana on the plate, then feed the banana to the user",
    "Prepare a healthy meal"
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

def add_info(model, env):
    try:
        # Get index of last get_info call
        last_index = len(model.conversation_history) - 1 - model.conversation_history[::-1].index({"role": "user", "content": "Get the current scene information"})
        
        # Remove last get_info call and response from history
        model.conversation_history.pop(last_index)
        model.conversation_history.pop(last_index)
    except ValueError:
        pass
    
    # Load pre-generated get_info data from file
    if env == "feeding":
        file_name = "get_info_feeding.json"
    else:
        file_name = "get_info_objects.json"

    with open(Path(__file__).parent / f"data/{file_name}", 'r', encoding='utf-8') as f:
        get_info = json.load(f)
    
    # Add current get_info data to history
    model.conversation_history.append({"role": "user", "content": "Get the current scene information"})
    model.conversation_history.append({"role": "assistant", "content": json.dumps(get_info, ensure_ascii=False)})

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate LLM-generated code for controlling a Kinova robot in Unity.")
    parser.add_argument("--env", type=str, choices=["objects", "feeding"], default="objects", help="Environment to use (default: objects)")
    parser.add_argument("--no-eval", action="store_true", help="Disable evaluation")
    parser.add_argument("--no-macro", action="store_true", help="Disable macro learning")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(__file__).parent / "results" / f"eval_{timestamp}.csv"

    # Initialize database
    traces = []
    macros = []

    # Initialize models
    if not OpenAILLM.API_KEY:
        raise ValueError("OpenAI API key not set")
    OpenAILLM.init_pipeline()
    code_model = OpenAILLM(system_prompt=SYSTEM_PROMPT_CODE, temperature=1.0, reasoning="medium")
    if not args.no_eval:
        eval_model = OpenAILLM(system_prompt=SYSTEM_PROMPT_EVAL, temperature=0.7)

    # Initialize logging
    start_logging("eval")
    write_log(f"=== LLM Evaluation Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    write_log(f"Code Model: {code_model.MODEL}")
    if not args.no_eval:
        write_log(f"Eval Model: {eval_model.MODEL}")
    write_log("")

    if "prompts" not in globals():
        prompts = prompts_feeding if args.env == "feeding" else prompts_objects
    
    # Evaluate each prompt multiple times
    for prompt in tqdm(prompts):
        total_time = 0
        successes = 0
        for i in range(num_reps):
            write_log(f"=== Iteration {i+1} of Prompt: {prompt} ===")
            start_time = datetime.now()
            success = True
            
            # Call get_info first, removing previous call
            add_info(code_model, args.env)
            if not args.no_eval:
                add_info(eval_model, args.env)

            # Generate and evaluate code
            try:
                if args.no_eval:
                    code_message = code_model.generate(prompt)
                    write_log(f"CODER RESULT: {code_message}\n")
                else:
                    code_message = geneval(code_model, eval_model, prompt, include_input_in_eval=True, macros=macros)                       
            except Exception as e:
                write_log(f"Error during evaluation: {e}\n")
                continue
            finally:
                code_model.reset()
                if not args.no_eval:
                    eval_model.reset()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            total_time += duration
            successes += 1

            # Compute experiment results
            parse_raw = parse_manual_function_call(code_message)
            write_log(f"Parsed functions: {parse_raw}\n")
            result = {
                "prompt": prompt,
                "code": code_message,
                "program_length": len(parse_raw),
                "macro_usage": len([f for f in parse_raw if f[0] not in [schema["name"] for schema in FUNCTION_SCHEMAS]]),
                "duration_seconds": duration,
            }

            # Append results to CSV file
            file_exists = os.path.exists(csv_path)
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=result.keys())

                if not file_exists:
                    writer.writeheader()

                writer.writerow(result)

            if not args.no_macro:
                # Append trace to database
                primitive_code_message = rewrite_json_calls(code_message)
                parsed_code = parse_manual_function_call(primitive_code_message)
                write_log(f"Parsed primitive functions: {parsed_code}\n")
                traces.append({
                    "timestamp": datetime.now().isoformat(),
                    "prompt": prompt,
                    "final_code": parsed_code,
                    "success": success,
                    "duration_seconds": duration,
                })

                # Lambda conversion and macro learning
                lambda_expr = sequence_to_expr(parsed_code)
                write_log(f"Lambda expression: {lambda_expr}\n")

                abstractions, macros = learn_macros(traces)
                write_log(f"Learned abstractions: {abstractions}\n")

                if macros:
                    write_log(f"MACROS:")
                    for schema in macros:
                        write_log(json.dumps(schema, indent=4, ensure_ascii=False) + "\n")

            code_model.reset(get_system_prompt_code(macros))
            if not args.no_eval:
                eval_model.reset(get_system_prompt_eval(macros))

        avg_time = total_time / successes if successes > 0 else float('inf')
        write_log(f"Success Rate for Prompt '{prompt}': {successes / num_reps}")
        write_log(f"Average Duration for Prompt '{prompt}': {avg_time:.1f} seconds\n\n")
