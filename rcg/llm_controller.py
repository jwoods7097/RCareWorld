import os
import json
import re
import traceback
from datetime import datetime
from typing import Dict, Any

# Import prompts from prompt.py
from rcg.prompt import SYSTEM_PROMPT_CODE, SYSTEM_PROMPT_EVAL, SYSTEM_PROMPT_SUMMARY, get_system_prompt_code, get_system_prompt_eval
# from rcg.learned_macros import *
from rcg.macro_stitch import learn_macros, sequence_to_expr, rewrite_json_calls
from rcg.llm import OpenAILLM
from rcg.robot import get_info, execute_function
from rcg.utils import geneval, parse_manual_function_call, start_logging, write_log, LOG_FILE

# ============================================================================
# LLM Controller
# ============================================================================

class LLMController:
    """LLM controller for processing natural language commands."""
    
    def __init__(self, enable_logging: bool = True, show_function_calls: bool = True):
        """Initialize LLM controller."""
        self.traces = []
        self.macros = []

        # Initialize models
        if not OpenAILLM.API_KEY:
            raise ValueError("OpenAI API key not set")
        OpenAILLM.init_pipeline()
        self.code_model = OpenAILLM(system_prompt=SYSTEM_PROMPT_CODE, temperature=1.0, reasoning="medium")
        self.eval_model = OpenAILLM(system_prompt=SYSTEM_PROMPT_EVAL, temperature=0.7)
        self.summary_model = OpenAILLM(system_prompt=SYSTEM_PROMPT_SUMMARY, temperature=0.7)

        # Initialize logging
        self.enable_logging = enable_logging
        self.show_function_calls = show_function_calls
        if self.enable_logging:
            start_logging("llm")
            write_log(f"Code Model: {self.code_model.MODEL}")
            write_log(f"Eval Model: {self.eval_model.MODEL}")
            write_log(f"Summary Model: {self.summary_model.MODEL}")
            write_log("")

        print(f"[LLM Controller] Initialized")
        if self.enable_logging:
            print(f"[LLM Controller] Logging to {LOG_FILE}")
    
    def process_command(self, user_input: str) -> Dict[str, Any]:
        """Process user command using wrapper function calling."""
        # Log user input
        if self.enable_logging:
            write_log("─" * 80)
            write_log(f"[{datetime.now().strftime('%H:%M:%S')}] USER: {user_input}")
            write_log("")

        # Call get_info first, removing previous call
        info = get_info()
        self.code_model.add_info(info)
        self.eval_model.add_info(info)

        try:
            # Generate and evaluate code
            code_message = geneval(self.code_model, self.eval_model, user_input, include_input_in_eval=True, macros=self.macros)
            code_message = rewrite_json_calls(code_message)

            # Try to parse manual function call from text
            parsed_functions = parse_manual_function_call(code_message)

            # Check for function call
            if parsed_functions:
                function_names = []
                function_args_list = []
                function_results = []
                result_message = ""

                for parsed in parsed_functions:
                    # Found function calls
                    function_name, function_args_str = parsed
                    function_names.append(function_name)
                    function_args = json.loads(function_args_str)
                    function_args_list.append(function_args)

                    # Log function call
                    if self.enable_logging:
                        write_log(f"FUNCTION CALL DETECTED: {function_name}")
                        write_log(f"Arguments: {json.dumps(function_args, indent=2, ensure_ascii=False)}")
                        write_log("")

                    if self.show_function_calls:
                        print(f"\n[LLM] Function call: {function_name}")
                        print(f"[LLM] Arguments: {json.dumps(function_args, indent=2)}")

                    # Execute function
                    function_result = execute_function(function_name, function_args)
                    function_results.append(function_result if function_result else {})

                    # Log function result
                    if self.enable_logging:
                        write_log(f"FUNCTION RESULT:")
                        write_log(f"  Success: {function_result.get('success', False)}")
                        write_log(f"  Message: {function_result.get('message', 'N/A')}")
                        if function_result.get('data'):
                            data_str = json.dumps(function_result['data'], indent=2, ensure_ascii=False)
                            write_log(f"  Data: {data_str}")
                        write_log("")

                    # Create user message with function result
                    result_message += f"Function {function_name} returned: {json.dumps(function_result)}\n"

                # Get final response
                final_message = self.summary_model.generate(f"User Request: {user_input}\nFunction Results:\n{result_message}", memory=False)

                # Remove <think> tags from final message
                final_message = re.sub(r'<think>.*?</think>', '', final_message, flags=re.DOTALL).strip()

                # Log LLM response
                if self.enable_logging:
                    write_log(f"LLM RESPONSE:")
                    write_log(final_message)
                    write_log("")

                self.traces.append({
                    "timestamp": datetime.now().isoformat(),
                    "prompt": user_input,
                    "final_code": parsed_functions,
                })

                lambda_expr = sequence_to_expr(parsed_functions)
                write_log(f"Lambda expression: {lambda_expr}\n")

                abstractions, self.macros = learn_macros(self.traces)
                write_log(f"Learned abstractions: {abstractions}\n")

                if self.macros:
                    write_log(f"MACROS:")
                    for schema in self.macros:
                        write_log(json.dumps(schema, indent=4, ensure_ascii=False) + "\n")

                self.code_model.update_system_prompt(get_system_prompt_code(self.macros))
                self.eval_model.update_system_prompt(get_system_prompt_eval(self.macros))

                return {
                    "success": True,
                    "function_called": function_names,
                    "function_args": function_args_list,
                    "function_result": function_results,
                    "llm_response": final_message
                }
            else:
                # No function call at all
                # Log LLM response
                if self.enable_logging:
                    write_log(f"LLM RESPONSE (no function call):")
                    write_log(code_message)
                    write_log("")

                return {
                    "success": True,
                    "function_called": None,
                    "llm_response": code_message
                }
        
        except Exception as e:
            traceback.print_exc()
            return {
                "success": False,
                "error": f"LLM error: {str(e)}"
            }
    
    def reset(self):
        """Reset conversation history."""
        self.code_model.reset()
        print("[LLM Controller] Conversation history reset")
        if self.enable_logging:
            write_log("\n" + "="*80)
            write_log("CONVERSATION RESET")
            write_log("="*80 + "\n")
