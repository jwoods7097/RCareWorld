from stitch_core import compress
import json
import re
from typing import Any, Iterable, List, Tuple, Dict
from collections import defaultdict
from rcg.llm import OpenAILLM
from rcg.prompt import FUNCTION_SCHEMAS, SYSTEM_PROMPT_DOCUMENT

document_model = OpenAILLM(system_prompt=SYSTEM_PROMPT_DOCUMENT, temperature=0.7)

lambda_traces = []


def tokenize_sexpr(s: str) -> List[str]:
    """
    Split an S-expression string into tokens.
    Keeps quoted strings together.
    """
    return re.findall(r'\(|\)|"[^"\\]*(?:\\.[^"\\]*)*"|[^\s()]+', s)


def parse_sexpr(s: str) -> Any:
    """
    Parse an S-expression string into nested Python lists.
    Example:
        (then (f a b) (g c))
        -> ['then', ['f', 'a', 'b'], ['g', 'c']]
    """
    tokens = tokenize_sexpr(s)
    stack = []
    current = []

    for tok in tokens:
        if tok == '(':
            stack.append(current)
            new_list = []
            current.append(new_list)
            current = new_list
        elif tok == ')':
            if not stack:
                raise ValueError("Unbalanced closing parenthesis")
            current = stack.pop()
        else:
            if tok.startswith('"') and tok.endswith('"'):
                tok = json.loads(tok)
            current.append(tok)

    if stack:
        raise ValueError("Unbalanced opening parenthesis")

    if len(current) == 1:
        return current[0]
    return current


def collect_var_usage(expr: Any) -> Dict[str, List[Tuple[str, str]]]:
    """
    Walk the parsed tree and collect:
        #0 -> [(function_name, arg_name), ...]
    """
    usage = defaultdict(list)

    def walk(node: Any):
        if not isinstance(node, list) or not node:
            return

        head = node[0]

        # If this is a function call, inspect its (arg ...) children
        if isinstance(head, str):
            func_name = head
            for child in node[1:]:
                if (
                    isinstance(child, list)
                    and len(child) == 3
                    and child[0] == "arg"
                ):
                    arg_name = child[1]
                    value = child[2]
                    if isinstance(value, str) and value.startswith("#"):
                        usage[value].append((func_name, arg_name))

        # Recurse into all children
        for child in node:
            walk(child)

    walk(expr)
    return dict(usage)


def lookup_arg_type(function_name: str, arg_name: str) -> str:
    """
    Find the type of arg_name inside the schema for function_name.
    Falls back to "string" if not found.
    """
    for schema in FUNCTION_SCHEMAS:
        if schema.get("name") == function_name:
            props = schema.get("parameters", {}).get("properties", {})
            if arg_name in props:
                return props[arg_name].get("type", "string")
    return "string"


def build_function_schema(s_expr: str) -> Dict[str, Any]:
    """
    Build a new function schema from an abstraction S-expression.
    """
    parsed = parse_sexpr(s_expr)
    var_usage = collect_var_usage(parsed)

    properties = {}
    required = []

    # Keep variable-to-parameter mapping stable
    used_param_names = set()

    for var_name in sorted(var_usage.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else x):
        usages = var_usage[var_name]
        if not usages:
            continue

        func_name, arg_name = usages[0]
        arg_type = lookup_arg_type(func_name, arg_name)

        # Use the first seen argument name as the parameter name
        param_name = arg_name

        # Avoid collisions if two different vars map to the same arg name
        if param_name in used_param_names:
            base = param_name
            i = 2
            while f"{base}_{i}" in used_param_names:
                i += 1
            param_name = f"{base}_{i}"

        used_param_names.add(param_name)

        properties[param_name] = {
            "type": arg_type,
            "description": ""
        }
        required.append(param_name)

    return {
        "name": "",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required
        }
    }


def fill_args(actions: Iterable[Tuple[str, str]]) -> Iterable[Tuple[str, str]]:
    """
    For each (function_name, json_string) in actions, fill in any missing arguments
    with their default values from FUNCTION_SCHEMAS. Return a new list of (function_name, json_string).
    """
    filled_actions = []
    for func_name, json_args in actions:
        args = json.loads(json_args)
        if not isinstance(args, dict):
            raise ValueError(f"Expected JSON object for function arguments, got: {args!r}")

        for schema in FUNCTION_SCHEMAS:
            if schema["name"] != func_name:
                continue

            properties = schema["parameters"]["properties"]
            required = schema["parameters"]["required"]

            # Fill in defaults for missing optional args
            filled_args = {}
            for arg_name, arg_schema in properties.items():
                if arg_name in args:
                    filled_args[arg_name] = args[arg_name]
                elif "default" in arg_schema:
                    filled_args[arg_name] = arg_schema["default"]
                elif arg_name in required:
                    raise ValueError(f"Missing required argument '{arg_name}' for function '{func_name}'")

            filled_json_args = json.dumps(filled_args)
            filled_actions.append((func_name, filled_json_args))

    return filled_actions


def sanitize_symbol(value: str) -> str:
    """
    Convert a human-readable string into a lambda-calculus-friendly symbol.
    Example: "Banana 1" -> "Banana_1"
    """
    value = value.strip()
    value = re.sub(r"\s", "_", value)
    return value


def render_value(value: Any) -> str:
    """
    Recursively render a Python/JSON value into an S-expression-friendly form.
    """
    if value is None:
        return "nil"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return sanitize_symbol(value)
    if isinstance(value, list):
        return "(list " + " ".join(render_value(v) for v in value) + ")"
    if isinstance(value, dict):
        # Render as flat key-value pairs:
        # {"name": "Banana 1", "count": 2}
        # -> (arg name Banana_1) (arg count 2)
        parts = [f"(arg {sanitize_symbol(k)} {render_value(v)})" for k, v in value.items()]
        return " ".join(parts) if parts else "nil"

    return sanitize_symbol(value)


def action_to_expr(action: Tuple[str, str]) -> str:
    """
    Convert one (function_name, json_string) pair into an S-expression.
    Example:
        ("move_to_object", '{"name": "Banana 1"}')
        -> (move_to_object (arg name Banana_1))
    """
    func_name, json_args = action
    args = json.loads(json_args)

    if not isinstance(args, dict):
        raise ValueError(f"Expected JSON object for function arguments, got: {args!r}")

    rendered_args = [f"(arg {sanitize_symbol(k)} {render_value(v)})" for k, v in args.items()]
    if rendered_args:
        return f"({sanitize_symbol(func_name)} " + " ".join(rendered_args) + ")"
    return f"({sanitize_symbol(func_name)})"


def nest_then(expressions: List[str]) -> str:
    """
    Build a right-nested then chain:
        [a, b, c] -> (then a (then b c))
    """
    if not expressions:
        return "()"
    if len(expressions) == 1:
        return expressions[0]

    expr = expressions[-1]
    for prev in reversed(expressions[:-1]):
        expr = f"(then {prev} {expr})"
    return expr


def sequence_to_expr(actions: Iterable[Tuple[str, str]]) -> str:
    """
    Convert a sequence of actions into a nested then-expression.
    """
    filled_actions = fill_args(actions)

    exprs = [action_to_expr(action) for action in filled_actions]
    expr = nest_then(exprs)

    lambda_traces.append(expr)
    return expr


def learn_macros(traces):
    if len(lambda_traces) < 3:
        return [], []

    res = compress(lambda_traces, iterations=1, max_arity=5)

    schemas = []
    for abstraction in res.abstractions:
        document_prompt = f"Abstraction: {abstraction}\n\nTraces that use this abstraction:\n"
        
        for i, trace in enumerate(res.rewritten):
            if abstraction.name in trace:
                document_prompt += f"Prompt: {traces[i]['prompt']}\nProgram: {abstraction.body}\n"

        schema = build_function_schema(abstraction.body)
        document_prompt += f"Schema:\n{json.dumps(schema, indent=4)}"
        schema = json.loads(document_model.generate(document_prompt, memory=False))
        schemas.append(schema)

    return res.abstractions, schemas
