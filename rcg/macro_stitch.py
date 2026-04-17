from copy import deepcopy
from stitch_core import compress, rewrite
import json
import re
from typing import Any, Iterable, List, Tuple, Dict, Optional
from collections import defaultdict
from rcg.llm import OpenAILLM
from rcg.prompt import FUNCTION_SCHEMAS, SYSTEM_PROMPT_DOCUMENT

document_model = OpenAILLM(system_prompt=SYSTEM_PROMPT_DOCUMENT, temperature=0.7)

lambda_traces = []
macro_schemas = []


def tokenize_sexpr(s: str) -> List[str]:
    return re.findall(r'\(|\)|"[^"\\]*(?:\\.[^"\\]*)*"|[^\s()]+', s)


def parse_sexpr(s: str) -> Any:
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


def get_schema(function_name: str) -> Optional[Dict[str, Any]]:
    for schema in FUNCTION_SCHEMAS + macro_schemas:
        if schema.get("name") == function_name:
            return schema
    return None


def get_ordered_arg_names(function_name: str) -> List[str]:
    schema = get_schema(function_name)
    if not schema:
        return []
    props = schema.get("parameters", {}).get("properties", {})
    return list(props.keys())


def get_required_args(function_name: str) -> List[str]:
    schema = get_schema(function_name)
    if not schema:
        return []
    return list(schema.get("parameters", {}).get("required", []))


def lookup_arg_type(function_name: str, arg_name: str) -> str:
    schema = get_schema(function_name)
    if schema:
        props = schema.get("parameters", {}).get("properties", {})
        if arg_name in props:
            return props[arg_name].get("type", "string")
    return "string"


def sanitize_symbol(value: str) -> str:
    value = value.strip()
    value = re.sub(r"\s+", "_", value)
    return value


def render_value(value: Any) -> str:
    """
    Render a Python/JSON value as a pure S-expression value.
    No (arg ...) wrappers.
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
        parts = []
        for k, v in value.items():
            parts.append(f"({sanitize_symbol(k)} {render_value(v)})")
        return "(dict " + " ".join(parts) + ")" if parts else "(dict)"
    return sanitize_symbol(str(value))


def fill_args(actions: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    For each (function_name, json_string), fill in missing args using defaults
    and emit args in the order specified by the schema.
    """
    filled_actions = []

    for func_name, json_args in actions:
        args = json.loads(json_args)
        if not isinstance(args, dict):
            raise ValueError(f"Expected JSON object for function arguments, got: {args!r}")

        schema = get_schema(func_name)
        if schema is None:
            raise ValueError(f"Unknown function schema: {func_name}")

        properties = schema["parameters"]["properties"]
        required = set(schema["parameters"].get("required", []))

        filled_args = {}
        for arg_name, arg_schema in properties.items():
            if arg_name in args:
                filled_args[arg_name] = args[arg_name]
            elif "default" in arg_schema:
                filled_args[arg_name] = arg_schema["default"]
            elif arg_name in required:
                raise ValueError(f"Missing required argument '{arg_name}' for function '{func_name}'")

        filled_actions.append((func_name, json.dumps(filled_args)))

    return filled_actions


def action_to_expr(action: Tuple[str, str]) -> str:
    """
    Convert one (function_name, json_string) pair into:
        (func arg1 arg2 arg3 ...)
    using the argument order from FUNCTION_SCHEMAS.
    """
    func_name, json_args = action
    args = json.loads(json_args)

    if not isinstance(args, dict):
        raise ValueError(f"Expected JSON object for function arguments, got: {args!r}")

    schema = get_schema(func_name)
    if schema is None:
        raise ValueError(f"Unknown function schema: {func_name}")

    ordered_arg_names = list(schema["parameters"]["properties"].keys())
    rendered_args = [render_value(args[arg_name]) for arg_name in ordered_arg_names]

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
    filled_actions = fill_args(actions)
    exprs = [action_to_expr(action) for action in filled_actions]
    expr = nest_then(exprs)

    lambda_traces.append(expr)
    return expr


def collect_var_usage(expr: Any) -> Dict[str, List[Tuple[str, str]]]:
    """
    Collect:
        #0 -> [(function_name, arg_name), ...]
    assuming direct positional args:
        (foo a #0 c)
    """
    usage = defaultdict(list)

    def walk(node: Any):
        if not isinstance(node, list) or not node:
            return

        head = node[0]
        if isinstance(head, str) and head != "then":
            func_name = head
            arg_names = get_ordered_arg_names(func_name)

            for i, child in enumerate(node[1:]):
                if isinstance(child, str) and child.startswith("#"):
                    arg_name = arg_names[i] if i < len(arg_names) else f"arg_{i}"
                    usage[child].append((func_name, arg_name))

        for child in node:
            walk(child)

    walk(expr)
    return dict(usage)


def build_function_schema(s_expr: str) -> Dict[str, Any]:
    """
    Build a new function schema from an abstraction S-expression that uses
    direct positional args instead of (arg name value).
    """
    parsed = parse_sexpr(s_expr)
    var_usage = collect_var_usage(parsed)

    properties = {}
    required = []
    used_param_names = set()

    def var_sort_key(x: str):
        return int(x[1:]) if x.startswith("#") and x[1:].isdigit() else x

    for var_name in sorted(var_usage.keys(), key=var_sort_key):
        usages = var_usage[var_name]
        if not usages:
            continue

        func_name, arg_name = usages[0]
        arg_type = lookup_arg_type(func_name, arg_name)

        param_name = arg_name
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


def count_primitive_calls(expr: Any) -> int:
    known_funcs = {schema["name"] for schema in FUNCTION_SCHEMAS + macro_schemas}

    def walk(node: Any) -> int:
        if not isinstance(node, list) or not node:
            return 0
        head = node[0]
        total = 0
        if isinstance(head, str) and head in known_funcs and head != "then":
            total += 1
        for child in node[1:]:
            total += walk(child)
        return total

    return walk(expr)


def validate_hash_positions(expr: Any, in_value_position: bool = False) -> bool:
    """
    Allow #vars only as direct argument values of primitive calls.
    Since args are now positional, any child of a primitive call may be a #var.
    """
    known_funcs = {schema["name"] for schema in FUNCTION_SCHEMAS + macro_schemas}

    if isinstance(expr, str):
        if expr.startswith("#"):
            return in_value_position
        return True

    if not isinstance(expr, list) or not expr:
        return True

    head = expr[0]

    if head == "then":
        if len(expr) < 3:
            return False
        return all(validate_hash_positions(child, False) for child in expr[1:])

    if isinstance(head, str) and head in known_funcs:
        for child in expr[1:]:
            if isinstance(child, str):
                if child.startswith("#"):
                    continue
            elif isinstance(child, list):
                # Allow literal list/dict-like structures, but do not allow nested
                # primitive calls or then inside argument values.
                if child and isinstance(child[0], str) and (child[0] == "then" or child[0] in known_funcs):
                    return False
                if not validate_hash_positions(child, True):
                    return False
            else:
                continue
        return True

    # Any other top-level list form is invalid as a plan expression
    return False


def is_valid_abstraction(s_expr: str) -> bool:
    try:
        parsed = parse_sexpr(s_expr)
    except Exception:
        return False

    if count_primitive_calls(parsed) < 2:
        return False

    if not contains_then(parsed):
        return False

    if not validate_hash_positions(parsed):
        return False

    return True


def contains_then(expr: Any) -> bool:
    if isinstance(expr, list):
        if expr and expr[0] == "then":
            return True
        return any(contains_then(child) for child in expr)
    return False


def learn_macros(traces, max_macros=10) -> Tuple[List[str], List[Dict[str, Any]]]:
    global lambda_traces, macro_schemas

    # Make sure we have a bit of a library before looking for abstractions
    if len(lambda_traces) < 3:
        return [], []

    # Manual abstraction learning
    abstractions = []
    programs = deepcopy(lambda_traces)
    while len(abstractions) < max_macros:
        # Generate candidate abstractions using Stitch's compression algorithm
        res = compress(
            programs,
            iterations=max_macros*2,
            max_arity=5,
            tasks=[trace["prompt"] for trace in traces],
            allow_single_task=False
        )

        # Filter out invalid abstractions that don't meet our criteria
        valid_macros = [abs for abs in res.abstractions if is_valid_abstraction(abs.body)]
        if not valid_macros:
            break

        # Take first valid abstraction, add to library, and rewrite programs to use it for further abstraction discovery
        abstractions.append(valid_macros[0])
        programs = rewrite(programs, abstractions).rewritten

    # Schema generation
    macro_schemas = []
    for abstraction in abstractions:
        document_prompt = f"Abstraction: {abstraction}\n\nTraces that use this abstraction:\n"

        # Collect traces that use this abstraction to provide context for documentation generation
        for i, trace in enumerate(programs):
            if abstraction.name in trace:
                document_prompt += f"Prompt: {traces[i]['prompt']}\nProgram: {abstraction.body}\n"

        # Generate schema for this abstraction
        schema = build_function_schema(abstraction.body)
        document_prompt += f"Schema:\n{json.dumps(schema, indent=4)}"
        schema = json.loads(document_model.generate(document_prompt, memory=False))
        macro_schemas.append(schema)

    return abstractions, macro_schemas