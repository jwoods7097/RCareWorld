from copy import deepcopy
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json
import re

from stitch_core import Abstraction, StitchException, compress, rewrite
from rcg.llm import OpenAILLM
from rcg.robot import FUNCTION_SCHEMAS
from rcg.prompt import SYSTEM_PROMPT_DOCUMENT


document_model = OpenAILLM(system_prompt=SYSTEM_PROMPT_DOCUMENT, temperature=0.7)

lambda_traces: List[str] = []
macro_schemas: List[Dict[str, Any]] = []
learned_abstractions: List[Abstraction] = []


# -------------------------
# S-EXPRESSION UTILITIES
# -------------------------

def tokenize_sexpr(s: str) -> List[str]:
    return re.findall(r'\(|\)|"[^"\\]*(?:\\.[^"\\]*)*"|[^\s()]+', s)


def parse_sexpr(s: str) -> Any:
    tokens = tokenize_sexpr(s)
    stack: List[List[Any]] = []
    current: List[Any] = []

    for tok in tokens:
        if tok == "(":
            stack.append(current)
            new_list: List[Any] = []
            current.append(new_list)
            current = new_list
        elif tok == ")":
            if not stack:
                raise ValueError("Unbalanced closing parenthesis")
            current = stack.pop()
        else:
            if tok.startswith('"') and tok.endswith('"'):
                tok = json.loads(tok)
            current.append(tok)

    if stack:
        raise ValueError("Unbalanced opening parenthesis")

    return current[0] if len(current) == 1 else current


def sexpr_to_string(expr: Any) -> str:
    if isinstance(expr, list):
        return "(" + " ".join(sexpr_to_string(x) for x in expr) + ")"
    if expr is True:
        return "true"
    if expr is False:
        return "false"
    if expr is None:
        return "nil"
    return str(expr)


def flatten_then(expr: Any) -> List[Any]:
    if not isinstance(expr, list) or not expr or expr[0] != "then":
        return [expr]

    out: List[Any] = []
    for child in expr[1:]:
        out.extend(flatten_then(child))
    return out


def make_nested_then(items: List[Any]) -> Any:
    if not items:
        return []
    if len(items) == 1:
        return items[0]

    expr = items[-1]
    for item in reversed(items[:-1]):
        expr = ["then", item, expr]
    return expr


# -------------------------
# SCHEMA UTILITIES
# -------------------------

def get_schema(function_name: str) -> Optional[Dict[str, Any]]:
    for schema in FUNCTION_SCHEMAS + macro_schemas:
        if schema.get("name") == function_name:
            return schema
    return None


def get_ordered_arg_names(function_name: str) -> List[str]:
    schema = get_schema(function_name)
    if schema is None:
        return []
    return list(schema.get("parameters", {}).get("properties", {}).keys())


def lookup_arg_type(function_name: str, arg_name: str) -> str:
    schema = get_schema(function_name)
    if schema is None:
        return "string"
    return schema.get("parameters", {}).get("properties", {}).get(arg_name, {}).get("type", "string")


def is_known_function(function_name: str) -> bool:
    return get_schema(function_name) is not None


# -------------------------
# VALUE RENDERING / CASTING
# -------------------------

def sanitize_symbol(value: str) -> str:
    return re.sub(r"\s+", "_", value.strip())


def render_value(value: Any) -> str:
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
        parts = [f"({sanitize_symbol(str(k))} {render_value(v)})" for k, v in value.items()]
        return "(dict " + " ".join(parts) + ")" if parts else "(dict)"
    return sanitize_symbol(str(value))


def parse_atom_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if value == "true":
        return True
    if value == "false":
        return False
    if value == "nil":
        return None
    if re.fullmatch(r"-?\d+(\.\d+)?", value):
        return float(value) if "." in value else int(value)
    return value


def cast_value(value: Any, expected_type: str) -> Any:
    if value is None:
        return None
    if expected_type == "number":
        return value if isinstance(value, (int, float)) else float(value)
    if expected_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() == "true"
        return bool(value)
    if expected_type == "string":
        return str(value)
    return value


def normalize_object_names(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: normalize_object_names(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize_object_names(v) for v in obj]
    if isinstance(obj, str):
        return (
            obj.replace("Banana_1", "Banana 1")
            .replace("Banana_2", "Banana 2")
            .replace("Banana_3", "Banana 3")
        )
    return obj


# -------------------------
# TRACE CONVERSION
# -------------------------

def fill_args(actions: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
    filled: List[Tuple[str, str]] = []

    for func_name, json_args in actions:
        args = json.loads(json_args)
        if not isinstance(args, dict):
            raise ValueError(f"Expected JSON object for {func_name} arguments, got {args!r}")

        schema = get_schema(func_name)
        if schema is None:
            raise ValueError(f"No schema found for function '{func_name}'")

        props = schema["parameters"]["properties"]
        required = set(schema["parameters"].get("required", []))

        filled_args: Dict[str, Any] = {}
        for name, prop in props.items():
            if name in args:
                filled_args[name] = args[name]
            elif "default" in prop:
                filled_args[name] = prop["default"]
            elif name in required:
                raise ValueError(f"Missing required argument '{name}' for function '{func_name}'")

        filled.append((func_name, json.dumps(filled_args)))

    return filled


def action_to_expr(action: Tuple[str, str]) -> str:
    func_name, json_args = action
    args = json.loads(json_args)
    values = [render_value(args[name]) for name in get_ordered_arg_names(func_name)]
    return f"({func_name} {' '.join(values)})" if values else f"({func_name})"


def nest_then(exprs: List[str]) -> str:
    if not exprs:
        return "()"
    if len(exprs) == 1:
        return exprs[0]

    expr = exprs[-1]
    for item in reversed(exprs[:-1]):
        expr = f"(then {item} {expr})"
    return expr


def sequence_to_expr(actions: Iterable[Tuple[str, str]]) -> str:
    exprs = [action_to_expr(action) for action in fill_args(actions)]
    expr = nest_then(exprs)
    if expr.strip() != "()" and expr.strip() != "":
        lambda_traces.append(expr)
    return expr


# -------------------------
# VARIABLE / SCHEMA LEARNING
# -------------------------

def collect_var_usage(expr: Any) -> Dict[str, List[Tuple[str, str]]]:
    usage: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    stack = [expr]

    while stack:
        node = stack.pop()
        if not isinstance(node, list) or not node:
            continue

        head = node[0]
        if isinstance(head, str) and head != "then":
            arg_names = get_ordered_arg_names(head)
            for i, child in enumerate(node[1:]):
                if isinstance(child, str) and re.fullmatch(r"#\d+", child):
                    arg_name = arg_names[i] if i < len(arg_names) else f"arg_{i}"
                    usage[child].append((head, arg_name))

        stack.extend(reversed(node))

    return dict(usage)


def build_function_schema(s_expr: str) -> Dict[str, Any]:
    usage = collect_var_usage(parse_sexpr(s_expr))
    properties: Dict[str, Dict[str, str]] = {}
    required: List[str] = []
    used_names = set()

    for var in sorted(usage.keys(), key=lambda x: int(x[1:])):
        func_name, arg_name = usage[var][0]
        param_name = unique_name(arg_name, used_names)
        properties[param_name] = {
            "type": lookup_arg_type(func_name, arg_name),
            "description": "",
        }
        required.append(param_name)

    return {
        "name": "",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def unique_name(name: str, used_names: set) -> str:
    if name not in used_names:
        used_names.add(name)
        return name

    i = 2
    while f"{name}_{i}" in used_names:
        i += 1
    out = f"{name}_{i}"
    used_names.add(out)
    return out


# -------------------------
# ABSTRACTION EXPANSION
# -------------------------

def get_formals(abs_obj: Abstraction) -> List[str]:
    return [f"#{i}" for i in range(abs_obj.arity)]


def fresh_var(counter: List[int]) -> str:
    value = f"#{counter[0]}"
    counter[0] += 1
    return value


def substitute(expr: Any, env: Dict[str, Any]) -> Any:
    if isinstance(expr, str):
        return deepcopy(env[expr]) if expr in env else expr
    if isinstance(expr, list):
        return [substitute(x, env) for x in expr]
    return expr


def expand_expr(
    expr: Any,
    abs_by_name: Dict[str, Abstraction],
    cache: Dict[str, Any],
    env: Optional[Dict[str, Any]] = None,
    stack: Optional[List[str]] = None,
    counter: Optional[List[int]] = None,
) -> Any:
    if env is None:
        env = {}
    if stack is None:
        stack = []
    if counter is None:
        counter = [0]

    if isinstance(expr, str):
        return deepcopy(env[expr]) if expr in env else expr
    if not isinstance(expr, list) or not expr:
        return expr

    head = expr[0]

    if isinstance(head, str) and head in abs_by_name:
        if head in stack or len(stack) > 50:
            return expr

        callee = abs_by_name[head]
        formals = get_formals(callee)
        actuals = [expand_expr(arg, abs_by_name, cache, env, stack, counter) for arg in expr[1:]]

        if len(actuals) < len(formals):
            actuals += [fresh_var(counter) for _ in range(len(formals) - len(actuals))]
        elif len(actuals) > len(formals):
            actuals = actuals[:len(formals)]

        if head not in cache:
            cache[head] = expand_expr(
                parse_sexpr(callee.body),
                abs_by_name,
                cache,
                env={},
                stack=stack + [head],
                counter=counter,
            )

        return substitute(cache[head], dict(zip(formals, actuals)))

    expanded_head = expand_expr(head, abs_by_name, cache, env, stack, counter)
    expanded_children = [expand_expr(child, abs_by_name, cache, env, stack, counter) for child in expr[1:]]

    if expanded_head == "then":
        return make_nested_then(flatten_then(["then"] + expanded_children))

    return [expanded_head] + expanded_children


def deduce_arity(expr: Any) -> int:
    max_index = -1
    stack = [expr]

    while stack:
        node = stack.pop()
        if isinstance(node, str):
            match = re.fullmatch(r"#(\d+)", node)
            if match:
                max_index = max(max_index, int(match.group(1)))
        elif isinstance(node, list):
            stack.extend(node)

    return max_index + 1 if max_index >= 0 else 0


def build_abstraction_index(abstractions: Iterable[Abstraction]) -> Dict[str, Abstraction]:
    return {a.name: a for a in abstractions}


def build_macro_expansion_index(
    abstractions: Iterable[Abstraction],
    schemas: Iterable[Dict[str, Any]],
) -> Dict[str, Abstraction]:
    index: Dict[str, Abstraction] = {}
    for abs_obj, schema in zip(abstractions, schemas):
        if abs_obj.name:
            index[abs_obj.name] = abs_obj
        schema_name = schema.get("name")
        if schema_name:
            index[schema_name] = abs_obj
    return index


def rewrite_abstractions(abstractions: Iterable[Abstraction]) -> List[Abstraction]:
    abstractions = list(abstractions)
    abs_by_name = build_abstraction_index(abstractions)
    cache: Dict[str, Any] = {}
    rewritten: List[Abstraction] = []

    for i, abs_obj in enumerate(abstractions):
        body = expand_expr(parse_sexpr(abs_obj.body), abs_by_name, cache)
        rewritten.append(Abstraction(f"fn_{i}", sexpr_to_string(body), deduce_arity(body)))

    return rewritten


# -------------------------
# VALIDATION / DEDUPLICATION
# -------------------------

def count_primitive_calls(expr: Any) -> int:
    if not isinstance(expr, list) or not expr:
        return 0

    head = expr[0]
    count = 1 if isinstance(head, str) and is_known_function(head) else 0
    return count + sum(count_primitive_calls(child) for child in expr[1:])


def is_literal_structure(expr: Any) -> bool:
    """
    Validate literal-only list structures such as (list ...) or (dict ...).

    This is used for primitive argument values. It deliberately rejects:
    - then expressions
    - primitive or macro function calls
    - variables in head position, e.g. (#0 ...)
    - non-symbol heads
    """
    if not isinstance(expr, list):
        return True
    if not expr:
        return True

    head = expr[0]
    if not isinstance(head, str):
        return False
    if head == "then" or head.startswith("#") or is_known_function(head):
        return False

    return all(is_value_expr(child) for child in expr[1:])


def is_value_expr(expr: Any) -> bool:
    """
    Validate something used as a primitive argument value.

    Variables like #0 are allowed here, but full program expressions are not.
    This prevents invalid abstractions such as:
        (then (move_to_position ...) #0)
    and function-valued arguments such as:
        (move_to_position (grasp_object ...) ...)
    """
    if isinstance(expr, str):
        return True
    if not isinstance(expr, list):
        return True
    return is_literal_structure(expr)


def is_valid_structure(expr: Any) -> bool:
    """
    Validate a full program expression.

    At the program level, only these shapes are valid:
    - (then <program> <program>)
    - (<known_function> <value> ...)

    Bare variables such as #0 are invalid as standalone program steps,
    and variables/functions are invalid in function position.
    """
    if isinstance(expr, str):
        return False
    if not isinstance(expr, list) or not expr:
        return False

    head = expr[0]

    if head == "then":
        return len(expr) == 3 and all(is_valid_structure(child) for child in expr[1:])

    if not isinstance(head, str):
        return False
    if head.startswith("#") or not is_known_function(head):
        return False

    return all(is_value_expr(arg) for arg in expr[1:])


def is_valid_abstraction(s_expr: str) -> bool:
    try:
        parsed = parse_sexpr(s_expr)
    except Exception:
        return False
    return count_primitive_calls(parsed) >= 2 and is_valid_structure(parsed)


def canonicalize_vars(expr: Any) -> Any:
    mapping: Dict[str, str] = {}
    counter = 0
    stack: List[Tuple[Any, Optional[List[Any]], Optional[int]]] = [(expr, None, None)]
    root = None

    while stack:
        node, parent, index = stack.pop()
        if isinstance(node, str) and re.fullmatch(r"#\d+", node):
            if node not in mapping:
                mapping[node] = f"#{counter}"
                counter += 1
            replacement: Any = mapping[node]
        elif isinstance(node, list):
            replacement = [None] * len(node)
            for i in reversed(range(len(node))):
                stack.append((node[i], replacement, i))
        else:
            replacement = node

        if parent is None:
            root = replacement
        else:
            parent[index] = replacement

    return root


def canonicalize_abstraction(abs_obj: Abstraction) -> str:
    return sexpr_to_string(canonicalize_vars(parse_sexpr(abs_obj.body)))


def abstraction_exists(new_abs: Abstraction, abstractions: Iterable[Abstraction]) -> bool:
    target = canonicalize_abstraction(new_abs)
    return any(canonicalize_abstraction(abs_obj) == target for abs_obj in abstractions)


# -------------------------
# PYTHON CODE GENERATION
# -------------------------

def sexpr_arg_to_python(arg: Any, var_map: Dict[str, str]) -> str:
    if isinstance(arg, str):
        if re.fullmatch(r"#\d+", arg):
            return var_map.get(arg, f"arg{arg[1:]}")
        if arg == "true":
            return "True"
        if arg == "false":
            return "False"
        if arg == "nil":
            return "None"
        if re.fullmatch(r"-?\d+(\.\d+)?", arg):
            return arg
        return repr(normalize_object_names(arg))
    return str(arg)


def sexpr_call_to_python(expr: Any, var_map: Dict[str, str]) -> str:
    head = expr[0]
    args = ", ".join(sexpr_arg_to_python(arg, var_map) for arg in expr[1:])
    return f"{head}({args})"


def build_python_docstring(schema: Dict[str, Any]) -> str:
    lines: List[str] = []
    description = schema.get("description")
    if description:
        lines.extend([description, ""])

    lines.append("Args:")
    for name, info in schema["parameters"]["properties"].items():
        lines.append(f"    {name} ({info.get('type', 'Any')}): {info.get('description', '')}")
    return "\n".join(lines)


def abstraction_to_python(abs_obj: Abstraction, schema: Dict[str, Any]) -> str:
    body_expr = parse_sexpr(abs_obj.body)

    param_names = list(schema["parameters"]["properties"].keys())
    required_arity = max(abs_obj.arity, deduce_arity(body_expr))

    while len(param_names) < required_arity:
        base = f"arg{len(param_names)}"
        name = base
        suffix = 2
        while name in param_names:
            name = f"{base}_{suffix}"
            suffix += 1
        param_names.append(name)

    var_map = {f"#{i}": param_names[i] for i in range(required_arity)}
    calls = flatten_then(body_expr)
    call_lines = [sexpr_call_to_python(call, var_map) for call in calls]

    indent = "    "
    func_name = schema.get("name", abs_obj.name)
    params = ", ".join(param_names)
    docstring = build_python_docstring(schema)
    body = "\n".join(indent + line for line in call_lines) or indent + "pass"

    return f'''def {func_name}({params}):\n{indent}"""\n{indent}{docstring}\n{indent}"""\n{body}\n'''


# -------------------------
# JSON MACRO REWRITING
# -------------------------

def parse_json_calls(json_text: str) -> List[Dict[str, Any]]:
    text = json_text.strip()
    if not text:
        return []

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data
    except Exception:
        pass

    return [json.loads(line) for line in text.splitlines() if line.strip()]


def json_call_to_sexpr(call: Dict[str, Any]) -> List[Any]:
    func = call["function"]
    args = call.get("args", {})
    schema = get_schema(func)
    if schema is None:
        raise ValueError(f"No schema found for function '{func}'")

    values: List[Any] = []
    props = schema["parameters"]["properties"]
    required = set(schema["parameters"].get("required", []))

    for name, prop in props.items():
        if name in args:
            value = args[name]
        elif "default" in prop:
            value = prop["default"]
        elif name in required:
            raise ValueError(f"Missing required argument '{name}' for function '{func}'")
        else:
            value = None
        values.append(render_value(value))

    return [func] + values


def sexpr_call_to_json(expr: List[Any]) -> Dict[str, Any]:
    func = expr[0]
    schema = get_schema(func)
    if schema is None:
        raise ValueError(f"No schema found for function '{func}'")

    props = schema["parameters"]["properties"]
    arg_names = list(props.keys())
    arg_dict: Dict[str, Any] = {}

    for i, raw_value in enumerate(expr[1:]):
        if i >= len(arg_names):
            break
        name = arg_names[i]
        expected_type = props[name].get("type", "string")
        arg_dict[name] = cast_value(parse_atom_value(raw_value), expected_type)

    return {"function": func, "args": arg_dict}


def rewrite_json_calls(json_str: str) -> str:
    if len(macro_schemas) != len(learned_abstractions):
        raise ValueError(
            "macro_schemas and learned_abstractions must have the same length "
            f"(got {len(macro_schemas)} and {len(learned_abstractions)})"
        )

    abs_by_name = build_macro_expansion_index(learned_abstractions, macro_schemas)
    cache: Dict[str, Any] = {}
    output_calls: List[Dict[str, Any]] = []

    for call in parse_json_calls(json_str):
        expanded = expand_expr(
            json_call_to_sexpr(call),
            abs_by_name,
            cache,
            env={},
            stack=[],
            counter=[0],
        )

        for expr in flatten_then(expanded):
            if not isinstance(expr, list) or not expr or not isinstance(expr[0], str):
                continue
            if expr[0] in abs_by_name:
                continue
            if not is_known_function(expr[0]):
                continue
            output_calls.append(normalize_object_names(sexpr_call_to_json(expr)))

    return "\n".join(json.dumps(call) for call in output_calls)


# -------------------------
# LEARNING LOOP
# -------------------------

def build_document_prompt(abstraction: Abstraction, traces: List[Dict[str, Any]], programs: List[str]) -> str:
    prompt = f"Abstraction: {abstraction}\n\nTraces that use this abstraction:\n"
    for i, trace in enumerate(programs):
        if abstraction.name in trace:
            prompt += f"Prompt: {traces[i]['prompt']}\nProgram: {programs[i]}\n"
    return prompt


def write_learned_macros_file(abstractions: List[Abstraction], schemas: List[Dict[str, Any]]) -> None:
    code_file = "from rcg.robot import get_info, move_to_object, grasp_object, release_object, move_to_position\n\n"
    macro_map = "\nMACRO_MAP = {\n"

    for abstraction, schema in zip(abstractions, schemas):
        code_file += abstraction_to_python(abstraction, schema)
        macro_map += f"    \"{schema['name']}\": {schema['name']},\n"

    macro_map += "}"
    with open("rcg/learned_macros.py", "w", encoding="utf-8") as f:
        f.write(code_file)
        f.write(macro_map)


def learn_macros(traces: List[Dict[str, Any]], max_macros: int = 10) -> Tuple[List[Abstraction], List[Dict[str, Any]]]:
    global macro_schemas, learned_abstractions

    learned_abstractions = []
    macro_schemas = []
    programs = deepcopy(lambda_traces)

    # During the loop, macro_schemas is populated with a structural placeholder
    # for each accepted abstraction so that is_known_function recognises macro
    # names when validating candidates in subsequent iterations.  The
    # placeholders are replaced with real LLM-generated schemas after the loop.
    while len(learned_abstractions) < max_macros:
        res = compress(programs, iterations=max_macros * 2, max_arity=5)
        expanded = rewrite_abstractions(res.abstractions)
        candidates = [
            abstraction
            for abstraction in expanded
            if is_valid_abstraction(abstraction.body)
            and not abstraction_exists(abstraction, learned_abstractions)
        ]

        if not candidates:
            break

        candidate = candidates[0]
        candidate.name = f"fn_{len(learned_abstractions)}"
        learned_abstractions.append(candidate)

        # Register a minimal placeholder schema immediately so that
        # is_known_function returns True for this name in future iterations.
        placeholder = build_function_schema(candidate.body)
        placeholder["name"] = candidate.name
        macro_schemas.append(placeholder)

        try:
            programs = rewrite(programs, learned_abstractions).rewritten
        except StitchException as exc:
            print(f"Rewrite failed: {exc}")
            learned_abstractions.pop()
            macro_schemas.pop()

    # Replace placeholders with LLM-generated schemas.  Skip the document
    # model for any body we have already documented in this run.
    _schema_cache: Dict[str, Dict[str, Any]] = {}
    for i, abstraction in enumerate(learned_abstractions):
        if abstraction.body in _schema_cache:
            macro_schemas[i] = _schema_cache[abstraction.body]
            continue
        schema = build_function_schema(abstraction.body)
        prompt = build_document_prompt(abstraction, traces, programs)
        prompt += f"Schema:\n{json.dumps(schema, indent=4)}"
        result = json.loads(document_model.generate(prompt, memory=False))
        _schema_cache[abstraction.body] = result
        macro_schemas[i] = result

    write_learned_macros_file(learned_abstractions, macro_schemas)
    return learned_abstractions, macro_schemas