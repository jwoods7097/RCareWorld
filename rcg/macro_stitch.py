# :contentReference[oaicite:0]{index=0} (rewritten)

from copy import deepcopy
from stitch_core import Abstraction, StitchException, compress, rewrite
import json
import re
from typing import Any, Iterable, List, Tuple, Dict, Optional
from collections import defaultdict
from rcg.llm import OpenAILLM
from rcg.prompt import FUNCTION_SCHEMAS, SYSTEM_PROMPT_DOCUMENT

document_model = OpenAILLM(system_prompt=SYSTEM_PROMPT_DOCUMENT, temperature=0.7)

lambda_traces = []
macro_schemas = []


# -------------------------
# S-EXPR PARSING
# -------------------------

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

    return current[0] if len(current) == 1 else current


def sexpr_to_string(expr: Any) -> str:
    if isinstance(expr, list):
        return "(" + " ".join(sexpr_to_string(x) for x in expr) + ")"
    return str(expr)


# -------------------------
# SCHEMA HELPERS
# -------------------------

def get_schema(function_name: str) -> Optional[Dict[str, Any]]:
    global macro_schemas
    for schema in FUNCTION_SCHEMAS + macro_schemas:
        if schema.get("name") == function_name:
            return schema
    return None


def get_ordered_arg_names(function_name: str) -> List[str]:
    schema = get_schema(function_name)
    if not schema:
        return []
    return list(schema["parameters"]["properties"].keys())


def lookup_arg_type(function_name: str, arg_name: str) -> str:
    schema = get_schema(function_name)
    if schema:
        props = schema["parameters"]["properties"]
        if arg_name in props:
            return props[arg_name].get("type", "string")
    return "string"


# -------------------------
# RENDERING
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
    return sanitize_symbol(str(value))


def fill_args(actions: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
    filled = []
    for func_name, json_args in actions:
        args = json.loads(json_args)
        schema = get_schema(func_name)

        props = schema["parameters"]["properties"]
        required = set(schema["parameters"].get("required", []))

        filled_args = {}
        for k, v in props.items():
            if k in args:
                filled_args[k] = args[k]
            elif "default" in v:
                filled_args[k] = v["default"]
            elif k in required:
                raise ValueError(f"Missing required argument '{k}'")

        filled.append((func_name, json.dumps(filled_args)))

    return filled


def action_to_expr(action: Tuple[str, str]) -> str:
    func_name, json_args = action
    args = json.loads(json_args)

    ordered = get_ordered_arg_names(func_name)
    values = [render_value(args[k]) for k in ordered]

    return f"({func_name} {' '.join(values)})"


def nest_then(exprs: List[str]) -> str:
    if len(exprs) == 1:
        return exprs[0]

    expr = exprs[-1]
    for e in reversed(exprs[:-1]):
        expr = f"(then {e} {expr})"
    return expr


def sequence_to_expr(actions):
    global lambda_traces
    exprs = [action_to_expr(a) for a in fill_args(actions)]
    expr = nest_then(exprs)
    lambda_traces.append(expr)
    return expr


# -------------------------
# VARIABLE + SCHEMA LEARNING
# -------------------------

def collect_var_usage(expr: Any) -> Dict[str, List[Tuple[str, str]]]:
    usage = defaultdict(list)

    def walk(node):
        if not isinstance(node, list) or not node:
            return

        head = node[0]
        if isinstance(head, str) and head != "then":
            arg_names = get_ordered_arg_names(head)

            for i, child in enumerate(node[1:]):
                if isinstance(child, str) and child.startswith("#"):
                    name = arg_names[i] if i < len(arg_names) else f"arg_{i}"
                    usage[child].append((head, name))

        for c in node:
            walk(c)

    walk(expr)
    return dict(usage)


def build_function_schema(s_expr: str):
    parsed = parse_sexpr(s_expr)
    usage = collect_var_usage(parsed)

    props = {}
    required = []
    used = set()

    for var in sorted(usage.keys(), key=lambda x: int(x[1:])):
        func, arg = usage[var][0]
        name = arg

        if name in used:
            i = 2
            while f"{name}_{i}" in used:
                i += 1
            name = f"{name}_{i}"

        used.add(name)

        props[name] = {
            "type": lookup_arg_type(func, arg),
            "description": ""
        }
        required.append(name)

    return {
        "name": "",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": props,
            "required": required
        }
    }


# -------------------------
# EXPANSION (KEY PART)
# -------------------------

def get_formals(abs_obj):
    return [f"#{i}" for i in range(abs_obj.arity)]


def expand_expr(expr, abs_by_name, cache, env=None, stack=None, counter=None):
    if env is None:
        env = {}
    if stack is None:
        stack = []
    if counter is None:
        counter = [0]

    def fresh():
        v = f"#{counter[0]}"
        counter[0] += 1
        return v

    def make_then(items):
        if len(items) == 1:
            return items[0]
        x = items[-1]
        for i in reversed(items[:-1]):
            x = ["then", i, x]
        return x

    if isinstance(expr, str):
        return deepcopy(env[expr]) if expr in env else expr

    if not isinstance(expr, list):
        return expr

    head = expr[0]

    # abstraction call
    if isinstance(head, str) and head in abs_by_name:

        # STOP infinite recursion
        if head in stack:
            return expr

        if len(stack) > 50:
            return expr

        callee = abs_by_name[head]

        formals = get_formals(callee)
        actuals = [expand_expr(a, abs_by_name, cache, env, stack, counter) for a in expr[1:]]

        # curry handling
        if len(actuals) < len(formals):
            actuals += [fresh() for _ in range(len(formals) - len(actuals))]
        else:
            actuals = actuals[:len(formals)]

        if head not in cache:
            cache[head] = expand_expr(
                parse_sexpr(callee.body),
                abs_by_name,
                cache,
                {},
                stack + [head],
                counter
            )

        return substitute(cache[head], dict(zip(formals, actuals)))

    # recurse
    head = expand_expr(head, abs_by_name, cache, env, stack, counter)
    children = [expand_expr(c, abs_by_name, cache, env, stack, counter) for c in expr[1:]]

    if head == "then":
        items = []

        def collect(n):
            if isinstance(n, list) and n and n[0] == "then":
                for c in n[1:]:
                    collect(c)
            else:
                items.append(n)

        for c in children:
            collect(c)

        return make_then(items)

    return [head] + children


def substitute(expr, env):
    if isinstance(expr, str):
        return deepcopy(env[expr]) if expr in env else expr
    if isinstance(expr, list):
        return [substitute(x, env) for x in expr]
    return expr


def deduce_arity(expr):
    max_i = -1

    def walk(n):
        nonlocal max_i
        if isinstance(n, str):
            m = re.fullmatch(r"#(\d+)", n)
            if m:
                max_i = max(max_i, int(m.group(1)))
        elif isinstance(n, list):
            for c in n:
                walk(c)

    walk(expr)
    return max_i + 1 if max_i >= 0 else 0


def rewrite_abstractions(abstractions):
    idx = {a.name: a for a in abstractions}
    cache = {}
    out = []

    for i, a in enumerate(abstractions):
        body = expand_expr(parse_sexpr(a.body), idx, cache)
        out.append(Abstraction(f"fn_{i}", sexpr_to_string(body), deduce_arity(body)))

    return out


# -------------------------
# VALIDATION
# -------------------------

def count_primitive_calls(expr):
    global macro_schemas
    known = {s["name"] for s in FUNCTION_SCHEMAS + macro_schemas}

    def walk(n):
        if not isinstance(n, list) or not n:
            return 0

        head = n[0]

        # FIX: ensure head is a string before lookup
        count = 1 if isinstance(head, str) and head in known else 0

        return count + sum(walk(x) for x in n[1:])

    return walk(expr)


def is_valid_structure(expr):
    """
    Enforce:
    - 'then' is the only structural operator
    - function heads must be valid symbols (not #vars, not lists)
    """

    if isinstance(expr, str):
        return True

    if not isinstance(expr, list) or not expr:
        return False

    head = expr[0]

    # --- THEN ---
    if head == "then":
        # must be binary (after your normalization)
        if len(expr) != 3:
            return False
        return all(is_valid_structure(child) for child in expr[1:])

    # --- FUNCTION CALL ---
    # head must be a valid symbol
    if not isinstance(head, str):
        return False

    # reject variable as function
    if head.startswith("#"):
        return False

    # reject unknown functions (optional but recommended)
    known = {s["name"] for s in FUNCTION_SCHEMAS + macro_schemas}
    if head not in known:
        return False

    # arguments must be valid recursively
    for arg in expr[1:]:
        if isinstance(arg, list):
            # disallow nested function calls as arguments
            # (this is key to your bug)
            if arg and isinstance(arg[0], str):
                if arg[0] == "then" or arg[0] in known:
                    return False
            if not is_valid_structure(arg):
                return False

    return True


def is_valid_abstraction(s_expr):
    try:
        parsed = parse_sexpr(s_expr)
    except:
        return False

    if count_primitive_calls(parsed) < 2:
        return False

    return is_valid_structure(parsed)


# -------------------------
# DEDUPLICATION
# -------------------------

def canonicalize_vars(expr):
    mapping = {}
    c = 0

    def walk(n):
        nonlocal c
        if isinstance(n, str) and re.fullmatch(r"#\d+", n):
            if n not in mapping:
                mapping[n] = f"#{c}"
                c += 1
            return mapping[n]
        if isinstance(n, list):
            return [walk(x) for x in n]
        return n

    return walk(expr)


def abstraction_exists(new_abs, abstractions):
    target = sexpr_to_string(canonicalize_vars(parse_sexpr(new_abs.body)))
    for a in abstractions:
        if sexpr_to_string(canonicalize_vars(parse_sexpr(a.body))) == target:
            return True
    return False


# -------------------------
# CODE EXECUTION
# -------------------------

def abstraction_to_python(abs_obj, schema: dict) -> str:
    """
    Convert abstraction + schema into a documented Python function.
    """

    def flatten_then(expr):
        if not isinstance(expr, list):
            return [expr]
        if expr[0] != "then":
            return [expr]

        result = []
        for child in expr[1:]:
            result.extend(flatten_then(child))
        return result

    def expr_to_call(expr, var_map):
        head = expr[0]
        args = expr[1:]

        py_args = []
        for a in args:
            if isinstance(a, str):
                if a.startswith("#"):
                    py_args.append(var_map[a])
                elif a == "true":
                    py_args.append("True")
                elif a == "false":
                    py_args.append("False")
                elif a == "nil":
                    py_args.append("None")
                else:
                    py_args.append(a)
            else:
                py_args.append(str(a))

        return f"{head}({', '.join(py_args)})"

    # -------------------------
    # Build parameter mapping
    # -------------------------
    param_names = list(schema["parameters"]["properties"].keys())

    # map #0 -> param_names[0], etc.
    var_map = {f"#{i}": param_names[i] for i in range(abs_obj.arity)}

    # -------------------------
    # Parse + flatten
    # -------------------------
    parsed = parse_sexpr(abs_obj.body)
    calls = flatten_then(parsed)

    # -------------------------
    # Convert calls
    # -------------------------
    call_lines = [expr_to_call(c, var_map) for c in calls]

    # -------------------------
    # Build docstring
    # -------------------------
    doc_lines = []

    if schema.get("description"):
        doc_lines.append(schema["description"])
        doc_lines.append("")

    doc_lines.append("Args:")

    for name, info in schema["parameters"]["properties"].items():
        desc = info.get("description", "")
        typ = info.get("type", "Any")
        doc_lines.append(f"    {name} ({typ}): {desc}")

    docstring = "\n".join(doc_lines)

    # -------------------------
    # Build function
    # -------------------------
    indent = "    "
    body = "\n".join(indent + line for line in call_lines)

    func_name = schema.get("name", abs_obj.name)
    params = ", ".join(param_names)

    return f'''def {func_name}({params}):
{indent}"""
{indent}{docstring}
{indent}"""
{body}
'''


# -------------------------
# LEARNING LOOP
# -------------------------

def learn_macros(traces, max_macros=10):
    global lambda_traces, macro_schemas

    if len(lambda_traces) < 3:
        return [], []

    abstractions = []
    programs = deepcopy(lambda_traces)

    while len(abstractions) < max_macros:
        res = compress(programs, iterations=max_macros * 2, max_arity=5)
        print(f"\nCompress Result: {res.abstractions}")

        expanded = rewrite_abstractions(res.abstractions)
        print(f"\nRewritten Abstractions: {expanded}")

        candidates = [
            a for a in expanded
            if is_valid_abstraction(a.body)
            and not abstraction_exists(a, abstractions)
        ]

        if not candidates:
            break

        a = candidates[0]
        a.name = f"fn_{len(abstractions)}"
        abstractions.append(a)

        try:
            programs = rewrite(programs, abstractions).rewritten
        except StitchException as e:
            print(f"Rewrite failed: {e}")
            abstractions.pop()

    macro_schemas = []
    code_file = "from rcg.llm import get_info, move_to_object, grasp_object, release_object, move_to_position\n\n"
    macro_map = "\nMACRO_MAP = {\n"
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

        # Generate code for this abstraction
        code = abstraction_to_python(abstraction, schema)
        code_file += code
        macro_map += f"    \"{schema['name']}\": {schema['name']}\n"

    with open("rcg/learned_macros.py", "w", encoding="utf-8") as f:
        f.write(code_file)
        macro_map += "}"
        f.write(macro_map)

    return abstractions, macro_schemas
