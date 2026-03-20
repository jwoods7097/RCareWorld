import json
from rcg.prompt import SYSTEM_PROMPT_DESCRIBE, SYSTEM_PROMPT_NAME, FUNCTION_SCHEMAS
from rcg.llm import OpenAILLM
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

existing_macros = []

@dataclass(frozen=True)
class Occurrence:
    call_index: int
    func_name: str
    arg_name: str


def _normalize_traces(traces):
    """
    Accepts either:
      1) [ [call, call, ...], [call, call, ...] ]
      2) [ {"code": [call, call, ...]}, {"code": [call, call, ...]} ]
    """
    if traces and isinstance(traces[0], dict) and "code" in traces[0]:
        return [t["code"] for t in traces]
    return traces


def _fresh_name(base: str, used: set) -> str:
    base = base or "arg"
    name = base
    i = 2
    while name in used:
        name = f"{base}_{i}"
        i += 1
    used.add(name)
    return name


def infer_minimal_template(traces):
    """
    Returns:
      {
        "parameters": ["name", ...],
        "template": [
          {
            "name": "move_to_object",
            "args": {
              "name": {"kind": "var", "name": "name"}  # or {"kind": "const", "value": ...}
            }
          },
          ...
        ]
      }
    """
    traces = _normalize_traces(traces)
    if not traces:
        return {"parameters": [], "template": []}

    # Require same call structure across traces.
    n_calls = len(traces[0])
    for t in traces:
        if len(t) != n_calls:
            raise ValueError("All traces must have the same number of calls.")

    for i in range(n_calls):
        expected_name = traces[0][i]["name"]
        expected_args = set(traces[0][i].get("args", {}).keys())
        for t in traces[1:]:
            if t[i]["name"] != expected_name:
                raise ValueError(
                    f"Call mismatch at position {i}: expected {expected_name!r}."
                )
            if set(t[i].get("args", {}).keys()) != expected_args:
                raise ValueError(
                    f"Argument-key mismatch at call {i} ({expected_name!r})."
                )

    # Build a signature for each argument occurrence:
    # signature = tuple(value in trace_1, value in trace_2, ...)
    occ_to_sig = {}
    sig_to_occurrences = defaultdict(list)

    for i in range(n_calls):
        fn = traces[0][i]["name"]
        for arg_name in traces[0][i].get("args", {}):
            sig = tuple(t[i]["args"][arg_name] for t in traces)
            occ = Occurrence(i, fn, arg_name)
            occ_to_sig[occ] = sig
            sig_to_occurrences[sig].append(occ)

    # Variable groups = signatures with more than one unique value.
    # Constant groups = signatures with exactly one unique value.
    used_names = set()
    sig_to_var = {}
    parameters = []

    for sig, occs in sig_to_occurrences.items():
        if len(set(sig)) > 1:
            base = occs[0].arg_name
            var_name = _fresh_name(base, used_names)
            sig_to_var[sig] = var_name
            parameters.append(var_name)

    template_calls = []
    for i in range(n_calls):
        fn = traces[0][i]["name"]
        args = {}
        for arg_name in traces[0][i].get("args", {}):
            occ = Occurrence(i, fn, arg_name)
            sig = occ_to_sig[occ]
            if len(set(sig)) == 1:
                args[arg_name] = {"kind": "const", "value": sig[0]}
            else:
                args[arg_name] = {"kind": "var", "name": sig_to_var[sig]}
        template_calls.append({"name": fn, "args": args})

    return {"parameters": parameters, "template": template_calls}

def render_python(template, function_name="macro"):
    """
    Turn the inferred template into a readable Python function.
    """
    params = ", ".join(template["parameters"])
    lines = [f"def {function_name}({params}):"]

    if not template["template"]:
        lines.append("    pass")
        return "\n".join(lines)

    for call in template["template"]:
        parts = []
        for arg_name, spec in call["args"].items():
            if spec["kind"] == "const":
                parts.append(f"{arg_name}={spec['value']!r}")
            else:
                parts.append(f"{arg_name}={spec['name']}")
        lines.append(f"    {call['name']}({', '.join(parts)})")

    return "\n".join(lines)

def render_function_schema(template, function_name="execute", description=""):
    """
    Convert inferred template into an OpenAI-style function schema.

    Output format:
    {
      "name": "...",
      "description": "",
      "parameters": {
        "type": "object",
        "properties": {...},
        "required": [...]
      }
    }
    """
    properties = {}
    required = []

    for param in template["parameters"]:
        properties[param] = {
            "type": "string",
            "description": ""
        }
        required.append(param)

    return {
        "name": function_name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required
        }
    }

def learn_macros(ngrams):
    name_model = OpenAILLM(system_prompt=SYSTEM_PROMPT_NAME, temperature=0.7)
    describe_model = OpenAILLM(system_prompt=SYSTEM_PROMPT_DESCRIBE, temperature=0.7)

    names = []
    macros = []
    schemas = []
    for ngram, traces in ngrams.items():
        if len(traces) < 5 or ngram in existing_macros:  # Skip ngrams with fewer than 5 traces or already existing macros
            continue

        top_tokens = {}
        for trace in traces:
            for token in trace['topk_tokens']:
                top_tokens[token] = top_tokens.get(token, 0) + 1
        topk_tokens = [k for k, _ in sorted(top_tokens.items(), key=lambda item: item[1], reverse=True)][:5]  

        template = infer_minimal_template(traces)

        name_prompt = f"Functions:{ngram}\nTop Keywords:{topk_tokens}"
        name = name_model.generate(name_prompt, memory=False)

        code = render_python(template, function_name=name)
        schema = render_function_schema(template, function_name=name)

        describe_prompt = f"Code Trace:{ngram}\nSchema:\n{json.dumps(schema, indent=4)}"
        schema = json.loads(describe_model.generate(describe_prompt, memory=False))

        names.append(name)
        macros.append(code)
        schemas.append(schema)
        existing_macros.append(ngram)

        # print(f"Top-k tokens for ngram {ngram}: {topk_tokens}")   
        # print(f"Macro for ngram {ngram}:")
        # print(template)
        # print(f"Suggested name: {name}")
        # print(schema)
        # print("\n")

    # return names, macros, schemas
    return schemas

if __name__ == "__main__":

    OpenAILLM.init_pipeline()

    with open("rcg/data/ngrams.json", "r") as f:
        ngrams = json.load(f)

    names, macros, schemas = learn_macros(ngrams)

    output = "\n".join(macros)
    output += "\n\nMACRO_MAP = {"
    for name, code in zip(names, macros):
        output += f"\n    '{name}': {name},"
    output += "\n}"

    with open("rcg/learned_macros.py", "w") as f:
        f.write(output)