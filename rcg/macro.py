import json
from rcg.prompt import SYSTEM_PROMPT_DESCRIBE, SYSTEM_PROMPT_NAME, FUNCTION_SCHEMAS
from rcg.llm import OpenAILLM
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

existing_macros = []
MISSING = object()


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


def _infer_json_schema_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "string"


def infer_minimal_template(traces):
    """
    Infer a minimal macro template from traces.

    Returns:
      {
        "parameters": {
          "name": {
            "kind": "var" | "const",
            "required": bool,
            "value": ...   # only for const
          },
          ...
        },
        "template": [
          {
            "name": "move_to_object",
            "args": {
              "name": {"kind": "var", "name": "name"},
              "speed": {"kind": "const", "value": 0.5}
            }
          },
          ...
        ]
      }

    Notes:
      - Missing args are allowed.
      - If an arg is missing in some traces, it is marked optional.
      - If all present values are identical, it is marked const.
      - If values differ across traces, it is marked var.
      - Args with the same value signature across call sites are tied together.
    """
    traces = _normalize_traces(traces)
    if not traces:
        return {"parameters": {}, "template": []}

    n_calls = len(traces[0])
    for t in traces:
        if len(t) != n_calls:
            raise ValueError("All traces must have the same number of calls.")

    # Verify call order/function names are aligned.
    for i in range(n_calls):
        expected_name = traces[0][i]["name"]
        for t in traces[1:]:
            if t[i]["name"] != expected_name:
                raise ValueError(
                    f"Call mismatch at position {i}: expected {expected_name!r}."
                )

    # Build signatures for each argument occurrence.
    # Signature includes presence info via MISSING.
    occ_to_sig = {}
    sig_to_occs = defaultdict(list)

    for i in range(n_calls):
        func_name = traces[0][i]["name"]
        all_arg_names = set()
        for t in traces:
            all_arg_names.update(t[i].get("args", {}).keys())

        for arg_name in all_arg_names:
            sig = tuple(
                t[i]["args"].get(arg_name, MISSING)
                for t in traces
            )
            occ = Occurrence(i, func_name, arg_name)
            occ_to_sig[occ] = sig
            sig_to_occs[sig].append(occ)

    used_names = set()
    parameters = {}

    # Assign one parameter name per varying signature group.
    for sig, occs in sig_to_occs.items():
        present_values = [v for v in sig if v is not MISSING]
        if not present_values:
            continue

        unique_present_values = set(present_values)
        is_required = all(v is not MISSING for v in sig)
        is_constant = len(unique_present_values) == 1

        if is_constant:
            # Constant values are hard-coded in the template, but if the arg
            # is missing in some traces, it is still optional at the schema level.
            value = present_values[0]
            # Use one shared name only if it is not a fully hard-coded arg.
            # For required constants we do not need a parameter entry.
            if not is_required:
                param_name = _fresh_name(occs[0].arg_name, used_names)
                parameters[param_name] = {
                    "kind": "const",
                    "required": False,
                    "value": value,
                }
            continue

        # Variable value across traces: this is a real parameter.
        param_name = _fresh_name(occs[0].arg_name, used_names)
        parameters[param_name] = {
            "kind": "var",
            "required": is_required,
        }

    # Build the call template.
    template_calls = []
    sig_to_param = {}

    for param_name, spec in parameters.items():
        # Map every signature group to the chosen parameter name.
        # Need to find the signature group for this parameter.
        pass

    # Reconstruct the mapping from signature to parameter name.
    sig_to_param = {}
    for sig, occs in sig_to_occs.items():
        present_values = [v for v in sig if v is not MISSING]
        if not present_values:
            continue

        unique_present_values = set(present_values)
        is_required = all(v is not MISSING for v in sig)
        is_constant = len(unique_present_values) == 1

        if is_constant:
            if not is_required:
                # Optional constant param was given a schema entry.
                # Recreate its generated name from the occurrence group.
                # We rely on the first occurrence's arg name as the base.
                base = occs[0].arg_name
                # Find the matching parameter by kind/value/base.
                for pname, pspec in parameters.items():
                    if pspec["kind"] == "const" and pspec["value"] == present_values[0]:
                        sig_to_param[sig] = pname
                        break
            continue

        # Find the matching variable param.
        for pname, pspec in parameters.items():
            if pspec["kind"] == "var":
                # Match by requiredness and signature group order via first occurrence.
                # Since we created names in signature order, this is stable enough for
                # a single-pass template build.
                if pname not in sig_to_param.values():
                    sig_to_param[sig] = pname
                    break

    # Build a stable signature->parameter lookup by iterating in the same order again.
    sig_to_param = {}
    var_params = [(pname, pspec) for pname, pspec in parameters.items() if pspec["kind"] == "var"]
    const_params = [(pname, pspec) for pname, pspec in parameters.items() if pspec["kind"] == "const"]

    var_i = 0
    const_i = 0
    for sig, occs in sig_to_occs.items():
        present_values = [v for v in sig if v is not MISSING]
        if not present_values:
            continue

        unique_present_values = set(present_values)
        is_required = all(v is not MISSING for v in sig)
        is_constant = len(unique_present_values) == 1

        if is_constant:
            if not is_required:
                pname = const_params[const_i][0]
                const_i += 1
                sig_to_param[sig] = pname
            continue
        else:
            pname = var_params[var_i][0]
            var_i += 1
            sig_to_param[sig] = pname

    # Emit the template.
    for i in range(n_calls):
        func_name = traces[0][i]["name"]
        args = {}
        all_arg_names = set()
        for t in traces:
            all_arg_names.update(t[i].get("args", {}).keys())

        for arg_name in all_arg_names:
            sig = tuple(t[i]["args"].get(arg_name, MISSING) for t in traces)
            present_values = [v for v in sig if v is not MISSING]
            unique_present_values = set(present_values)
            is_required = all(v is not MISSING for v in sig)
            is_constant = len(unique_present_values) == 1

            if is_constant:
                args[arg_name] = {"kind": "const", "value": present_values[0]}
            else:
                pname = sig_to_param[sig]
                args[arg_name] = {"kind": "var", "name": pname, "required": is_required}

        template_calls.append({"name": func_name, "args": args})

    return {"parameters": parameters, "template": template_calls}


def render_function_schema(template, function_name="execute", description=""):
    """
    Render the inferred template as a JSON-schema-like function schema.
    Optional args are omitted from the required list.
    Constant optional args are emitted with `const`.
    """
    properties = {}
    required = []

    for param_name, spec in template["parameters"].items():
        if spec["kind"] == "var":
            properties[param_name] = {
                "type": "string",
                "description": ""
            }
            if spec["required"]:
                required.append(param_name)
        else:
            value = spec["value"]
            schema_type = _infer_json_schema_type(value)
            properties[param_name] = {
                "type": schema_type,
                "const": value,
                "description": ""
            }
            if spec["required"]:
                required.append(param_name)

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

        # code = render_python(template, function_name=name)
        schema = render_function_schema(template, function_name=name)

        describe_prompt = f"Code Trace:{ngram}\nSchema:\n{json.dumps(schema, indent=4)}"
        schema = json.loads(describe_model.generate(describe_prompt, memory=False))

        names.append(name)
        # macros.append(code)
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