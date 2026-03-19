import json

def minimal_args_from_traces(traces):
    """
    traces: list of traces
      each trace is a list of calls,
      each call is [function_name, arg_json_string_or_dict]
      Example call: ["move_to_object", '{"name": "Banana 3"}'] or ["move_to_object", {"name":"Banana 3"}]

    Returns: {
      "hardcoded": [ (func_name, args_dict), ... ],
      "variables": { (position, func_name): [arg_keys_that_vary], ... }
    }
    """
    # normalize input: parse strings into dicts
    norm_traces = []
    for t in traces:
        calls = []
        for call in t.get("code", []):
            func = call[0]
            raw = call[1]
            if isinstance(raw, str):
                args = json.loads(raw)
            elif isinstance(raw, dict):
                args = raw
            else:
                args = dict(raw)
            calls.append((func, args))
        norm_traces.append(calls)

    # Determine max trace length
    max_len = max(len(t) for t in norm_traces)

    hardcoded = []
    variables = {}  # key: (pos, func) -> list of arg keys that vary

    for pos in range(max_len):
        # collect calls at this position across traces (some traces may be shorter)
        calls_at_pos = []
        for calls in norm_traces:
            if pos < len(calls):
                calls_at_pos.append(calls[pos])
            else:
                calls_at_pos.append(None)  # missing

        # if all missing, skip
        if all(c is None for c in calls_at_pos):
            continue

        # If functions differ at this position, treat function name as variable (advanced case)
        funcs = [c[0] if c is not None else None for c in calls_at_pos]
        if len(set(funcs)) != 1:
            # mark function name as variable
            variables[(pos, "FUNCTION_NAME")] = list(set(funcs))
            continue

        func_name = funcs[0]
        # collect all arg keys
        all_keys = set()
        for c in calls_at_pos:
            if c is not None:
                all_keys.update(c[1].keys())

        hard_args = {}
        varying_keys = []
        for k in all_keys:
            values = []
            for c in calls_at_pos:
                if c is None:
                    values.append(None)
                else:
                    values.append(c[1].get(k))
            # if all values equal (including equal None) -> hard-code
            first = values[0]
            if all(v == first for v in values):
                hard_args[k] = first
            else:
                varying_keys.append(k)

        hardcoded.append((pos, func_name, hard_args))
        if varying_keys:
            variables[(pos, func_name)] = varying_keys

    return {"hardcoded": hardcoded, "variables": variables}

def learn_macros(ngrams):
    for ngram, traces in ngrams.items():
        if len(traces) < 5:
            continue

        res = minimal_args_from_traces(traces)
        print(f"Macro for ngram {ngram}:")
        print(res)
        print("\n\n")

if __name__ == "__main__":

    with open("rcg/data/ngrams.json", "r") as f:
        ngrams = json.load(f)

    learn_macros(ngrams)