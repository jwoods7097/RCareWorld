from rcg.data_gen import prompt_templates, code_templates, objects, grab_objects, directions
import re
import json
from typing import Any, Dict, List, Tuple, Optional

# --- Placeholder -> regex pattern mapping ---
# numeric placeholders => signed int/float
_NUM_RE = r"(?P<{name}>-?\d+(?:\.\d+)?)"
# generic text placeholders (object names may include spaces, hyphens, underscores)
_TEXT_RE = r"(?P<{name}>[\w\-\s\.]+?)"
# direction names (word)
_DIR_RE = r"(?P<{name}>\w+)"

# mapping for which placeholders are numeric vs text vs direction
NUMERIC_PLACEHOLDERS = {"ox", "oy", "oz", "time", "distance", "lx", "ly", "lz"}
TEXT_PLACEHOLDERS = {"object", "grab_object"}
DIRECTION_PLACEHOLDERS = {"direction"}


def _placeholder_regex(name: str) -> str:
    """Return the regex (with named group) for a placeholder name."""
    if name in NUMERIC_PLACEHOLDERS:
        return _NUM_RE.format(name=name)
    if name in TEXT_PLACEHOLDERS:
        return _TEXT_RE.format(name=name)
    if name in DIRECTION_PLACEHOLDERS:
        return _DIR_RE.format(name=name)
    # default to a permissive token (non-greedy)
    return r"(?P<%s>.+?)" % name


def _escape_literal(s: str) -> str:
    """Escape punctuation in literal parts of the template so they match literally."""
    return re.escape(s)


def compile_template(prompt_template: str) -> Tuple[re.Pattern, List[str]]:
    """
    Compile a prompt template into a regex pattern and return the pattern
    and the list of placeholder names (extraction order).
    """
    # We'll scan the template and replace sequences like <name> with the appropriate regex.
    parts: List[str] = []
    placeholder_names: List[str] = []
    i = 0
    while i < len(prompt_template):
        if prompt_template[i] == "<":
            # read until '>'
            j = prompt_template.find(">", i + 1)
            if j == -1:
                raise ValueError("Unmatched '<' in template")
            name = prompt_template[i + 1:j].strip()
            placeholder_names.append(name)
            parts.append(_placeholder_regex(name))
            i = j + 1
        else:
            # consume literal until next '<' or end
            j = prompt_template.find("<", i)
            literal = prompt_template[i:] if j == -1 else prompt_template[i:j]
            parts.append(_escape_literal(literal))
            i = len(prompt_template) if j == -1 else j

    # anchor the whole pattern
    pattern_text = r"^\s*" + "".join(parts) + r"\s*$"
    # make regex case-insensitive to be a bit more flexible
    return re.compile(pattern_text, flags=re.IGNORECASE), placeholder_names


def compile_all(prompt_templates: List[str]) -> List[Tuple[re.Pattern, List[str], str]]:
    """
    Compile all prompt_templates and pair each compiled regex with its corresponding code_template.
    Returns a list of tuples: (pattern, placeholder_names, code_template).
    """
    if len(prompt_templates) != len(code_templates):
        raise ValueError("prompt_templates and code_templates length mismatch")
    compiled = []
    for p_templ, c_templ in zip(prompt_templates, code_templates):
        pattern, names = compile_template(p_templ)
        compiled.append((pattern, names, c_templ))
    return compiled


# --- Validation helpers ---
def _validate_object(name: str) -> str:
    # match ignoring case, but return canonical name if matched
    for o in objects:
        if o.lower() == name.lower():
            return o
    raise ValueError(f"Unknown object: '{name}'")


def _validate_grab_object(name: str) -> str:
    for o in grab_objects:
        if o.lower() == name.lower():
            return o
    raise ValueError(f"Unknown grab_object: '{name}'")


def _validate_direction(name: str) -> str:
    for d in directions:
        if d.lower() == name.lower():
            return d
    raise ValueError(f"Unknown direction: '{name}'")


def _coerce_value(name: str, raw: str) -> Any:
    """Coerce raw string capture to an int/float/string and validate if needed."""
    if name in NUMERIC_PLACEHOLDERS:
        # choose int when integer-looking, else float
        if re.fullmatch(r"-?\d+", raw):
            return int(raw)
        return float(raw)
    if name in TEXT_PLACEHOLDERS:
        # validate against lists
        if name == "object":
            return _validate_object(raw.strip())
        if name == "grab_object":
            return _validate_grab_object(raw.strip())
        return raw.strip()
    if name in DIRECTION_PLACEHOLDERS:
        return _validate_direction(raw.strip())
    # default: return stripped string
    return raw.strip()


def render_code_template(code_template: str, params: Dict[str, Any]) -> Dict:
    """
    Substitute params into code_template string robustly and return a Python dict.
    Strategy:
      - replace occurrences of '"<name>"' with the JSON-encoded value (so string placeholders
        are inserted without double-quoting errors),
      - replace occurrences of '<name>' (without quotes) with the JSON-encoded primitive
        (numbers / booleans / null) or raw string (if appropriate).
    """
    s = code_template

    # First replace quoted placeholders: '"<name>"' -> json.dumps(value)
    for k, v in params.items():
        quoted_placeholder = f'"<{k}>"'
        if quoted_placeholder in s:
            s = s.replace(quoted_placeholder, json.dumps(v))

    # Then replace unquoted placeholders (e.g., <ox>, <time>)
    # For these, we want the JSON literal representation (no extra quotes)
    for k, v in params.items():
        placeholder = f"<{k}>"
        if placeholder in s:
            # if value is a string but the template left it unquoted, that's probably an error;
            # we still insert a JSON string literal to keep JSON validity.
            s = s.replace(placeholder, json.dumps(v))

    # Finally load as JSON into dict (raises if malformed)
    return json.loads(s)


def prompt_to_code_general(prompt: str) -> Dict[str, Any]:
    prompt = prompt.strip()
    # try each compiled template in order
    for pattern, names, code_templ in compile_all(prompt_templates):
        m = pattern.fullmatch(prompt)
        if not m:
            continue
        # extract and coerce
        params: Dict[str, Any] = {}
        for name in names:
            raw = m.group(name)
            if raw is None:
                continue
            try:
                params[name] = _coerce_value(name, raw)
            except ValueError as e:
                # validation failed for this match — skip to next template
                raise ValueError(f"Validation error for placeholder <{name}>: {e}")
        # some templates use the same placeholder multiple times (ok)
        # render template
        try:
            code_dict = render_code_template(code_templ, params)
            return json.dumps(code_dict)
        except Exception as e:
            raise ValueError(f"Failed to render code template: {e}")
    raise ValueError("No template matched the given prompt.")