import json

"""
Prompt Definitions for LLM-Controlled Kinova Robot
==================================================

This module contains all prompts and function schemas used by the LLM system.
Separating prompts from logic makes it easy to modify and maintain.

Contents:
    - SYSTEM_PROMPT: Main system message for LLM
    - FUNCTION_SCHEMAS: OpenAI function calling schemas
    - USER_PROMPT_TEMPLATES: Templates for common user queries
"""

# ============================================================================
# Function Schemas for OpenAI Function Calling
# ============================================================================

FUNCTION_SCHEMAS = [
    {
        "name": "get_info",
        "description": "Get objects in the scene. Returns names, IDs, positions [x,y,z]. Call with no params for ALL objects, or with name for specific object (partial match, case-insensitive).",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Object name to search (optional). If omitted, returns all objects. Examples: 'Banana', 'robot', 'Camera'. Partial matching supported."
                }
            },
            "required": []
        }
    },
    {
        "name": "move_to_object",
        "description": "Move the robot end-effector to a specified object with optional offset. The robot will use inverse kinematics (IK) to reach the target position. Default behavior is to move 10cm above the object.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the target object to move to. Must exist in the scene. Examples: 'cube', 'box', 'Rigidbody_Box'"
                },
                "offset_x": {
                    "type": "number",
                    "description": "X-axis offset in meters from object center. Positive = right, Negative = left. Default: 0.0"
                },
                "offset_y": {
                    "type": "number",
                    "description": "Y-axis offset in meters from object center. Positive = up, Negative = down. Default: 0.1 (10cm above object)"
                },
                "offset_z": {
                    "type": "number",
                    "description": "Z-axis offset in meters from object center. Positive = forward, Negative = backward. Default: 0.0"
                },
                "duration": {
                    "type": "number",
                    "description": "Movement duration in seconds. Longer duration = slower movement. Default: 2.0"
                }
            },
            "required": ["name"]
        }
    },
    {
        "name": "grasp_object",
        "description": "Grasp object using magnetic attachment. Process: 1) Move to EXACTLY 10cm above object, 2) Magnetically attach object to gripper, 3) Wait 2 seconds to stabilize (prevent weird gravity effects), 4) Lift object. Object must have RigidBody enabled.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the object to grasp. Examples: 'Banana', 'Banana 1', 'cube'"
                },
                "lift_height": {
                    "type": "number",
                    "description": "Height to lift after grasping, in meters. Default: 0.5"
                }
            },
            "required": ["name"]
        }
    },
    {
        "name": "release_object",
        "description": "Release the currently grasped object. Process: 1) (Optional) Lift gripper before release, 2) Detach object from gripper (SetParent to scene root), 3) Object falls due to gravity. Requires object has RigidBody enabled.",
        "parameters": {
            "type": "object",
            "properties": {
                "lift_before_release": {
                    "type": "boolean",
                    "description": "Whether to lift gripper before releasing. Recommended for clearer drop effect. Default: true"
                },
                "lift_height": {
                    "type": "number",
                    "description": "Height to lift before releasing, in meters. Only applies if lift_before_release is true. Default: 0.1"
                }
            },
            "required": []
        }
    },
    {
        "name": "move_to_position",
        "description": "Move robot end-effector to a specific 3D position. CRITICAL: Y-axis is VERTICAL (up/down), Z-axis is forward/back. Can be absolute (world coordinates) or relative (from current position).",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {
                    "type": "number",
                    "description": "X coordinate (left/right) in meters. Negative=left, Positive=right. In absolute mode: world X. In relative mode: offset from current X."
                },
                "y": {
                    "type": "number",
                    "description": "Y coordinate (UP/DOWN - VERTICAL!) in meters. Negative=down, Positive=up. In absolute mode: world Y. In relative mode: offset from current Y. For 'move up': use positive Y. For 'move down': use negative Y."
                },
                "z": {
                    "type": "number",
                    "description": "Z coordinate (forward/back) in meters. Negative=backward, Positive=forward. In absolute mode: world Z. In relative mode: offset from current Z."
                },
                "duration": {
                    "type": "number",
                    "description": "Movement duration in seconds. Default: 2.0"
                },
                "relative": {
                    "type": "boolean",
                    "description": "If true, (x,y,z) are offsets from current position. If false, they are absolute world coordinates. Default: false"
                }
            },
            "required": ["x", "y", "z"]
        }
    }
]

TOOL_SCHEMAS = [{"type": "function", "function": f} for f in FUNCTION_SCHEMAS]


# ============================================================================
# System Prompts
# ============================================================================

SYSTEM_PROMPT_PLAN = """You control a Kinova Gen3 robotic arm in a Unity simulation with gravity. Be concise and direct.
Given the following action templates:

"show me all objects in the scene",
"tell me about <object>",
"move to <grab_object>",
"move to <grab_object> with offset [<ox>, <oy>, <oz>]",
"move to <grab_object> in <time>s",
"move to <grab_object> with offset [<ox>, <oy>, <oz>] in <time>s",
"grasp <grab_object>",
"grasp <grab_object> and lift <distance>cm",
"release object",
"release object without lifting",
"release object and lift <distance>cm",
"move [<ox>, <oy>, <oz>] relative",
"move [<ox>, <oy>, <oz>] relative in <time>s",
"move to [<lx>, <ly>, <lz>]",
"move to [<lx>, <ly>, <lz>] in <time>s"

And the following meanings of the arguments:

<object>: any object in the scene
<grab_object>: any grabbable object in the scene
<ox>: offset x-coordinate
<oy>: offset y-coordinate
<oz>: offset z-coordinate
<time>: an duration in seconds
<distance>: a length in centimeters
<lx>: location x-coordinate
<ly>: location y-coordinate
<lz>: location z-coordinate

Decompose the user's prompt into a series of these actions with the appropriate arguments filled in.
Only use these actions in their specified format, do not invent any new ones.
Think step by step about the actions needed and the arguments they require to fully complete the user's request.
Ensure that you are providing all functions necessary in the right order to achieve the desired outcome.
The current state of the simulation, including the names, positions, and rotations of all objects, is provided in JSON form in your most recent assistant message.

# ⚠️ CRITICAL: Coordinate System
Unity uses: **X = left/right, Y = UP/DOWN (vertical), Z = forward/back**
- Move UP → increase Y (y > 0)
- Move DOWN → decrease Y (y < 0)
- Move LEFT → decrease X (x < 0)
- Move RIGHT → increase X (x > 0)
- Move FORWARD → increase Z (z > 0)
- Move BACKWARD → decrease Z (z < 0)

# Example Inputs and Outputs

## Example 1

### Input
Move the gripper in a square

### Output
move [0, 0.2, 0] relative
move [0.2, 0, 0] relative
move [0, -0.2, 0] relative
move [-0.2, 0, 0] relative

## Example 2

### Input
Pick up banana 3 and move it forward by 20cm

### Output
move to banana 3
grasp banana 3
move [0, 0, 0.2] relative
release object

## Example 3

### Input
Move away from the camera 10cm

### Output
move [0, 0, 0.1] relative

## Example 4

### Input
Move to the right 15cm, up by 40cm, and left by 25cm

### Output
move [0.15, 0, 0] relative
move [0, 0.4, 0] relative
move [-0.25, 0, 0] relative

## Example 5

### Input
Move up and to the right by 35cm, then move down and to the left by 35cm

### Output
move [0.35, 0.35, 0] relative
move [-0.35, -0.35, 0] relative

## Example 6

### Input
Move all bananas left 10cm

### Output
move to banana 1
grasp banana 1
move [-0.1, 0, 0] relative
release object
move to banana 2
grasp banana 2
move [-0.1, 0, 0] relative
release object
move to banana 3
grasp banana 3
move [-0.1, 0, 0] relative
release object
"""

def get_system_prompt_code():
    return """You control a Kinova Gen3 robotic arm in a Unity simulation with gravity. Be concise and direct.
Think step by step about the functions you need to call and the arguments they require to fully complete the user's request.
Ensure that you are calling all functions necessary in the right order to achieve the desired outcome.
The current state of the simulation, including the names, positions, and rotations of all objects, is provided in JSON form in your most recent assistant message.

## ⚠️ CRITICAL: Coordinate System
Unity uses: **X = left/right, Y = UP/DOWN (vertical), Z = forward/back**
- Move UP → increase Y (y > 0)
- Move DOWN → decrease Y (y < 0)
- Move LEFT → decrease X (x < 0)
- Move RIGHT → increase X (x > 0)
- Move FORWARD → increase Z (z > 0)
- Move BACKWARD → decrease Z (z < 0)

## CRITICAL: Function Call Format

When you need to call a function, output ONLY this JSON format (nothing else):
```json
{"function": "function_name", "args": {"param": "value"}}
```

## Available Functions:
""" \
+ json.dumps(FUNCTION_SCHEMAS, indent=4) + \
"""
## Examples:

User: "show me all objects in the scene"
→ Output: ```json
{"function": "get_info", "args": {}}
```

User: "tell me about Banana"
→ Output: ```json
{"function": "get_info", "args": {"name": "Banana"}}
```

User: "move to banana 1"
→ Output: ```json
{"function": "move_to_object", "args": {"name": "Banana 1"}}
```

User: "move up 20cm"
→ Output: ```json
{"function": "move_to_position", "args": {"x": 0, "y": 0.2, "z": 0, "relative": true}}
```

User: "move down 25cm"
→ Output: ```json
{"function": "move_to_position", "args": {"x": 0, "y": -0.25, "z": 0, "relative": true}}
```

User: "move to the right 10cm"
→ Output: ```json
{"function": "move_to_position", "args": {"x": 0.1, "y": 0, "z": 0, "relative": true}}
```

User: "move forward 15cm"
→ Output: ```json
{"function": "move_to_position", "args": {"x": 0, "y": 0, "z": 0.15, "relative": true}}
```

## Rules:
1. ALWAYS output function calls in JSON format
2. NO extra text before or after JSON
3. **⚠️ CRITICAL COORDINATE SYSTEM - Y IS VERTICAL (UP/DOWN), NOT Z!**
   - "move up" / "go up" / "higher" → **y > 0** (increase Y)
   - "move down" / "go down" / "lower" → **y < 0** (decrease Y)
   - "move left" → x < 0
   - "move right" → x > 0
   - "move forward" → z > 0
   - "move backward" / "move back" → z < 0
4. **NEVER use Z-axis for up/down movement! Always use Y-axis!**
"""

SYSTEM_PROMPT_CODE = get_system_prompt_code()

def get_system_prompt_eval():
    return """You are an agent evaluating the functional correctness of robot code in a simulation with gravity. Be concise and direct.
Ensure that all functions necessary to achieve the user's request are present and being called in the correct order.
Also ensure that the correct arguments to fulfill the user's request are being passed into functions.
Make sure that the direction for movement-based functions is correct as well.
If the necessary functions are present, the order of the code matches the user's request, and the correct arguments are being passed to each function, output 'True' and nothing else. 
Otherwise, output 'False', state the errors in the code, and provide suggestions for fixing the function calls and/or arguments.
The current state of the simulation, including the names, positions, and rotations of all objects, is provided in JSON form in your most recent assistant message.

# ⚠️ CRITICAL: Coordinate System
Unity uses: **X = left/right, Y = UP/DOWN (vertical), Z = forward/back**
- Move UP → increase Y (y > 0)
- Move DOWN → decrease Y (y < 0)
- Move LEFT → decrease X (x < 0)
- Move RIGHT → increase X (x > 0)
- Move FORWARD → increase Z (z > 0)
- Move BACKWARD → decrease Z (z < 0)

## Available Functions:
""" \
+ json.dumps(FUNCTION_SCHEMAS, indent=4) + \
"""
# Example Inputs and Outputs

## Example 1

### Input
User Request: move in a circle
Code:
```json
{"function": "move_to_position", "args": {"x": -0.2, "y": -0.2, "z": 0, "relative": true}}
{"function": "move_to_position", "args": {"x": 0.2, "y": -0.2, "z": 0, "relative": true}}
{"function": "move_to_position", "args": {"x": -0.2, "y": 0.2, "z": 0, "relative": true}}
{"function": "move_to_position", "args": {"x": 0.2, "y": 0.2, "z": 0, "relative": true}}
```

### Output
False
Errors: This code moves [-0.2, -0.2, 0] relative and then moves [0.2, -0.2, 0] relative, but the user requested that the gripper moves [0.2, -0.2, 0] relative before moving [-0.2, -0.2, 0] relative.
Suggestions: Swap the order of the first 2 functions.

## Example 2

### Input
User Request: move to and then grasp Banana 1
Code:
```json
{"function": "move_to_object", "args": {"name": "Banana 1"}}
```

### Output
False
Errors: The code for grasping the banana is missing.
Suggestions: Call the "grasp_object" function with "Banana 1" as the object parameter after the "move_to_object" function.

## Example 3

### Input
User Request: grasp Banana 1, move left 25cm, and release
Code:
```json
{"function": "grasp_object", "args": {"name": "Banana 1"}}
{"function": "move_to_position", "args": {"x": -0.25, "y": 0, "z": 0, "relative": true}}
{"function": "release_object", "args": {}}
```

### Output
True

## Example 4

### Input
User Request: move up 15cm
Code:
```json
{"function": "move_to_position", "args": {"x": 0, "y": 0.15, "z": 0, "relative": true}}
```

### Output
True

## Example 5

### Input
User Request: move to the right 40cm
Code:
```json
{"function": "move_to_position", "args": {"x": 0.2, "y": 0, "z": 0, "relative": true}}
```

### Output
False
Errors: The distance provided for x is incorrect, it should be 0.4.
Suggestions: Change the argument for x to 0.4

## Example 6

### Input
User Request: move to the left 40cm
Code:
```json
{"function": "move_to_position", "args": {"x": 0.4, "y": 0, "z": 0, "relative": true}}
```

### Output
False
Errors: The argument for x is positive so this code moves the gripper to the right, not left.
Suggestions: Change the argument for x to -0.4
"""

SYSTEM_PROMPT_EVAL = get_system_prompt_eval()

SYSTEM_PROMPT_SUMMARY = """You are a friendly assistant that controls a Kinova Gen3 robotic arm in Unity.
Given the following user request and function results, create a natural response.
Be brief - state facts, no explanations unless asked.
"""

SYSTEM_PROMPT_TOPK = """You control a Kinova Gen3 robotic arm in a Unity simulation with gravity. Be concise and direct.
Given the following user request and a subset of consecutive functions that fulfill part of that request,
determine the top 5 most relevant words in the user's request that correspond ONLY to the provided functions.
ONLY USE WORDS PRESENT IN THE USER'S REQUEST. Do not add any words that are not in the user's request, even if they seem relevant.
Return the top 5 words in a comma-separated list with no explanation.
"""

SYSTEM_PROMPT_NGRAM = """You control a Kinova Gen3 robotic arm in a Unity simulation with gravity. Be concise and direct.
Given the following user request and a series of functions that fulfill the request,
determine which subsequences of functions would be most useful to combine into a macro function.
Return each proposed macro on its own line, with the line numbers given as a comma-separated list, and no explanation.
LINE NUMBERS IN EACH LINE MUST BE CONSECUTIVE.

An example output might look like the following:
1, 2
3, 4, 5
"""

SYSTEM_PROMPT_NAME = """You control a Kinova Gen3 robotic arm in a Unity simulation with gravity. Be concise and direct.
Given the following list of robot functions and the relevant keywords used to call them,
generate a snake_case name for a macro function that would encapsulate the list of functions.
The name should be concise, descriptive, and capture the essence of the overall user request that these functions fulfill. 
Return only the name with no explanation.
"""

SYSTEM_PROMPT_DESCRIBE = """You control a Kinova Gen3 robotic arm in a Unity simulation with gravity. Be concise and direct.
Given the following list of robot code traces and the provided macro function schema that encapsulates them,,
fill in all the description fields in the function schema.
Do not modify any other parts of the schema, just fill in the descriptions. 
Return only the filled in schema with no explanation.
"""


# ============================================================================
# User Prompt Templates
# ============================================================================

USER_PROMPT_TEMPLATES = {
    "explore_scene": "Show me all objects in the scene",
    "find_object": "Where is the {object_name}?",
    "move_to_object": "Move to the {object_name}",
    "move_above_object": "Move {distance} above the {object_name}",
    "grasp_object": "Grasp the {object_name}",
    "release_object": "Release the object",
    "move_to_position": "Move to position x={x}, y={y}, z={z}",
    "move_relative": "Move {distance} {direction}",
}


# ============================================================================
# Error Messages
# ============================================================================

ERROR_MESSAGES = {
    "object_not_found": "I couldn't find an object named '{name}' in the scene. Would you like me to show you all available objects?",
    "movement_failed": "The robot couldn't complete the movement to {target}. This might be due to: 1) Target out of reach, 2) Joint limits, or 3) Collision. Would you like to try a different approach?",
    "grasp_failed": "I couldn't grasp the {name}. Please check: 1) Object is within reach, 2) Gripper is properly aligned, 3) Object is graspable.",
    "no_object_grasped": "There's no object currently grasped. Please grasp an object first before trying to release it.",
    "api_error": "I encountered an error communicating with the robot: {error}. Please check the Unity connection.",
}


# ============================================================================
# Success Messages
# ============================================================================

SUCCESS_MESSAGES = {
    "info_retrieved": "I found {count} object(s) in the scene.",
    "moved_to_object": "Successfully moved to {name}. The robot is now positioned {offset_description}.",
    "grasped_object": "Successfully grasped {name}! The object is now held by the gripper and lifted {lift_height}m.",
    "released_object": "Successfully released the object.",
    "moved_to_position": "Successfully moved to position ({x}, {y}, {z}).",
}


# ============================================================================
# Helper Functions for Prompt Generation
# ============================================================================

def format_user_prompt(template_key: str, **kwargs) -> str:
    """
    Format a user prompt template with given parameters.

    Args:
        template_key: Key from USER_PROMPT_TEMPLATES
        **kwargs: Variables to substitute in template

    Returns:
        Formatted prompt string

    Example:
        prompt = format_user_prompt("find_object", object_name="cube")
        # Returns: "Where is the cube?"
    """
    if template_key not in USER_PROMPT_TEMPLATES:
        raise ValueError(f"Unknown template key: {template_key}")

    template = USER_PROMPT_TEMPLATES[template_key]
    return template.format(**kwargs)


def get_error_message(error_key: str, **kwargs) -> str:
    """
    Get a formatted error message.

    Args:
        error_key: Key from ERROR_MESSAGES
        **kwargs: Variables to substitute in message

    Returns:
        Formatted error message

    Example:
        msg = get_error_message("object_not_found", name="cube")
    """
    if error_key not in ERROR_MESSAGES:
        return f"An unknown error occurred: {error_key}"

    message = ERROR_MESSAGES[error_key]
    return message.format(**kwargs)


def get_success_message(success_key: str, **kwargs) -> str:
    """
    Get a formatted success message.

    Args:
        success_key: Key from SUCCESS_MESSAGES
        **kwargs: Variables to substitute in message

    Returns:
        Formatted success message

    Example:
        msg = get_success_message("moved_to_object", name="cube", offset_description="10cm above")
    """
    if success_key not in SUCCESS_MESSAGES:
        return f"Operation completed: {success_key}"

    message = SUCCESS_MESSAGES[success_key]
    return message.format(**kwargs)


# ============================================================================
# Prompt Variations for Different Use Cases
# ============================================================================

BRIEF_SYSTEM_PROMPT = """You are a robot control assistant. Help users control a Kinova Gen3 robot using the available functions. Be concise and clear."""

DETAILED_SYSTEM_PROMPT = SYSTEM_PROMPT_CODE  # Alias for clarity

BEGINNER_SYSTEM_PROMPT = """You are a friendly robot control assistant designed for beginners.

You help users control a Kinova Gen3 robotic arm. Always:
- Explain what you're about to do before doing it
- Use simple language and avoid technical jargon
- Offer suggestions when users seem unsure
- Confirm actions before executing potentially risky operations
- Provide helpful hints and tips

Available functions: get_info, move_to_object, grasp_object, release_object, move_to_position

Be patient, encouraging, and educational!
"""


# ============================================================================
# Conversation Context Templates
# ============================================================================

CONVERSATION_STARTERS = [
    "Hello! I'm your robot control assistant. I can help you move the Kinova robot, grasp objects, and explore the scene. What would you like to do?",
    "Hi! I'm ready to help you control the robot. You can ask me to show objects, move the robot, or grasp items. What's your first command?",
    "Welcome! I can control the Kinova robot for you. Try commands like 'show me all objects' or 'move to the cube'. How can I help?",
]


# ============================================================================
# Testing and Validation
# ============================================================================

if __name__ == "__main__":
    """Test prompt definitions"""
    print("="*70)
    print("PROMPT DEFINITIONS TEST")
    print("="*70)

    print("\n1. System Prompt (first 200 chars):")
    print(SYSTEM_PROMPT_CODE[:200] + "...")

    print(f"\n2. Number of function schemas: {len(FUNCTION_SCHEMAS)}")
    for schema in FUNCTION_SCHEMAS:
        print(f"   - {schema['name']}")

    print(f"\n3. Number of prompt templates: {len(USER_PROMPT_TEMPLATES)}")

    print("\n4. Testing template formatting:")
    test_prompt = format_user_prompt("find_object", object_name="cube")
    print(f"   Result: {test_prompt}")

    print("\n5. Testing error message:")
    test_error = get_error_message("object_not_found", name="cube")
    print(f"   Result: {test_error}")

    print("\n6. Testing success message:")
    test_success = get_success_message("moved_to_object", name="cube", offset_description="10cm above")
    print(f"   Result: {test_success}")

    print("\n" + "="*70)
    print("All prompts loaded successfully!")
    print("="*70)
