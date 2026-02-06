import argparse
import json
import os
import random

prompt_templates = [
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
    "move <direction> <distance>cm",
    "move <direction> <distance>cm in <time>s",
    "move [<ox>, <oy>, <oz>] relative",
    "move [<ox>, <oy>, <oz>] relative in <time>s",
    "move to [<lx>, <ly>, <lz>]",
    "move to [<lx>, <ly>, <lz>] in <time>s",
]

code_templates = [
    r'{"function": "get_info", "args": {}}',
    r'{"function": "get_info", "args": {"name": "<object>"}}',
    r'{"function": "move_to_object", "args": {"name": "<grab_object>"}}',
    r'{"function": "move_to_object", "args": {"name": "<grab_object>", "offset_x": <ox>, "offset_y": <oy>, "offset_z": <oz>}}',
    r'{"function": "move_to_object", "args": {"name": "<grab_object>", "duration": <time>}}',
    r'{"function": "move_to_object", "args": {"name": "<grab_object>", "offset_x": <ox>, "offset_y": <oy>, "offset_z": <oz>, "duration": <time>}}',
    r'{"function": "grasp_object", "args": {"name": "<grab_object>"}}',
    r'{"function": "grasp_object", "args": {"name": "<grab_object>", "lift_height": <distance>}}',
    r'{"function": "release_object", "args": {}}',
    r'{"function": "release_object", "args": {"lift_before_release": false}}',
    r'{"function": "release_object", "args": {"lift_height": <distance>}}',
    r'{"function": "move_to_position", "args": {"x": <ox>, "y": <oy>, "z": <oz>, "relative": true}}',
    r'{"function": "move_to_position", "args": {"x": <ox>, "y": <oy>, "z": <oz>, "relative": true, "duration": <time>}}',
    r'{"function": "move_to_position", "args": {"x": <ox>, "y": <oy>, "z": <oz>, "relative": true}}',
    r'{"function": "move_to_position", "args": {"x": <ox>, "y": <oy>, "z": <oz>, "relative": true, "duration": <time>}}',
    r'{"function": "move_to_position", "args": {"x": <lx>, "y": <ly>, "z": <lz>}}',
    r'{"function": "move_to_position", "args": {"x": <lx>, "y": <ly>, "z": <lz>, "duration": <time>}}',
]

objects = [
    "robotiq_arg2f_85_model",
    "kinova_gen3_7dof-robotiq85",
    "Banana 1",
    "Banana 2",
    "Banana 3",
    "Camera",
]

grab_objects = [
    "Banana 1",
    "Banana 2",
    "Banana 3",
]

directions = [
    "forward",
    "backward",
    "left",
    "right",
    "up",
    "down",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_prompts", type=int, default=1000, help="Number of prompts to generate")
    args = parser.parse_args()
    
    data = []
    i = 0

    while i < args.num_prompts:
        prompts = []
        codes = []

        # Generate 1 to 5 step prompts
        for _ in range(num_steps := random.randint(1, 5)):
            # Individual prompt and code
            min_idx = 0 if num_steps == 1 else 2  # get_info calls don't make sense in multi-step prompts
            idx = random.randint(min_idx, len(prompt_templates) - 1)
            prompt = prompt_templates[idx]
            code = code_templates[idx]

            # Fill direction parameters
            while "<direction>" in prompt:
                # Fill direction in prompt
                direction = random.choice(directions)
                prompt = prompt.replace("<direction>", direction, 1)
                
                # Fill distance in prompt
                distance = random.randint(0, 100) / 100.0
                prompt = prompt.replace("<distance>", str(distance), 1)

                # Determine offset from direction and distance
                if direction == "up":
                    ox, oy, oz = 0, distance, 0
                elif direction == "down":
                    ox, oy, oz = 0, -distance, 0
                elif direction == "left":
                    ox, oy, oz = -distance, 0, 0
                elif direction == "right":
                    ox, oy, oz = distance, 0, 0
                elif direction == "forward":
                    ox, oy, oz = 0, 0, distance
                elif direction == "backward":
                    ox, oy, oz = 0, 0, -distance

                # Fill offsets in code
                code = code.replace("<ox>", str(ox), 1)
                code = code.replace("<oy>", str(oy), 1)
                code = code.replace("<oz>", str(oz), 1)

            # Fill object parameters
            while "<grab_object>" in prompt:
                obj = random.choice(grab_objects)
                prompt = prompt.replace("<grab_object>", obj, 1)
                code = code.replace("<grab_object>", obj, 1)

            while "<object>" in prompt:
                obj = random.choice(objects)
                prompt = prompt.replace("<object>", obj, 1)
                code = code.replace("<object>", obj, 1)

            # Fill time parameters
            while "<time>" in prompt:
                time = round(random.uniform(0.5, 5.0), 2)
                prompt = prompt.replace("<time>", str(time), 1)
                code = code.replace("<time>", str(time), 1)

            # Fill distance parameters
            while "<distance>" in prompt:
                distance = random.randint(0, 100) / 100.0
                prompt = prompt.replace("<distance>", str(distance), 1)
                code = code.replace("<distance>", str(distance), 1)

            # Fill offset parameters
            while "<ox>" in prompt:
                ox = round(random.uniform(-1, 1), 2)
                prompt = prompt.replace("<ox>", str(ox), 1)
                code = code.replace("<ox>", str(ox), 1)

            while "<oy>" in prompt:
                oy = round(random.uniform(-1, 1), 2)
                prompt = prompt.replace("<oy>", str(oy), 1)
                code = code.replace("<oy>", str(oy), 1)

            while "<oz>" in prompt:
                oz = round(random.uniform(-1, 1), 2)
                prompt = prompt.replace("<oz>", str(oz), 1)
                code = code.replace("<oz>", str(oz), 1)

            # Fill location parameters
            while "<lx>" in prompt:
                lx = round(random.uniform(-2, 2), 2)
                prompt = prompt.replace("<lx>", str(lx), 1)
                code = code.replace("<lx>", str(lx), 1)

            while "<ly>" in prompt:
                ly = round(random.uniform(-1, 1), 2)
                prompt = prompt.replace("<ly>", str(ly), 1)
                code = code.replace("<ly>", str(ly), 1)

            while "<lz>" in prompt:
                lz = round(random.uniform(-2, 2), 2)
                prompt = prompt.replace("<lz>", str(lz), 1)
                code = code.replace("<lz>", str(lz), 1)

            prompts.append(prompt)
            codes.append(code)

        # Merge into single prompt and code block
        if not any([prompt == p["prompt"] for p in data]):
            data.append({"prompt": ", ".join(prompts), "correct": codes})
            i += 1

    # Generate incorrect code
    for i, item in enumerate(data):
        code = json.loads("[" + ",".join(item["correct"]) + "]")

        # Introduce errors by randomly swapping functions, deleting functions, or changing parameters
        for _ in range(random.randint(1, 3)):
            error_type = random.choice(["none", "swap", "delete", "change_param"])
            idx1 = random.randint(0, len(code) - 1)
            
            if error_type == "swap" and len(code) > 1:
                idx2 = idx1
                while idx2 == idx1:
                    idx2 = random.randint(0, len(code) - 1)
                code[idx1], code[idx2] = code[idx2], code[idx1]
            elif error_type == "delete" and len(code) > 1:
                del code[idx1]
            elif error_type == "change_param" and len(code[idx1]["args"]) > 0:
                key, value = random.choice(list(code[idx1]["args"].items()))
                if isinstance(value, (int, float)):
                    code[idx1]["args"][key] = round(value + random.uniform(-1.0, 1.0), 2)
                elif isinstance(value, str):
                    choice = value
                    while choice == value:
                        choice = random.choice(objects)
                    code[idx1]["args"][key] = choice
                elif isinstance(value, bool):
                    code[idx1]["args"][key] = not value
            else:
                pass

        data[i]["incorrect"] = "\n".join([json.dumps(c) for c in code])
        data[i]["correct"] = "\n".join(item["correct"])

    # Save code to file
    os.makedirs("rcg/data", exist_ok=True)
    with open("rcg/data/generated_data.json", "w") as f:
        json.dump(data, f, indent=4)