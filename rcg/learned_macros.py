from rcg.llm import get_info, move_to_object, grasp_object, release_object, move_to_position

def move_to_and_grasp_object(name):
    """
    Move the robot arm to the specified object with a small vertical offset and then close the gripper to grasp it.

Args:
    name (string): The name or identifier of the target object to move to and grasp (e.g., 'banana 1').
    """
    move_to_object(name, 0.0, 0.1, 0.0, 2.0)
    grasp_object(name, 0.5)

MACRO_MAP = {
    "move_to_and_grasp_object": move_to_and_grasp_object
}