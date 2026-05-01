from rcg.llm import get_info, move_to_object, grasp_object, release_object, move_to_position

def move_grasped_object_and_release(z, x, name):
    """
    Grasp a specified object, move it horizontally in the X–Z plane to a new position relative to its current location, then release it.

Args:
    z (number): Offset in meters to move the grasped object along the Z-axis (forward/backward). Positive = move forward, negative = move backward.
    x (number): Offset in meters to move the grasped object along the X-axis (left/right). Positive = move right, negative = move left.
    name (string): Identifier of the object to grasp, move, and release.
    """
    grasp_object(name, 0.5)
    move_to_position(x, 0, z, 2.0, True)
    release_object(True, 0.1)

MACRO_MAP = {
    "move_grasped_object_and_release": move_grasped_object_and_release
}