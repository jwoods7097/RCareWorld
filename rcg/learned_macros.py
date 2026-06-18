from rcg.robot import get_info, move_to_object, grasp_object, release_object, move_to_position

def grasp_object_and_move_closer_to_mouth(z, name):
    """
    Grasp a specified object and move it to a position closer to the user’s mouth along the Z (forward/back) axis while keeping a fixed X and Y.

Args:
    z (number): Target Z coordinate (forward/back) for the object after it is grasped, to adjust how close it is to the user’s mouth.
    name (string): The name/identifier of the object for the arm to grasp before moving it closer.
    """
    grasp_object(name, 0.5)
    move_to_position(-1.02, 1.6, z, 2.0, False)

MACRO_MAP = {
    "grasp_object_and_move_closer_to_mouth": grasp_object_and_move_closer_to_mouth,
}