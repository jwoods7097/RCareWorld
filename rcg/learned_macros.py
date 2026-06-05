from rcg.robot import get_info, move_to_object, grasp_object, release_object, move_to_position

def deliver_object_to_user_and_release(duration, z, lift_height, name):
    """
    Grasp a specified object, move it in front of the user at a fixed X/Y position with given forward distance and vertical lift, then release it.

Args:
    duration (number): Time (in seconds) allotted for the final move toward the user before releasing the object.
    z (number): Forward/backward offset (Z coordinate in Unity) from the user where the object will be delivered.
    lift_height (number): Vertical lift height (Y offset in Unity) used when moving the grasped object toward the user.
    name (string): Name or identifier of the object to grasp, move to the user, and release.
    """
    grasp_object(name, lift_height)
    move_to_position(-1.02, 1.6, z, duration, False)
    release_object(False, 0.1)

MACRO_MAP = {
    "deliver_object_to_user_and_release": deliver_object_to_user_and_release,
}