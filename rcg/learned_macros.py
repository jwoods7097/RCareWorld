def move_and_grasp_banana_object(name):
    move_to_object(name=name)
    grasp_object(name=name)

MACRO_MAP = {
    'move_and_grasp_banana_object': move_and_grasp_banana_object,
}