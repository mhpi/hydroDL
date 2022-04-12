"""A file which contains reading functions"""


def get_real_path(file):
    """
    A function that gets the absolute path to a file you're looking for
    :param file: the file you want the path to
    :return: a str file path
    """

    import os

    dir_path = os.path.dirname(os.path.realpath(__file__))
    abs_path = dir_path and os.path.join(dir_path, file)
    return abs_path
