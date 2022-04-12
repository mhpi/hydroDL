"""A file that contains testing functions requiring permutation"""


def load_data(model, inputs, settings, i):
    """
    A testing function for models that require permutation
    :param model: the model you want to train
    :param kwargs: a keyword dictionary for any other passed params
    :return:
    """
    data = {}
    data["x"] = inputs["xTest"].permute(1, 2, 0)
    return results
