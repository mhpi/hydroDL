"""A file that contains testing functions that may generate yss"""


def load_data(model, inputs, settings, i):
    """
    A testing function for models ANN and LSTM
    :param model: the model you want to train
    :param kwargs: a keyword dictionary for any other passed params
    :return: an array of results. If doMC is true, results are [yp
    """
    data = {}
    data["x"] = inputs["xTest"]
    return data
