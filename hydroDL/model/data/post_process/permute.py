import numpy as np


def post_process(results, inputs, settings):
    results["yP"] = results["yP"].permute(2, 0, 1)
    return results
