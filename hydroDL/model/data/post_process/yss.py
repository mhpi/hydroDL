import numpy as np


def post_process(results, inputs, settings):
    if settings["doMC"] is not False:
        ySS = np.zeros(results["yP"].shape)
        yPnp = results["yP"].detach().cpu().numpy()
        for k in range(doMC):
            # log.info(k)
            yMC = model(inputs["xTest"], doDropMC=True).detach().cpu().numpy()
            ySS = ySS + np.square(yMC - yPnp)
        results["ySS"] = np.sqrt(ySS) / settings["doMC"]
    return results
