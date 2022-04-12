"""A file that contains parameter learning code"""


def load_data(model, inputs, settings, i):
    data = {}
    xTemp = torch.from_numpy(np.swapaxes(inputs["x"], 1, 0)).float()
    cTemp = torch.from_numpy(np.swapaxes(inputs["c"], 1, 0)).float()
    xTemp = xTemp.cuda()
    cTemp = cTemp.cuda()
    data["x"] = (xTemp, cTemp)
    data["outModel"] = settings["outModel"]
    return data
