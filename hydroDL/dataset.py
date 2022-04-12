"""
Define 
"""


class geoObj(object):
    def __init__(self):
        pass


class Dataset(object):
    def __init__(self):
        pass

    def __repr__(self):
        return "later"


class DatasetRaster(Dataset):
    def __init__(self):
        pass

    def __repr__(self):
        return "later"


class DatasetVector(Dataset):
    def __init__(self):
        pass

    def __repr__(self):
        return "later"


class Dataframe(object):
    def getGeo(self):
        return self.lat, self.lon

    def getT(self):
        return self.time


class DataModel:
    def getDataTrain(self):
        return self.x, self.y, self.c
