"""
:dataset: db.dataset is a container of data
"""


class Dataframe(object):
    def getGeo(self):
        return self.lat, self.lon

    def getT(self):
        return self.time


class DataModel():
    def getDataTrain(self):
        return self.x, self.y, self.c


from .dbCsv import DataframeCsv, DataModelCsv