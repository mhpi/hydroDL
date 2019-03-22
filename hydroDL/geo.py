"""
everything about geospatial corrdinates / locations / geometries ...
"""


class GeoObj(object):
    """overall object describes the geo-locations
    """

    def __init__(self, typeStr, **arg):
        pass


class GeoRaster(GeoObj):
    """ geospatial raster
    """

    def __init__(self, **arg):
        print(arg)


class GeoVector(GeoObj):
    """geospatial vector
    """

    def __init__(self, **arg):
        print('coming soon')
