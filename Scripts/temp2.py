
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from osgeo import gdal
from numpy import linspace
from numpy import meshgrid

map = Basemap(projection='tmerc', 
              lat_0=0, lon_0=3,
              llcrnrlon=1.819757266426611, 
              llcrnrlat=41.583851612359275, 
              urcrnrlon=1.841589961763497, 
              urcrnrlat=41.598674173123)

ds = gdal.Open("../sample_files/dem.tiff")
data = ds.ReadAsArray()

x = linspace(0, map.urcrnrx, data.shape[1])
y = linspace(0, map.urcrnry, data.shape[0])

xx, yy = meshgrid(x, y)

map.contour(xx, yy, data)

plt.show()
