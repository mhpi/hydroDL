

import shapefile
from mpl_toolkits.basemap import Basemap, cm
import numpy as np
import matplotlib.pyplot as plt

hucShapeFile = '/mnt/sdc/Kuai/Map/HUC/HUC2_CONUS'
shape = shapefile.Reader(hucShapeFile)
feature = shape.shapes()[0]
crd = np.array(feature.points)
par = feature.parts

map = Basemap(llcrnrlat=25, urcrnrlat=50,
              llcrnrlon=-120, urcrnrlon=-70,
              projection='cyl', resolution='c')


if len(par) > 1:
    for k in range(0, len(par)-1):
        x = crd[par[k]:par[k+1], 0]
        y = crd[par[k]:par[k+1], 1]        
    map.plot(x, y, color='r', linewidth=3)
else:
    x = crd[:, 0]
    y = crd[:, 1]
    map.plot(x, y, color='r', linewidth=3)
plt.show()
