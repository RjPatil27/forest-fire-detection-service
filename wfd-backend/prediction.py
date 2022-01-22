
from folium import plugins
import folium
import ee
# Trigger the authentication flow.
ee.Authenticate()

# Initialize the library.
ee.Initialize()

a = input()
b = input()


def getNDVI(image):
    return image.normalizedDifference(['B5', 'B4'])


point = ee.Geometry.Point([-122.292, 40.903016])
l8 = ee.ImageCollection('LANDSAT/LC08/C01/T1_TOA')
#image1 =ee.Image('LANDSAT/LT05/C01/T1_TOA/LT05_044034_19900604')
image1 = ee.Image(
    l8.filterBounds(point)
    .filterDate('2021-01-01', '2021-11-30')
    .sort('CLOUD_COVER')
    .first()
)
# Compute NDVI
ndvi1 = getNDVI(image1)
#ndviParams = {min: -1, max: 1, 'palette': ['white', 'green']};
# ndviParams = {
# 'palette': [
#  'FFFFFF', 'CE7E45', 'DF923D', 'F1B555', 'FCD163', '99B718', '74A901',
# '66A000', '529400', '3E8601', '207401', '056201', '004C00', '023B01',
#   '012E01', '011D01', '011301'
#  ],}

ndviParams = {'palette': ['#d43027', '#f46d43', '#fdae61',
                          '#fee08b', '#d9efeb', '#a6d96a', '#66bd63', '#1a9850', '#006d2c']}

# Import the Folium library.
# Define a method for displaying Earth Engine image tiles to folium map.


def add_ee_layer(self, ee_image_object, NONE, name):
    map_id_dict = ee.Image(ee_image_object).getMapId(NONE)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
        name=name,
        overlay=True,
        control=True
    ).add_to(self)


roi = ee.Geometry.Point([-122.292, 40.903016]).buffer(30000)
ndvi_masked = ndvi1.updateMask(ndvi1.gte(0))
# Add EE drawing method to folium.
folium.Map.add_ee_layer = add_ee_layer

# Create a folium map object.
my_map = folium.Map(location=[40.903016, -122.292], zoom_start=12)

# Add the elevation model to the map object.
#my_map.add_ee_layer(dem.updateMask(dem.gt(0)), vis_params, 'DEM')

# Add the llayer to the map object
my_map.add_ee_layer(ndvi_masked.clip(roi), ndviParams, 'NDVI')
folium.Marker(location=[40.903016, -122.292], popup='camera 1',
              icon=folium.Icon(color='red')).add_to(my_map)
folium.Marker(location=[40.903016, -123.156798],
              popup='camera 2 ').add_to(my_map)
folium.Marker(location=[39.901461, -123.157786],
              popup='camera 3 ').add_to(my_map)
folium.Marker(location=[40.380579, -122.187804],
              popup='camera 4 ').add_to(my_map)
folium.Marker(location=[39.688039, -121.987359],
              popup='camera 5 ').add_to(my_map)
folium.Marker(location=[40.337052, -123.733571],
              popup='camera 6 ').add_to(my_map)


#my_map.add_child(folium.CircleMarker(location=[40.901461, -123.157786], radius=100))

# Add a layer control panel to the map.
my_map.add_child(folium.LayerControl())

# Display the map.
display(my_map)

# Mask the non-watery parts of the image, where NDVI < 0.4.
ndvi_masked = ndvi1.updateMask(ndvi1.gte(0.7))

# Define a map centered on San Francisco Bay.
map_ndvi_masked = folium.Map(location=[40.903016, -123.156798], zoom_start=10)
# ndvi_viz = { 'palette': ['white','#bae4b3', '#74c476','#006d2c']};
ndvi_viz = {'palette': ['red']}
# Add the image layer to the map and display it.
map_ndvi_masked.add_ee_layer(ndvi_masked, ndvi_viz, 'NDVI masked')
display(my_map)
display(map_ndvi_masked)


def calculate_ndvi():
    ee.Authenticate()
    # Initialize the library.
    ee.Initialize()