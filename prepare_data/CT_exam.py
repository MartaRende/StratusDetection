import numpy as np
from matplotlib.path import Path
from pyproj import Transformer
from netCDF4 import Dataset, num2date
import folium
from folium.plugins import TimestampedGeoJson
import branca.colormap as cm
import random

# 1. Define polygon and transformations
polygon_points = [
     (46.15, 5.90),  # Sud-Ovest 
    (46.15, 6.48),  # Sud-Est 
    (46.55, 6.48),  # Nord-Est
    (46.55, 5.90)   #  # NW corner
]

transformer_to_proj = Transformer.from_crs("EPSG:4326", "EPSG:21781", always_xy=True)
transformer_to_latlon = Transformer.from_crs("EPSG:21781", "EPSG:4326", always_xy=True)

# 2. Load NetCDF data
nc = Dataset("/home/marta/Projects/tb/data/weather/inca/2024/20241111.nc")

# 3. Prepare grid and mask
x_vals = nc.variables['x'][:]
y_vals = nc.variables['y'][:]
xx, yy = np.meshgrid(x_vals, y_vals)
points_xy = np.vstack([xx.ravel(), yy.ravel()]).T

# Transform polygon and create mask
poly_xy = [transformer_to_proj.transform(lon, lat) for lat, lon in polygon_points]
mask = Path(poly_xy).contains_points(points_xy).reshape(xx.shape)
print(f"Mask created with {np.sum(mask)} points inside the polygon.")
# 4. Get coordinates and time data
lon_flat, lat_flat = transformer_to_latlon.transform(xx.ravel(), yy.ravel())
time_var = nc.variables['datetime']
datetimes = num2date(time_var[:], units=time_var.units)
datetimes_str = np.array([dt.isoformat() for dt in datetimes])
print(len(lon_flat), "points in the grid")

# 5. Sample data for better performance
sample_rate = 4
sample_indices = np.arange(len(lon_flat))[::sample_rate]

print(f"Sampling {len(sample_indices)} points from {len(lon_flat)} total points.")
# 6. Prepare SU data and colormap 
ct_var = nc.variables['SU'][:]

valid_ct = ct_var[np.isfinite(ct_var)]  

if len(valid_ct) == 0:
    raise ValueError("Nessun dato valido nella variabile SU")

min_ct, max_ct = np.min(valid_ct), np.max(valid_ct)

colormap = cm.LinearColormap(
    ['blue', 'green', 'yellow', 'red'],
    vmin=float(min_ct), 
    vmax=float(max_ct)  
)

# 7. Create features
features = []
time_step = 2

for t in range(0, ct_var.shape[0], time_step):
    ct_slice = ct_var[t, :, :].ravel()
    
    for i in sample_indices:
        val = ct_slice[i]
        if mask.ravel()[i] and np.isfinite(val):  
            try:
                color = colormap(val)
                features.append({
                    'type': 'Feature',
                    'geometry': {'type': 'Point', 'coordinates': [lon_flat[i], lat_flat[i]]},
                    'properties': {
                        'time': datetimes_str[t],
                        'style': {'color': 'black'},
                        'icon': 'circle',
                        'iconstyle': {
                            'fillColor': color,
                            'fillOpacity': 0.7,
                            'stroke': True,
                            'radius': 7,
                            'color': 'black',
                            'weight': 1
                        },
                        'value': float(val)
                    }
                })
            except ValueError as e:
                print(f"Warning:  {e} for value {val} at index {i}, skipping point.")
                continue
# 8. Create and configure map
m = folium.Map(location=[np.mean(lat_flat), np.mean(lon_flat)], zoom_start=9, tiles='CartoDB positron')
for i, (lat, lon) in enumerate(polygon_points):
    folium.Marker(
        location=[lat, lon],
        popup=f"Point {i+1}<br>Lat: {lat:.6f}<br>Lon: {lon:.6f}",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)

folium.PolyLine(
    locations=polygon_points + [polygon_points[0]],  # Chiudi il poligono tornando al primo punto
    color='blue',
    weight=2,
    opacity=0.7,
    popup="Area"
).add_to(m)
# Add colormap with larger font
colormap.caption = 'SU Values (°C)' if 'SU' in nc.variables and hasattr(nc.variables['SU'], 'units') else 'SU Values'
colormap.width = 500
colormap.add_to(m)
# Add the specific coordinate point (46°25'29.3"N 6°05'56.9"E)
special_lat = 46 + 25/60 + 29.3/3600  # 46.42480555555556
special_lon = 6 + 5/60 + 56.9/3600    # 6.099138888888889

folium.Marker(
    location=[special_lat, special_lon],
    popup="Special Point<br>Lat: {:.6f}<br>Lon: {:.6f}".format(special_lat, special_lon),
    icon=folium.Icon(color='green', icon='star')
).add_to(m)
TimestampedGeoJson(
    {'type': 'FeatureCollection', 'features': features},
    period='PT1H',
    duration='PT30M',
    auto_play=True,
    loop=True,
    max_speed=5,
    loop_button=True,
    date_options='HH:mm',
    time_slider_drag_update=True,
    transition_time=300
).add_to(m)

# 9. Add title
title_html = '''
    <h3 align="center" style="font-size:16px"><b>CT Values - {}</b></h3>
'''.format(datetimes[0].strftime('%Y-%m-%d'))
m.get_root().html.add_child(folium.Element(title_html))

# 10. Save map
m.save("ct_interactive_map_optimized.html")
print("Optimized map saved as ct_interactive_map_optimized.html")