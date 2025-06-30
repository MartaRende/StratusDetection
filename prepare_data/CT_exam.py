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
nc = Dataset("/home/marta/Projects/tb/data/weather/inca/2024/20241116.nc")

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
# 6. Prepare CT data and colormap 
ct_var = nc.variables['SU'][:]

valid_ct = ct_var[np.isfinite(ct_var)]  

if len(valid_ct) == 0:
    raise ValueError("Nessun dato valido nella variabile SU")

min_ct, max_ct = np.min(valid_ct), np.max(valid_ct)
print(f"SU values range: {min_ct:.2f} to {max_ct:.2f}")
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
colormap.caption = 'SU Values (%)' if 'SU' in nc.variables and hasattr(nc.variables['SU'], 'units') else 'SU Values'
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
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from pyproj import Transformer
from netCDF4 import Dataset, num2date
import os

# [Keep all your initial setup code until the plotting part...]

# Timestamp to save
target_time_str = "2024-11-16T11:30:00"  # ISO 8601 format
output_dir = "analysis/single_timestamp_maps"
os.makedirs(output_dir, exist_ok=True)

# 1. Find the corresponding time index
if target_time_str not in datetimes_str:
    raise ValueError(f"Timestamp {target_time_str} not found in data.")

t = np.where(datetimes_str == target_time_str)[0][0]
print(f"✅ Found timestamp at index t = {t}")

# 2. Extract slice and apply mask
ct_slice = ct_var[t, :, :]
ct_masked = np.where(mask, ct_slice, np.nan)

# Transform coordinates for proper geographic plotting
transformer_to_latlon = Transformer.from_crs("EPSG:21781", "EPSG:4326", always_xy=True)
lon, lat = transformer_to_latlon.transform(xx, yy)

# 3. Create the plot with proper geographic scaling
fig, ax = plt.subplots(figsize=(10, 8))

# Plot the data using pcolormesh with transformed coordinates
mesh = ax.pcolormesh(lon, lat, ct_masked, 
                    cmap="coolwarm", 
                    vmin=min_ct, 
                    vmax=max_ct,
                    shading='auto')

# Add colorbar
cbar = plt.colorbar(mesh, ax=ax, shrink=0.6)
cbar.set_label("SU (%)")

# Add polygon outline
poly_lons = [p[1] for p in polygon_points] + [polygon_points[0][1]]
poly_lats = [p[0] for p in polygon_points] + [polygon_points[0][0]]
ax.plot(poly_lons, poly_lats, 
       color='blue', linewidth=2,
       linestyle='--', marker='o')
# After creating your plot but before saving, add:

# 1. Calculate automatic bounds from your polygon
poly_lons = [p[1] for p in polygon_points]  # Longitude values
poly_lats = [p[0] for p in polygon_points]  # Latitude values

# 2. Calculate min/max with 10% buffer
buffer_factor = 0.10  # 10% buffer
lon_min, lon_max = min(poly_lons), max(poly_lons)
lat_min, lat_max = min(poly_lats), max(poly_lats)

lon_range = lon_max - lon_min
lat_range = lat_max - lat_min

# 3. Set automatic limits with buffer
ax.set_xlim([
    lon_min - buffer_factor * lon_range,
    lon_max + buffer_factor * lon_range
])
ax.set_ylim([
    lat_min - buffer_factor * lat_range,
    lat_max + buffer_factor * lat_range
])

# 4. Set smart ticks (automatically spaced)
ax.xaxis.set_major_locator(plt.AutoLocator())
ax.yaxis.set_major_locator(plt.AutoLocator())


# Add special point
dole_lat = 46 + 25/60 + 29.3/3600
dole_lon = 6 + 5/60 + 56.9/3600
ax.plot(dole_lon, dole_lat, 'k*', markersize=10, label='Dole Point')
geneva_lat = 46.220473615
geneva_lon = 6.132936441
ax.plot(geneva_lon, geneva_lat, 'yo', markersize=10, label='Geneva Point')
nyon_lat = 46.3789
nyon_lon = 6.2390
ax.plot(nyon_lon, nyon_lat, 'go', markersize=10, label='Nyon Point')
# Configure plot
ax.set_title(f"SU Values - {datetimes[t].strftime('%Y-%m-%d %H:%M')}")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.grid(True, linestyle=':', alpha=0.7)
ax.legend()
timestamp_img_path = os.path.join("/home/marta/Projects/tb/data/images/mch/1159/2/2024/11/16", "1159_2_2024-11-16_1130.jpeg")
print(f"Checking for timestamp image at: {timestamp_img_path}")
import matplotlib.image as mpimg

if os.path.exists(timestamp_img_path):
    print(f"✅ Timestamp image found: {timestamp_img_path}")
    img = mpimg.imread(timestamp_img_path)

    # Imposta spazio per l'immagine sotto il grafico
    fig.subplots_adjust(bottom=0.3)  # Lascia spazio sotto l'axes principale

    # Aggiungi un nuovo axes piccolo in basso, fuori dal grafico principale
    img_ax = fig.add_axes([0.3, -0.37, 0.3, 0.35])  # [left, bottom, width, height]
    img_ax.imshow(img)
    img_ax.set_title(f"Image: {datetimes[t].strftime('%Y-%m-%d %H:%M')}", fontsize=12)
    img_ax.axis('off')
    
    print("✅ Image label added below plot.")
else:
    print(f"⚠️ Timestamp image not found: {timestamp_img_path}. Skipping image label.")

# Add min/max values
plt.figtext(0.15, 0.02, 
           f"Min: {np.nanmin(ct_masked):.1f} % | Max: {np.nanmax(ct_masked):.1f} %",
           fontsize=10, ha='center')

# Set equal aspect ratio
ax.set_aspect('equal')

# 4. Save
output_path = os.path.join(output_dir, f"su_{datetimes[t].strftime('%Y%m%d_%H%M')}.png")
ax.set_aspect('equal', adjustable='datalim')  # Better auto-adjustment
plt.tight_layout()
plt.savefig(output_path, dpi=200, bbox_inches='tight')
plt.close()

print(f"✅ Map saved: {output_path}")
# 9. Add title
title_html = '''
    <h3 align="center" style="font-size:16px"><b>SU Values - {}</b></h3>
'''.format(datetimes[0].strftime('%Y-%m-%d'))
m.get_root().html.add_child(folium.Element(title_html))

# 10. Save map
m.save("ct_interactive_map_optimized.html")
print("✅ Interactive map saved as 'ct_interactive_map_optimized.html'")
