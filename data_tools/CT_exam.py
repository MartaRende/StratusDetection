import numpy as np
from matplotlib.path import Path
from pyproj import Transformer
from netCDF4 import Dataset, num2date
import folium
from folium.plugins import TimestampedGeoJson
import branca.colormap as cm
import random
import matplotlib.pyplot as plt
import os
from datetime import datetime as dt
import matplotlib.image as mpimg
import xarray as xr


def get_map(datetime, var="CT"):
    # 1. Define polygon and coordinate transformations
    polygon_points = [
         (46.15, 5.90),  # SW corner
        (46.15, 6.48),  # SE corner
        (46.55, 6.48),  # NE corner
        (46.55, 5.90)   # NW corner
    ]
    transformer_to_proj = Transformer.from_crs("EPSG:4326", "EPSG:21781", always_xy=True)
    transformer_to_latlon = Transformer.from_crs("EPSG:21781", "EPSG:4326", always_xy=True)

    # 2. Load NetCDF data for the given datetime
    dt_obj = dt.fromisoformat(datetime)
    year = dt_obj.year
    month = f"{dt_obj.month:02d}"
    day = f"{dt_obj.day:02d}"
    nc_path = f"/home/marta/Projects/tb/data/weather/inca/{year}/{year}{month}{day}.nc"
    nc = Dataset(nc_path)

    # 3. Prepare grid and mask for the polygon area
    x_vals = nc.variables['x'][:]
    y_vals = nc.variables['y'][:]
    xx, yy = np.meshgrid(x_vals, y_vals)
    points_xy = np.vstack([xx.ravel(), yy.ravel()]).T

    # Transform polygon to projected coordinates and create mask
    poly_xy = [transformer_to_proj.transform(lon, lat) for lat, lon in polygon_points]
    mask = Path(poly_xy).contains_points(points_xy).reshape(xx.shape)
    print(f"Mask created with {np.sum(mask)} points inside the polygon.")

    # 4. Get coordinates and time data from NetCDF
    lon_flat, lat_flat = transformer_to_latlon.transform(xx.ravel(), yy.ravel())
    time_var = nc.variables['datetime']
    datetimes = num2date(time_var[:], units=time_var.units)
    datetimes_str = np.array([dt.isoformat() for dt in datetimes])
    print(len(lon_flat), "points in the grid")

    # 5. Sample data for performance (reduce number of points)
    sample_rate = 4
    sample_indices = np.arange(len(lon_flat))[::sample_rate]
    print(f"Sampling {len(sample_indices)} points from {len(lon_flat)} total points.")
    ct_var = nc.variables[var][:]

    # 6. Prepare colormap based on data range
    valid_ct = ct_var[np.isfinite(ct_var)]  
    if len(valid_ct) == 0:
        raise ValueError("No valid data found in the dataset.")
    min_ct, max_ct = np.min(valid_ct), np.max(valid_ct)
    colormap = cm.LinearColormap(
        ['blue', 'green', 'yellow', 'red'],
        vmin=float(min_ct), 
        vmax=float(max_ct)  
    )

    # 7. Create GeoJSON features for folium map (animated)
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

    # 8. Create and configure folium map
    m = folium.Map(location=[np.mean(lat_flat), np.mean(lon_flat)], zoom_start=9, tiles='CartoDB positron')
    # Add polygon corner markers
    for i, (lat, lon) in enumerate(polygon_points):
        folium.Marker(
            location=[lat, lon],
            popup=f"Point {i+1}<br>Lat: {lat:.6f}<br>Lon: {lon:.6f}",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
    # Add polygon outline
    folium.PolyLine(
        locations=polygon_points + [polygon_points[0]],  # Close the polygon
        color='blue',
        weight=2,
        opacity=0.7,
        popup="Area"
    ).add_to(m)
    # Add colormap legend
    colormap.caption = f'{var} Values (%)' if var in nc.variables and hasattr(nc.variables[var], 'units') else f'{var} Values'
    colormap.width = 500
    colormap.add_to(m)
    # Add special coordinate marker
    special_lat = 46 + 25/60 + 29.3/3600
    special_lon = 6 + 5/60 + 56.9/3600
    folium.Marker(
        location=[special_lat, special_lon],
        popup="Special Point<br>Lat: {:.6f}<br>Lon: {:.6f}".format(special_lat, special_lon),
        icon=folium.Icon(color='green', icon='star')
    ).add_to(m)
    # Add animated GeoJSON layer
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

    # 9. Save static plot for the requested timestamp
    target_time_str = datetime
    output_dir = "analysis/single_timestamp_maps"
    os.makedirs(output_dir, exist_ok=True)
    # Find time index
    if target_time_str not in datetimes_str:
        raise ValueError(f"Timestamp {target_time_str} not found in data.")
    t = np.where(datetimes_str == target_time_str)[0][0]
    print(f"Found timestamp at index t = {t}")
    # Extract and mask data slice
    ct_slice = ct_var[t, :, :]
    ct_masked = np.where(mask, ct_slice, np.nan)
    # Transform grid for plotting
    lon, lat = transformer_to_latlon.transform(xx, yy)
    # Create matplotlib plot
    fig, ax = plt.subplots(figsize=(10, 8))
    mesh = ax.pcolormesh(lon, lat, ct_masked, 
                        cmap="coolwarm", 
                        vmin=min_ct, 
                        vmax=max_ct,
                        shading='auto')
    cbar = plt.colorbar(mesh, ax=ax, shrink=0.6)
    cbar.set_label(f"{var}(%)")
    # Draw polygon outline
    poly_lons = [p[1] for p in polygon_points] + [polygon_points[0][1]]
    poly_lats = [p[0] for p in polygon_points] + [polygon_points[0][0]]
    ax.plot(poly_lons, poly_lats, 
           color='blue', linewidth=2,
           linestyle='--', marker='o')
    # Set plot bounds with buffer
    poly_lons = [p[1] for p in polygon_points]
    poly_lats = [p[0] for p in polygon_points]
    buffer_factor = 0.10
    lon_min, lon_max = min(poly_lons), max(poly_lons)
    lat_min, lat_max = min(poly_lats), max(poly_lats)
    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min
    ax.set_xlim([
        lon_min - buffer_factor * lon_range,
        lon_max + buffer_factor * lon_range
    ])
    ax.set_ylim([
        lat_min - buffer_factor * lat_range,
        lat_max + buffer_factor * lat_range
    ])
    # Set axis ticks
    ax.xaxis.set_major_locator(plt.AutoLocator())
    ax.yaxis.set_major_locator(plt.AutoLocator())
    # Add special points
    dole_lat = 46 + 25/60 + 29.3/3600
    dole_lon = 6 + 5/60 + 56.9/3600
    ax.plot(dole_lon, dole_lat, 'k*', markersize=10, label='Dole Point')
    geneva_lat = 46.220473615
    geneva_lon = 6.132936441
    ax.plot(geneva_lon, geneva_lat, 'o', color='violet', markersize=10, label='Geneva Point')
    nyon_lat = 46.3789
    nyon_lon = 6.2390
    ax.plot(nyon_lon, nyon_lat, 'go', markersize=10, label='Nyon Point')
    # Configure plot
    ax.set_title(f"{var} Values - {datetimes[t].strftime('%Y-%m-%d %H:%M')}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend()
    # Build image path for timestamp
    dt_obj = dt.fromisoformat(datetime)
    year = dt_obj.year
    month = f"{dt_obj.month:02d}"
    day = f"{dt_obj.day:02d}"
    hour = f"{dt_obj.hour:02d}"
    minute = f"{dt_obj.minute:02d}"
    img_dir = f"/home/marta/Projects/tb/data/images/mch/1159/2/{year}/{month}/{day}"
    img_filename = f"1159_2_{year}-{month}-{day}_{hour}{minute}.jpeg"
    timestamp_img_path = os.path.join(img_dir, img_filename)
    print(f"Checking for timestamp image at: {timestamp_img_path}")
    # If image exists, add it below the plot
    if os.path.exists(timestamp_img_path):
        print(f"Timestamp image found: {timestamp_img_path}")
        img = mpimg.imread(timestamp_img_path)
        fig.subplots_adjust(bottom=0.3)
        img_ax = fig.add_axes([0.3, -0.37, 0.3, 0.35])
        img_ax.imshow(img)
        img_ax.set_title(f"Image: {datetimes[t].strftime('%Y-%m-%d %H:%M')}", fontsize=12)
        img_ax.axis('off')
        print("Image label added below plot.")
    else:
        print(f"Timestamp image not found: {timestamp_img_path}. Skipping image label.")
    # Add min/max values as text
    plt.figtext(0.15, 0.02, 
               f"Min: {np.nanmin(ct_masked):.1f} % | Max: {np.nanmax(ct_masked):.1f} %",
               fontsize=10, ha='center')
    # Set aspect ratio
    ax.set_aspect('equal')
    # Save static plot
    output_path = os.path.join(output_dir, f"{var}_{datetimes[t].strftime('%Y%m%d_%H%M')}.png")
    ax.set_aspect('equal', adjustable='datalim')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Map saved: {output_path}")

    # 10. Add title to folium map
    title_html = f'''
        <h3 align="center" style="font-size:16px"><b>{var} Values - {var}</b></h3>
    '''.format(datetimes[0].strftime('%Y-%m-%d'))
    m.get_root().html.add_child(folium.Element(title_html))
    # 11. Save folium map (currently commented out)
    # m.save(f"{var}_interactive_map_optimized.html")
    print(f"Interactive map saved as '{var}_interactive_map_optimized.html'")

if __name__ == "__main__":
    ds = xr.open_dataset("/home/marta/Projects/tb/data/weather/inca/2024/20241116.nc")
    # Example usage for different timestamps and variables
    get_map("2023-02-09T09:30:00", var="CT")
    get_map("2023-02-09T11:30:00", var="CT")
    get_map("2023-02-09T14:30:00", var="CT")
    get_map("2023-02-09T09:30:00", var="SU")
    get_map("2023-02-09T11:30:00", var="SU")
    get_map("2023-02-09T14:30:00", var="SU")
    print("Map generation completed successfully.")
