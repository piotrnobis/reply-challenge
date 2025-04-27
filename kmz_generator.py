import simplekml
import zipfile
import io
import csv

import math

import xml.etree.ElementTree as ET
import zipfile
import os

""" def create_dji_flighthub_kmz(start_point, target_points, output_filename):
    # Determine Moving Altitude
    highest_altitude = max([p[2] for p in target_points] + [start_point[2]])
    moving_altitude = highest_altitude + 5  # 5 meters higher

    # Sort points based on horizontal distance to the current location
    from geopy.distance import geodesic

    current_location = start_point
    remaining_points = target_points.copy()
    ordered_points = []

    while remaining_points:
        next_point = min(remaining_points, key=lambda p: geodesic((current_location[0], current_location[1]), (p[0], p[1])).meters)
        ordered_points.append(next_point)
        current_location = next_point
        remaining_points.remove(next_point)

    # Start building KML
    kml = simplekml.Kml()

    flight_path = []

    # get the end point (offset by 0.5 meter in lat and long to barcode)
    # 1. Ascend to Moving Altitude from Start
    flight_path.append((start_point[0], start_point[1], moving_altitude))

    # 2. Visit each point
    for point in ordered_points:
        lat, lon, alt = offset_half_meter_toward_end(start_point[0], start_point[1], start_point[2], point[0], point[1], point[2])
        flight_path.append((lat, lon, alt))
        # Fly horizontally to above point at Moving Altitude
        #flight_path.append((point[0], point[1], moving_altitude))
        # Descend vertically to point's altitude
        #flight_path.append((point[0], point[1], point[2]))
        # Ascend back to Moving Altitude
        #flight_path.append((point[0], point[1], moving_altitude))

    # 3. Return to Start Point at Moving Altitude
    flight_path.append((start_point[0], start_point[1], moving_altitude))
    print(flight_path)

    kml = simplekml.Kml()
    linestring = kml.newlinestring(name="Path")
    linestring.coords = flight_path
    linestring.altitudemode = simplekml.AltitudeMode.absolute
    kml.savekmz("output.kmz") """

# Example usage:
# start = (52.5200, 13.4050, 50)  # Berlin, 50m altitude
# points = [
#     (52.5205, 13.4060, 40),
#     (52.5210, 13.4070, 45),
#     (52.5195, 13.4040, 42),
# ]
# create_dji_flighthub_kmz(start, points, 'flight.kmz')

# Location of the camera of the shot 130
start = (48.18934758333333, 11.50016086111111, 565.702)
csv_file_path = 'triangulated_selected_points_gps.csv'
points = []
# Open and read CSV line by line
with open(csv_file_path, mode='r', newline='') as csvfile:
    reader = csv.DictReader(csvfile)  # Reads rows as dictionaries (column name -> value)

    for row in reader:
        # Example: Access columns by their header names
        latitude = float(row['Latitude'])
        longitude = float(row['Longitude'])
        altitude = float(row['Altitude'])
        points.append((latitude, longitude, altitude))

waypoints = [
    {"name": "Waypoint 1", "lat": 48.18934758333333, "lon": 11.50016086111111, "alt": 565.702}
]

ctr = 2
for latitude, longitude, altitude in points:
    waypoints.append({"name": f"Waypoint {ctr}", "lat": latitude, "lon": longitude, "alt": altitude})
    ctr = ctr + 1


def create_kml(waypoints):
    """
    Build a KML ElementTree from a list of waypoints.
    Each waypoint is a dict with 'name', 'lat', 'lon', and optional 'alt'.
    """
    # Define KML namespace
    kml_ns = "http://www.opengis.net/kml/2.2"
    ET.register_namespace("", kml_ns)
    
    # Root element
    kml = ET.Element(f"{{{kml_ns}}}kml")
    doc = ET.SubElement(kml, "Document")
    
    # Add each waypoint as a Placemark
    for wp in waypoints:
        pm = ET.SubElement(doc, "Placemark")
        name = ET.SubElement(pm, "name")
        name.text = wp["name"]
        
        point = ET.SubElement(pm, "Point")
        coords = ET.SubElement(point, "coordinates")
        # KML uses lon,lat,altitude
        lat = wp["lat"]
        lon = wp["lon"]
        alt = wp.get("alt", 0)  # default to 0 if no altitude provided
        coords.text = f"{lon},{lat},{alt}"
    
    return ET.ElementTree(kml)

def write_kmz(kml_tree, kmz_filename, kml_filename="doc.kml"):
    """
    Write the ElementTree kml_tree to a KML file in memory
    and then compress it into a KMZ archive.
    """
    # Write KML to a temporary file
    temp_kml = kml_filename
    kml_tree.write(temp_kml, encoding="utf-8", xml_declaration=True)
    
    # Create KMZ (zip) and add the KML
    with zipfile.ZipFile(kmz_filename, 'w', compression=zipfile.ZIP_DEFLATED) as kmz:
        kmz.write(temp_kml, arcname=kml_filename)
    
    # Clean up temporary KML
    os.remove(temp_kml)
    print(f"KMZ file created: {kmz_filename}")

 # Generate KML tree
kml_tree = create_kml(waypoints)

# Output KMZ filename
output_kmz = "waypoints_with_altitude.kmz"

# Write out the KMZ
write_kmz(kml_tree, output_kmz)

# create_dji_flighthub_kmz(start, points, 'flight.kmz')