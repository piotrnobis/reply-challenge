import simplekml
import zipfile
import io
import csv
import math

import math

def offset_half_meter_toward_end(lat1, lon1, alt1, lat2, lon2, alt2):
    # Calculate direction vector (normalized)
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1
    d_alt = alt2 - alt1

    length = math.sqrt(d_lat**2 + d_lon**2 + d_alt**2)
    if length == 0:
        raise ValueError("Start and end points are the same!")

    dir_lat = d_lat / length
    dir_lon = d_lon / length
    dir_alt = d_alt / length

    # Meters to degrees
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * math.cos(math.radians(lat1))

    # How much degree offset is 0.5 meters
    offset_lat_deg = 0.5 / meters_per_deg_lat
    offset_lon_deg = 0.5 / meters_per_deg_lon

    # Move 0.5m in lat and 0.5m in lon toward the end point
    new_lat = lat1 + offset_lat_deg * math.copysign(1, dir_lat)
    new_lon = lon1 + offset_lon_deg * math.copysign(1, dir_lon)
    new_alt = alt1 + (0.5 * dir_alt)  # Optional: you can move half a meter in alt too

    return new_lat, new_lon, alt2 + 1


def create_dji_flighthub_kmz(start_point, target_points, output_filename):
    """
    Create a KMZ file for DJI FlightHub 2 based on start point and target points.

    Parameters:
    - start_point: tuple (latitude, longitude, altitude)
    - target_points: list of tuples [(latitude, longitude, altitude), ...]
    - output_filename: string, filename for the resulting KMZ file (e.g., 'flight.kmz')
    """
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
    kml.savekmz("output.kmz")

# Example usage:
# start = (52.5200, 13.4050, 50)  # Berlin, 50m altitude
# points = [
#     (52.5205, 13.4060, 40),
#     (52.5210, 13.4070, 45),
#     (52.5195, 13.4040, 42),
# ]
# create_dji_flighthub_kmz(start, points, 'flight.kmz')

# Location of the camera of the shot 130
start = (48.18934758333333, 11.50016086111111, 565.702)  # Berlin, 50m altitude
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

create_dji_flighthub_kmz(start, points, 'flight.kmz')