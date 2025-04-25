from PIL import Image
from PIL.ExifTags import TAGS

class Pic:
    def __init__(self, image_path):
        self.image_path = image_path
        self.exif_data = self._load_exif_data()

    def _load_exif_data(self):
        """Load EXIF data from the image and store it in a dictionary."""
        image = Image.open(self.image_path)
        exif_data = image._getexif()
        exif_dict = {}

        if exif_data:
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)  # Get the human-readable tag name
                exif_dict[tag] = value

        # Process GPSInfo to decimal coordinates
        if "GPSInfo" in exif_dict:
            gps_data = self._process_gps(exif_dict["GPSInfo"])
            exif_dict["GPSInfo"] = gps_data

        return exif_dict
    
    def _process_gps(self, gps_info):
        """Process GPS data to decimal format (latitude, longitude, altitude)."""
        gps_data = {}

        if 2 in gps_info and 4 in gps_info:
            # Latitude and Longitude
            gps_data['latitude'] = float(self._dms_to_decimal(gps_info[2], gps_info[1]))
            gps_data['longitude'] = float(self._dms_to_decimal(gps_info[4], gps_info[3]))

        if 6 in gps_info:
            # Altitude
            gps_data['altitude'] = gps_info[6]

        return gps_data
    
    def get_metadata(self):
        return self.exif_data

    def _dms_to_decimal(self, dms, ref):
        """Convert DMS (Degrees, Minutes, Seconds) to Decimal Degrees."""
        degrees, minutes, seconds = dms
        decimal = degrees + (minutes / 60) + (seconds / 3600)
        if ref in ['S', 'W']:
            decimal = -decimal
        return decimal

    def __getitem__(self, key):
        """Allow access to EXIF data like a dictionary."""
        return self.exif_data.get(key, None)

    def __repr__(self):
        """Represent the Pic instance as a string."""
        return f"<Pic: {self.image_path}, EXIF data loaded>"