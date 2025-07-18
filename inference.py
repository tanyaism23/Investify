import pickle
import traceback
import pandas as pd
from flask import Flask, request, jsonify,current_app
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import warnings
import requests
import numpy as np
import logging
from typing import Dict, Tuple
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import unary_union
import requests
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Optional
import json
import shap
import requests
import pandas as pd
import numpy as np
import osmnx as ox
from datetime import datetime
import logging
from typing import Dict, List, Optional, Union
import time
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

warnings.filterwarnings('ignore')



# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define API keys and endpoints
OPENWEATHER_API_KEY = "9cdf050abe7b42592268c0bf78c0195a"
GBIF_API_BASE = "https://api.gbif.org/v1/occurrence/search"

# Add rate limiting parameters
REQUEST_DELAY = 1  # Delay between API requests in seconds

def fetch_climate_data(lat: float, lon: float) -> Optional[Dict]:
    """
    Fetch climate data using OpenWeatherMap API with improved precipitation handling.
    """
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        time.sleep(REQUEST_DELAY)
        
        # Improved precipitation handling
        precipitation = 0
        if "rain" in data:
            precipitation = data["rain"].get("1h", 0) or data["rain"].get("3h", 0)
        elif "snow" in data:
            precipitation = data["snow"].get("1h", 0) or data["snow"].get("3h", 0)
        
        return {
            "temperature": data["main"]["temp"],
            "precipitation": precipitation,
            "humidity": data["main"]["humidity"],
        }
    except Exception as e:
        logger.error(f"Error fetching climate data for {lat}, {lon}: {str(e)}")
        return None

def fetch_biodiversity_data(lat: float, lon: float) -> int:
    """
    Fetch biodiversity data using GBIF API with improved species counting.
    """
    try:
        params = {
            "decimalLatitude": f"{lat-0.5},{lat+0.5}",
            "decimalLongitude": f"{lon-0.5},{lon+0.5}",
            "limit": 300,  # Increased limit
            "hasCoordinate": True,
            "hasGeospatialIssue": False
        }
        response = requests.get(GBIF_API_BASE, params=params)
        response.raise_for_status()
        data = response.json()
        time.sleep(REQUEST_DELAY)
        
        # Count unique species
        species_set = set()
        for record in data.get("results", []):
            if record.get("species"):
                species_set.add(record["species"])
        
        return len(species_set)
    except Exception as e:
        logger.error(f"Error fetching biodiversity data for {lat}, {lon}: {str(e)}")
        return 0

def fetch_green_cover_data(lat: float, lon: float) -> float:
    """
    Fetch green cover data with improved NDVI calculation.
    """
    try:
        # Create a larger area for analysis
        dist = 1000  # increased from 500 to 1000 meters
        tags = {
            'landuse': ['forest', 'grass', 'park', 'meadow', 'recreation_ground'],
            'natural': ['wood', 'grassland', 'heath']
        }
        
        area = ox.geometries_from_point((lat, lon), tags=tags, dist=dist)
        
        if not area.empty:
            total_area = np.pi * (dist ** 2)  # Total circular area
            green_area = area.geometry.area.sum()
            ndvi_proxy = green_area / total_area
            return max(min(ndvi_proxy, 1), 0)
        return 0.0
    except Exception as e:
        logger.error(f"Error fetching green cover data for {lat}, {lon}: {str(e)}")
        return 0.0

def fetch_land_usage_data(lat: float, lon: float) -> float:
    """
    Fetch land usage data with improved urban density calculation.
    """
    try:
        dist = 1000  # Analysis radius in meters
        
        # Get both buildings and roads
        building_tags = {'building': True}
        buildings = ox.geometries_from_point((lat, lon), tags=building_tags, dist=dist)
        
        graph = ox.graph_from_point((lat, lon), dist=dist, network_type='all')
        
        total_area = np.pi * (dist ** 2)
        building_area = buildings.geometry.area.sum() if not buildings.empty else 0
        road_length = sum(d['length'] for u, v, d in graph.edges(data=True))
        
        # Combine building coverage and road density for urban usage metric
        urban_density = (building_area / total_area) + (road_length / (dist * 2 * np.pi))
        return max(min(urban_density, 1), 0)
    except Exception as e:
        logger.error(f"Error fetching land usage data for {lat}, {lon}: {str(e)}")
        return 0.0

def fetch_water_coverage_data(lat: float, lon: float) -> float:
    """
    Fetch water coverage data with improved calculation.
    """
    try:
        dist = 1000  # Analysis radius in meters
        water_tags = {
            'natural': ['water', 'wetland'],
            'water': True,
            'waterway': ['river', 'canal', 'stream']
        }
        
        water_features = ox.geometries_from_place((lat, lon), tags=water_tags, dist=dist)
        
        if not water_features.empty:
            total_area = np.pi * (dist ** 2)
            water_area = water_features.geometry.area.sum()
            water_coverage = water_area / total_area
            return max(min(water_coverage, 1), 0)
        return 0.0
    except Exception as e:
        logger.error(f"Error fetching water coverage data for {lat}, {lon}: {str(e)}")
        return 0.0

def generate_risk_score(
    ndvi: float,
    species_richness: int,
    urban_land_usage: float,
    water_coverage: float,
    land_use_type: str
) -> float:
    """
    Calculate a risk score based on environmental factors.
    
    Args:
        ndvi (float): Normalized Difference Vegetation Index
        species_richness (int): Number of species in area
        urban_land_usage (float): Urban density metric
        water_coverage (float): Water coverage ratio
        land_use_type (str): Type of land use
        
    Returns:
        float: Risk score between 0 and 1
    """
    land_use_weight = {
        "green-based use": 0.1,
        "agricultural use": 0.2,
        "urban home-type use": 0.3,
        "commercial/industrial use": 0.4
    }

    try:
        risk_score = (
            (1 - ndvi) * 0.35 +
            (species_richness / 100) * 0.25 +
            (urban_land_usage / 100) * 0.25 +
            (1 - water_coverage) * 0.15
        )
        
        risk_score *= land_use_weight.get(land_use_type, 0.25)
        return min(max(risk_score, 0), 1)
    except Exception as e:
        logger.error(f"Error generating risk score: {str(e)}")
        return 0.5

def create_dataset(
    regions: List[Dict[str, float]],
    land_use_types: List[str]
) -> pd.DataFrame:
    """
    Create a labeled dataset across various regions with selected land use types.
    
    Args:
        regions (list): List of dictionaries containing lat/lon coordinates
        land_use_types (list): List of land use type strings
        
    Returns:
        pandas.DataFrame: Dataset containing environmental metrics and risk scores
    """
    data = []
    total_regions = len(regions)
    
    for idx, region in enumerate(regions, 1):
        try:
            lat, lon = region["lat"], region["lon"]
            logger.info(f"Processing region {idx}/{total_regions}: {lat}, {lon}")
            
            climate_data = fetch_climate_data(lat, lon)
            biodiversity_data = fetch_biodiversity_data(lat, lon)
            green_cover = fetch_green_cover_data(lat, lon)
            land_usage = fetch_land_usage_data(lat, lon)
            water_coverage = fetch_water_coverage_data(lat, lon)
            
            if climate_data:
                for land_use_type in land_use_types:
                    risk_score = generate_risk_score(
                        green_cover,
                        biodiversity_data,
                        land_usage,
                        water_coverage,
                        land_use_type
                    )
                    
                    data.append({
                        "latitude": lat,
                        "longitude": lon,
                        "temperature": climate_data["temperature"],
                        "precipitation": climate_data["precipitation"],
                        "humidity": climate_data["humidity"],
                        "species_richness": biodiversity_data,
                        "ndvi": green_cover,
                        "urban_land_usage": land_usage,
                        "water_coverage": water_coverage,
                        "land_use_type": land_use_type,
                        "risk_score": risk_score,
                        # "timestamp": datetime.now().isoformat()
                    })
            
        except Exception as e:
            logger.error(f"Error processing region {lat}, {lon}: {str(e)}")
            continue
    
    return pd.DataFrame(data)



regions = [
        {"lat": 10.7449, "lon": 92.5000},  # New York City, USA
        # {"lat": -33.8688, "lon": 151.2093},  # Sydney, Australia
        # {"lat": 51.5074, "lon": -0.1278},   # London, UK
        # {"lat": -1.286389, "lon": 36.817223},  # Nairobi, Kenya
        # {"lat": 28.6139, "lon": 77.2090}    # New Delhi, India
    ]

    # Define possible land use types
land_use_types = [
        "green-based use",
        "agricultural use",
        "urban home-type use",
        "commercial/industrial use"
    ]

dataset = create_dataset(regions, land_use_types)

# #change this shit
# @app.route('/analyze', methods=['POST'])
# def process_data_as_json():
#     try:
#         # Parse JSON data from request
#         cache = request.get_json()
        
#         return jsonify(result), 200
#     except Exception as e:
#         return jsonify({"status": "error", "message": str(e)}), 400

# # cache = {'latitude':11.96,
# #         'longitude':75.92,
# #         'use_case_type':'Agricultural'}

with open(f'/Users/samarthjindal/Desktop/Agglo-Ecopolis/src/models2-4.pkl', 'rb') as f:
    loaded_models = pickle.load(f)

with open(f'/Users/samarthjindal/Desktop/Agglo-Ecopolis/src/ensemble_weights2.pkl', 'rb') as f:
    loaded_weights = pickle.load(f)
    
with open(f'/Users/samarthjindal/Desktop/Agglo-Ecopolis/src/preprocessors2-4.pkl', 'rb') as f:
    loaded_encoders, loaded_scaler = pickle.load(f)

# latitude = cache['latitude']
# longitude = cache['longitude']
# use_case_type = cache['use_case_type']

def fetch_climate_data(lat: float, lon: float) -> Optional[Dict]:
    """
    Fetch climate data using OpenWeatherMap API with 5-day precipitation forecast.
    """
    try:
        # Get current weather
        current_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        current_response = requests.get(current_url)
        current_response.raise_for_status()
        current_data = current_response.json()
        
        # Get 5 day forecast with 3-hour steps
        forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        forecast_response = requests.get(forecast_url)
        forecast_response.raise_for_status()
        forecast_data = forecast_response.json()
        
        # Calculate average precipitation from forecast
        total_precipitation = 0
        count = 0
        
        for item in forecast_data.get('list', []):
            # Get precipitation (rain or snow)
            rain_amount = item.get('rain', {}).get('3h', 0)
            snow_amount = item.get('snow', {}).get('3h', 0)
            total_precipitation += rain_amount + snow_amount
            count += 1
        
        # Convert 3-hourly precipitation to daily average
        avg_daily_precipitation = (total_precipitation / count) * 8 if count > 0 else 0
        # Estimate monthly precipitation (multiply by 30 days)
        estimated_monthly_precipitation = avg_daily_precipitation * 30
        
        return {
            "temperature": current_data["main"]["temp"],
            "precipitation": estimated_monthly_precipitation,
            "humidity": current_data["main"]["humidity"],
        }
        
    except Exception as e:
        logger.error(f"Error fetching climate data for {lat}, {lon}: {str(e)}")
        return None
    finally:
        time.sleep(REQUEST_DELAY)

def _get_overpass_data(query: str) -> Dict:
    """
    Helper function to fetch data from Overpass API.
    """
    overpass_url = "https://overpass-api.de/api/interpreter"
    try:
        response = requests.post(overpass_url, data={'data': query})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        # logger.error(f"Overpass API error: {str(e)}")
        return {'elements': []}

def fetch_land_usage_data(lat: float, lon: float) -> float:
    """
    Fetch land usage data using Overpass API.
    Returns urban density index between 0 and 1.
    """
    try:
        # Calculate bounding box (1km radius)
        radius = 1000  # meters
        deg_radius = radius / 111320  # Convert meters to degrees (approximate)
        
        # Overpass query for buildings and roads
        query = f"""
        [out:json][timeout:25];
        (
          way["building"](around:{radius},{lat},{lon});
          way["highway"](around:{radius},{lat},{lon});
        );
        out body geom;
        """
        
        data = _get_overpass_data(query)
        
        if not data['elements']:
            # Fallback to secondary API
            return _fetch_land_usage_fallback(lat, lon)
        
        # Calculate areas and lengths
        total_area = np.pi * (radius ** 2)  # Total circular area in square meters
        building_area = 0
        road_length = 0
        
        for element in data['elements']:
            if 'geometry' in element:
                coords = [(p['lon'], p['lat']) for p in element['geometry']]
                if element.get('tags', {}).get('building'):
                    # Calculate building area
                    if len(coords) >= 3:
                        try:
                            polygon = Polygon(coords)
                            building_area += polygon.area * 111320 * 111320  # Convert to square meters
                        except:
                            continue
                elif element.get('tags', {}).get('highway'):
                    # Calculate road length
                    if len(coords) >= 2:
                        try:
                            line = LineString(coords)
                            road_length += line.length * 111320  # Convert to meters
                        except:
                            continue
        
        # Calculate urban density
        building_density = min(building_area / total_area, 0.7)  # Cap at 70%
        road_density = min(road_length / (radius * 2 * np.pi), 0.3)  # Cap at 30%
        
        urban_density = building_density + road_density
        return min(max(urban_density, 0), 1)
    
    except Exception as e:
        # logger.error(f"Error in primary land usage calculation: {str(e)}")
        return _fetch_land_usage_fallback(lat, lon)

def _fetch_land_usage_fallback(lat: float, lon: float) -> float:
    """
    Fallback method using OpenStreetMap Nominatim API for land use data.
    """
    try:
        # Use Nominatim API to get area details
        nominatim_url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json&zoom=14"
        headers = {'User-Agent': 'Urban Density Calculator 1.0'}
        
        response = requests.get(nominatim_url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        # Analyze address components and category
        address = data.get('address', {})
        category = data.get('category', '')
        
        # Calculate urban density based on location type
        if any(k in address for k in ['city', 'town', 'suburb']):
            return 0.8  # Urban area
        elif 'village' in address:
            return 0.4  # Rural settlement
        elif any(k in address for k in ['industrial', 'commercial']):
            return 0.9  # Industrial/commercial area
        elif any(k in address for k in ['forest', 'park', 'nature_reserve']):
            return 0.1  # Natural area
        else:
            return 0.5  # Default semi-urban
            
    except Exception as e:
        # logger.error(f"Error in fallback land usage calculation: {str(e)}")
        return 0.5  # Default value

def fetch_water_coverage_data(lat: float, lon: float) -> float:
    """
    Fetch water coverage data using Overpass API.
    Returns water coverage ratio between 0 and 1.
    """
    try:
        # Calculate bounding box (1km radius)
        radius = 1000  # meters
        
        # Overpass query for water features
        query = f"""
        [out:json][timeout:25];
        (
          way["natural"="water"](around:{radius},{lat},{lon});
          way["waterway"](around:{radius},{lat},{lon});
          way["water"](around:{radius},{lat},{lon});
          way["natural"="wetland"](around:{radius},{lat},{lon});
        );
        out body geom;
        """
        
        data = _get_overpass_data(query)
        
        if not data['elements']:
            # Fallback to secondary API
            return _fetch_water_coverage_fallback(lat, lon)
        
        # Calculate areas
        total_area = np.pi * (radius ** 2)  # Total circular area in square meters
        water_area = 0
        
        for element in data['elements']:
            if 'geometry' in element:
                coords = [(p['lon'], p['lat']) for p in element['geometry']]
                if len(coords) >= 3:
                    try:
                        polygon = Polygon(coords)
                        water_area += polygon.area * 111320 * 111320  # Convert to square meters
                    except:
                        continue
        
        water_coverage = water_area / total_area
        return min(max(water_coverage, 0), 1)
    
    except Exception as e:
        # logger.error(f"Error in primary water coverage calculation: {str(e)}")
        return _fetch_water_coverage_fallback(lat, lon)

def _fetch_water_coverage_fallback(lat: float, lon: float) -> float:
    """
    Fallback method using OpenMeteo API for water proximity data.
    """
    try:
        # Use OpenMeteo API for water-related data
        url = f"https://marine-api.open-meteo.com/v1/marine?latitude={lat}&longitude={lon}&daily=wave_height"
        
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if 'daily' in data and 'wave_height' in data['daily']:
            # If wave height data is available, location is near water
            return min(max(np.mean(data['daily']['wave_height']) / 2, 0), 1)
            
        # If no marine data, check for inland water bodies using Nominatim
        nominatim_url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json&zoom=14"
        headers = {'User-Agent': 'Water Coverage Calculator 1.0'}
        
        response = requests.get(nominatim_url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        # Check location type
        if any(water_type in str(data).lower() for water_type in ['lake', 'river', 'sea', 'ocean', 'bay', 'wetland']):
            return 0.7  # Significant water presence
        return 0.1  # Minimal water presence
            
    except Exception as e:
        # logger.error(f"Error in fallback water coverage calculation: {str(e)}")
        return 0.1  # Default low water coverage

def fetch_green_cover_data(lat: float, lon: float) -> float:
    """
    Fetch green cover data using Copernicus Global Land Service API.
    Uses NDVI (Normalized Difference Vegetation Index) data.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
    
    Returns:
        float: NDVI value between 0 and 1
    """
    try:
        # Using Copernicus Global Land Service API
        base_url = "https://land.copernicus.vgt.vito.be/REST/TimeSeries/1.0/extract"
        
        # Current date and one month ago
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        params = {
            'lat': lat,
            'lon': lon,
            'startdate': start_date.strftime('%Y-%m-%d'),
            'enddate': end_date.strftime('%Y-%m-%d'),
            'collection': 'NDVI_V2',
            'format': 'json'
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json'
        }
        
        response = requests.get(base_url, params=params, headers=headers)
        
        if response.status_code != 200:
            # Fallback to alternative API: OpenMeteo
            return _fetch_green_cover_fallback(lat, lon)
            
        data = response.json()
        ndvi_values = [item['NDVI'] for item in data['results'] if 'NDVI' in item]
        
        if ndvi_values:
            # NDVI values are typically between -1 and 1
            # Normalize to 0-1 range
            avg_ndvi = np.mean(ndvi_values)
            normalized_ndvi = (avg_ndvi + 1) / 2
            return max(min(normalized_ndvi, 1), 0)
            
        return _fetch_green_cover_fallback(lat, lon)
        
    except Exception as e:
        logger.error(f"Error in primary green cover fetch: {str(e)}")
        return _fetch_green_cover_fallback(lat, lon)

def _fetch_green_cover_fallback(lat: float, lon: float) -> float:
    """
    Fallback method using OpenMeteo API for vegetation data.
    """
    try:
        # OpenMeteo API for soil and vegetation data
        url = (f"https://api.open-meteo.com/v1/forecast?"
               f"latitude={lat}&longitude={lon}"
               f"&hourly=soil_moisture_0_1cm,soil_moisture_1_3cm"
               f"&daily=et0_fao_evapotranspiration")
        
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Calculate green cover proxy using soil moisture and evapotranspiration
        soil_moisture = np.mean(data['hourly']['soil_moisture_0_1cm'][:24])  # First 24 hours
        evapotranspiration = data['daily']['et0_fao_evapotranspiration'][0]  # First day
        
        # Combine metrics to estimate vegetation cover
        # Normalize values based on typical ranges
        soil_moisture_norm = min(soil_moisture / 50, 1)  # Typical range 0-50
        evapotrans_norm = min(evapotranspiration / 10, 1)  # Typical range 0-10
        
        # Weight the factors
        green_cover = (soil_moisture_norm * 0.6 + evapotrans_norm * 0.4)
        
        return max(min(green_cover, 1), 0)
        
    except Exception as e:
        # logger.error(f"Error in fallback green cover fetch: {str(e)}")
        return _fetch_green_cover_last_resort(lat, lon)

def _fetch_green_cover_last_resort(lat: float, lon: float) -> float:
    """
    Last resort method using NASA POWER API for vegetation-related data.
    """
    try:
        # NASA POWER API
        base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        
        params = {
            'parameters': 'T2M,PRECTOT,RH2M',  # Temperature, precipitation, humidity
            'community': 'AG',
            'longitude': lon,
            'latitude': lat,
            'start': datetime.now().strftime('%Y%m%d'),
            'end': datetime.now().strftime('%Y%m%d'),
            'format': 'JSON'
        }
        
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Extract relevant parameters
        temp = float(data['properties']['parameter']['T2M'][datetime.now().strftime('%Y%m%d')])
        precip = float(data['properties']['parameter']['PRECTOT'][datetime.now().strftime('%Y%m%d')])
        humidity = float(data['properties']['parameter']['RH2M'][datetime.now().strftime('%Y%m%d')])
        
        # Create a simple vegetation index based on environmental conditions
        # This is a rough approximation based on typical conditions favorable for vegetation
        temp_factor = max(0, min(1 - abs(temp - 20) / 30, 1))  # Optimal temp around 20°C
        precip_factor = min(precip / 10, 1)  # Normalize precipitation (0-10mm)
        humidity_factor = humidity / 100  # Humidity is already 0-100
        
        # Combine factors with weights
        green_cover = (temp_factor * 0.3 + precip_factor * 0.4 + humidity_factor * 0.3)
        
        return max(min(green_cover, 1), 0)
        
    except Exception as e:
        # logger.error(f"Error in last resort green cover fetch: {str(e)}")
        # Return a reasonable default based on global averages
        return 0.3  # Global average vegetation cover is roughly 30%

def fetch_biodiversity_data(lat: float, lon: float) -> int:
    """
    Fetch biodiversity data using GBIF API with improved species counting.
    """
    try:
        params = {
            "decimalLatitude": f"{lat-0.5},{lat+0.5}",
            "decimalLongitude": f"{lon-0.5},{lon+0.5}",
            "limit": 300,  # Increased limit
            "hasCoordinate": True,
            "hasGeospatialIssue": False
        }
        response = requests.get(GBIF_API_BASE, params=params)
        response.raise_for_status()
        data = response.json()
        time.sleep(REQUEST_DELAY)
        
        # Count unique species
        species_set = set()
        for record in data.get("results", []):
            if record.get("species"):
                species_set.add(record["species"])
        
        return len(species_set)
    except Exception as e:
        # logger.error(f"Error fetching biodiversity data for {lat}, {lon}: {str(e)}")
        return 0   

def create_dataset(
    cache):
    """
    Create a labeled dataset across various regions with selected land use types.
    
    Args:
        regions (list): List of dictionaries containing lat/lon coordinates
        land_use_types (list): List of land use type strings
        
    Returns:
        pandas.DataFrame: Dataset containing environmental metrics and risk scores
    """
    data = []
    regions = [{"lat": cache['latitude'], "lon": cache['longitude']}]
    land_use_types = [cache['use_case_type']]
    total_regions = len(regions)
    
    for idx, region in enumerate(regions, 1):
        try:
            lat, lon = region["lat"], region["lon"]
            logger.info(f"Processing region {idx}/{total_regions}: {lat}, {lon}")
            
            climate_data = fetch_climate_data(lat, lon)
            biodiversity_data = fetch_biodiversity_data(lat, lon)
            green_cover = fetch_green_cover_data(lat, lon)
            land_usage = fetch_land_usage_data(lat, lon)
            water_coverage = fetch_water_coverage_data(lat, lon)
            
            if climate_data:
                for land_use_type in land_use_types:
                    
                    data.append({
                        "latitude": lat,
                        "longitude": lon,
                        "temperature": climate_data["temperature"],
                        "precipitation": climate_data["precipitation"],
                        "humidity": climate_data["humidity"],
                        "species_richness": biodiversity_data,
                        "ndvi": green_cover,
                        "urban_land_usage": land_usage,
                        "water_coverage": water_coverage,
                        "land_use_type": land_use_type,
                        # "timestamp": datetime.now().isoformat()
                    })
            
        except Exception as e:
            # logger.error(f"Error processing region {lat}, {lon}: {str(e)}")
            continue
    
    return pd.DataFrame(data)

def preprocess_inference_data(input_data, label_encoders, scaler):
    """
    Preprocess input data for inference.
    
    Args:
        input_data (pd.DataFrame): New input data for prediction.
        label_encoders (dict): Dictionary of fitted LabelEncoders for categorical features.
        scaler (StandardScaler): Fitted StandardScaler for numerical features.
        categorical_columns (list): List of categorical feature names.
        numerical_columns (list): List of numerical feature names.
    
    Returns:
        pd.DataFrame: Preprocessed input data.
    """
    data = input_data.copy()

    categorical_columns = ['land_use_type']
    numerical_columns = ['latitude', 'longitude', 'temperature', 'precipitation', 
                         'ndvi', 'urban_land_usage', 'water_coverage','humidity', 'species_richness']
    
    # Encode categorical features
    for col in categorical_columns:
        if col in data.columns and col in label_encoders:
            data[col] = label_encoders[col].transform(data[col])
    
    # Scale numerical features
    if set(numerical_columns).issubset(data.columns):
        data[numerical_columns] = scaler.transform(data[numerical_columns])
    
    return data
@app.route('/analyze', methods=['POST'])
def predict(input_data, models, weights, label_encoders, scaler):
    """
    Perform inference on input data.
    
    Args:
        input_data (pd.DataFrame): New input data for prediction.
        models (dict): Dictionary of trained models.
        weights (dict): Dictionary of model weights for ensemble.
        label_encoders (dict): Fitted label encoders for categorical features.
        scaler (StandardScaler): Fitted scaler for numerical features.
        categorical_columns (list): List of categorical feature names.
        numerical_columns (list): List of numerical feature names.
    
    Returns:
        float: Final ensemble prediction for `risk_score`.
    """
    # Preprocess the data
    processed_data = preprocess_inference_data(input_data, label_encoders, scaler)
    
    # Generate predictions for each model
    predictions = {
        name: model.predict(processed_data) for name, model in models.items()
    }
    
    # Compute weighted ensemble prediction
    ensemble_prediction = sum(predictions[model] * weight for model, weight in weights.items())
    return ensemble_prediction

def calculate_shap_values(input_data: pd.DataFrame, models: dict, weights: dict, label_encoders: dict, scaler: dict) -> dict:
    """
    Calculate SHAP values for the ensemble model predictions.
    
    Args:
        input_data (pd.DataFrame): Input data for prediction
        models (dict): Dictionary of trained models
        weights (dict): Dictionary of model weights for ensemble
        label_encoders (dict): Fitted label encoders for categorical features
        scaler (dict): Fitted scaler for numerical features
    
    Returns:
        dict: Dictionary containing SHAP values for each feature
    """
    # Preprocess the data
    processed_data = preprocess_inference_data(input_data, label_encoders, scaler)
    
    # Initialize SHAP values array
    final_shap_values = np.zeros(processed_data.shape[1])
    
    # Calculate SHAP values for each model and apply ensemble weights
    for model_name, model in models.items():
        # Create explainer based on model type
        if isinstance(model, LGBMRegressor):
            explainer = shap.TreeExplainer(model)
        elif isinstance(model, XGBRegressor):
            explainer = shap.TreeExplainer(model)
        elif isinstance(model, CatBoostRegressor):
            explainer = shap.TreeExplainer(model)
        else:
            continue
            
        # Calculate SHAP values for current model
        shap_values = explainer.shap_values(processed_data)
        
        # If shap_values is a list (happens with some models), take the first element
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
            
        # Apply model weight to SHAP values
        weighted_shap = shap_values * weights[model_name]
        
        # Add to final SHAP values
        final_shap_values += weighted_shap[0]  # Take first row as we're only predicting for one instance
    
    # Create dictionary with feature names and their SHAP values
    feature_names = ['latitude', 'longitude', 'temperature', 'precipitation', 
                    'humidity', 'species_richness', 'ndvi', 'urban_land_usage', 
                    'water_coverage', 'land_use_type']
    
    shap_dict = {
        feature: float(shap_value)  # Convert numpy float to Python float for JSON serialization
        for feature, shap_value in zip(feature_names, final_shap_values)
    }
    
    # Sort dictionary by absolute SHAP values to show features in order of importance
    shap_dict = dict(sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True))
    
    return {
        "shap_values": shap_dict,
        "explanation": {
            "positive_impact": [k for k, v in shap_dict.items() if v > 0],
            "negative_impact": [k for k, v in shap_dict.items() if v < 0],
            "most_influential_features": list(shap_dict.keys())[:3]
        }
    }
cache={
        "latitude": 11.96,
        "longitude": 75.92,
        "use_case_type": "Agricultural"
      }
#change this shit
print('hiiiiiiiiiiiiiiiiiiiiiiii')

@app.route('/main', methods=['POST'])
def main():
    try:
        print("hellllooooooo")
        print("HIIIII HELLLLOO")
        cache = request.get_json()  # Get the JSON data from request
        print(cache)
        
        # Create dataset from input cache
        input_data = create_dataset(cache)

        # Perform risk score prediction
        risk_score_prediction = predict(
            input_data=input_data,
            models=loaded_models,
            weights=loaded_weights,
            label_encoders=loaded_encoders,
            scaler=loaded_scaler
        )

        # Perform SHAP analysis
        shap_analysis = calculate_shap_values(
            input_data=input_data,
            models=loaded_models,
            weights=loaded_weights,
            label_encoders=loaded_encoders,
            scaler=loaded_scaler
        )

        # Combine the results
        result = {
            "status": "success",
            "risk_score_prediction": float(risk_score_prediction[0]),  # Convert numpy float to Python float
            "shap_analysis": shap_analysis
        }

        print(result)
        return jsonify({"risk_score_prediction":str(risk_score_prediction[0]),'shap_analysis':shap_analysis})

    except Exception as e:
        print("Error:", traceback.format_exc())
        return app.response_class(
            response=json.dumps({
                "status": "error", 
                "message": str(e)
            }),
            status=400,
            mimetype='application/json'
        )

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5020, debug=True)