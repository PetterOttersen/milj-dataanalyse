import unittest
import requests

endpoint = 'https://frost.met.no/observations/v0.jsonld'
parameters = {
    'sources': 'SN68860,SN39210',
    'elements': 'index, mean(air_temperature P1D),sum(precipitation_amount P1D), surface_snow_thickness',
    'referencetime': '2012-12-31/2024-12-31',
}

def test_api_connection():
    # Test if the API returns a successful status code (200)
    r = requests.get(endpoint, parameters, auth=(client_id, ''))
    assert r.status_code == 200, f"API request failed with status code {r.status_code}"

