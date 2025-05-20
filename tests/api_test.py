import unittest
import requests

#from dataCleaning import filter_weather

endpoint = 'https://frost.met.no/observations/v0.jsonld'
parameters = {
    'sources': 'SN68860,SN39210',
    'elements': 'index, mean(air_temperature P1D),sum(precipitation_amount P1D), surface_snow_thickness',
    'referencetime': '2012-12-31/2024-12-31',
}

class TestAPI(unittest.TestCase):
    client_id = '1b7e5cfb-47e3-4e89-b0b5-473bedac8160'

    def test_api_connection(self):
        # Sjekker om API-en henter statuskode 200
        r = requests.get(endpoint, parameters, auth=(self.client_id, ''))
        self.assertEqual(r.status_code, 200, f"API-forespørel feilet med statuskde {r.status_code}")

    def test_nonexistent_api(self):
        # Tester hva som skjer hvis man gir en ikke gyldig parameter til apien. Her tester vi en tid som ikke ligger i datasettet
        empty_params = {'sources': 'SN68860', 'referencetime': '1900-01-01/1900-01-02'}
        r = requests.get(endpoint, empty_params, auth=(self.client_id, ''))
        self.assertEqual(len(r.json().get('data', [])), 0, "Tom respons ikke lik 0")


def test_suite():
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAPI))
    return suite

runner = unittest.TextTestRunner()
runner.run(test_suite())
