import sys
sys.path.append("../raw_data")

import unittest
import requests
import pandas as pd

from dataCleaning import filter_weather

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
        assert r.status_code == 200, f"API request failed with status code {r.status_code}"

    def test_nonexistent_api(self):
        # Tester hva som skjer hvis man gir en ikke gyldig parameter til apien. Her tester vi en tid som ikke ligger i datasettet
        empty_params = {'sources': 'SN68860', 'referencetime': '1900-01-01/1900-01-02'}
        r = requests.get(endpoint, empty_params, auth=(self.client_id, ''))
        assert len(r.json().get('data', [])) == 0, "Empty response not handled"

class TestData(unittest.TestCase):
    def testDataHandling(self):
        df = pd.read_csv("raw_data/data/Vaerdata.csv")
        appended_rows = pd.DataFrame([[0,0], [0,0], [0,0], ["test", 6], [0,0], [0,0]])

        len_original_df = len(df)

        df.append(appended_rows)
        len_corrupted_df = len(df)

        

def test_suite():
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAPI))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestData))
    return suite

runner = unittest.TextTestRunner()
runner.run(test_suite())
