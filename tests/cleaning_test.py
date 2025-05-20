import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from raw_data.dataCleaning import filter_weather, filter_emissions
import pandas as pd

class TestData(unittest.TestCase):
    def testWeatherFilter(self):
        df = pd.read_csv("raw_data/data/Vaerdata.csv")
        appended_rows = pd.DataFrame([[0,0], [0,0], [0,0], ["test", 6], [0,0], [0,0]]) #Lager rader med bevisst feil data

        len_original_df = len(df)

        df_corrupted = pd.concat([df, appended_rows], ignore_index=True)
        len_corrupted_df = len(df_corrupted) #"Ødelegger" df for p sjekke om den fikses

        fixed_df = filter_weather(df_corrupted)
        len_fixed_df = len(fixed_df)

        self.assertLess(len(fixed_df), len(df_corrupted)) #Sjekker at det faktisk fjernes rader
        self.assertEqual(len_original_df, len_fixed_df) #Sjekker at den ødelagte df-en faktisk fikses

    def testEmissionFilter(self):
        df = pd.read_csv("raw_data/data/Utslippdata.csv")
        appended_rows = pd.DataFrame([[0,0], [0,0], [0,0], ["test", 2011], [0,0], [0,0]])

        len_original_df = len(df)

        df_corrupted = pd.concat([df, appended_rows], ignore_index=True)
        len_corrupted_df = len(df_corrupted)

        fixed_df = filter_emissions(df_corrupted)
        len_fixed_df = len(fixed_df)

        self.assertLess(len(fixed_df), len(df_corrupted))
        self.assertEqual(len_original_df, len_fixed_df)


def test_suite():
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestData))
    return suite

runner = unittest.TextTestRunner()
runner.run(test_suite())