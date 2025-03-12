# Kode hentet fra https://frost.met.no/python_example.html

import requests
import pandas as pd

# Insert your own client ID here
client_id = '1b7e5cfb-47e3-4e89-b0b5-473bedac8160'

# Define endpoint and parameters
endpoint = 'https://frost.met.no/observations/v0.jsonld'
parameters = {
    'sources': 'SN68860,SN39210',
    'elements': 'mean(air_temperature P1D),sum(precipitation_amount P1D), surface_snow_thickness',
    'referencetime': '2012-12-31/2024-12-31',
}
# Issue an HTTP GET request
r = requests.get(endpoint, parameters, auth=(client_id,''))
# Extract JSON data
json = r.json()

# Check if the request worked, print out any errors
if r.status_code == 200:
    data = json['data']
    print('Data retrieved from frost.met.no!')
else:
    print('Error! Returned status code %s' % r.status_code)
    print('Message: %s' % json['error']['message'])
    print('Reason: %s' % json['error']['reason'])

# This will return a Dataframe with all of the observations in a table format
df = pd.DataFrame()
for i in range(len(data)):
    row = pd.DataFrame(data[i]['observations'])
    row['referenceTime'] = data[i]['referenceTime']
    row['sourceId'] = data[i]['sourceId']
    df = df._append(row)

df = df.reset_index()

df.to_csv("raw_data/data/csv_df.csv")