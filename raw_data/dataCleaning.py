import pandas as pd

#Vi behandler dataframen sånn at vi kun får de dataene vi synes er relevante

df_weather = pd.read_csv("raw_data/data/csv_weather.csv")
df_emissions = pd.read_csv("raw_data/data/csv_emissions.csv")

df_weather.to_csv("raw_data/data/Vaerdata_ubehandlet.csv")

df_trimmed_weather = df_weather[["elementId", "value", "timeOffset", "qualityCode","referenceTime","sourceId"]]
df_trimmed_emissions = df_emissions[["kilde (aktivitet)", "energiprodukt", "komponent", "år", "statistikkvariabel", "13931: Klimagasser AR5, etter kilde (aktivitet), energiprodukt, komponent, år og statistikkvariabel"]]


quality_dict = df_trimmed_weather["qualityCode"].to_dict()
non_valid_qualities = []

cutoff_quality = 3 #Dataen fra FROST API kommer med en kvalitetsparameter. Her velger vi å beholde dataen hvis kvaliteten er under 3

df_trimmed_weather = df_trimmed_weather.dropna(subset=["qualityCode"]) #Fjerner NA-verdier fra dfen

cutoff_quality = 3

df_trimmed_weather = df_trimmed_weather.reset_index(drop=True)
quality_dict = df_trimmed_weather["qualityCode"].to_dict()

non_valid_indices = [
    index for index, value in quality_dict.items()
    if pd.isna(value) or not str(value).isdigit() or int(value) > cutoff_quality
]
df_trimmed_weather = df_trimmed_weather.drop(non_valid_indices)

"""
for key in quality_dict:
    try:
        int(quality_dict[key])

    except ValueError:  #Sorterer ut Nan-verdier
        non_valid_qualities.append(key)
        continue

    if (int(quality_dict[key]) > cutoff_quality):  #Sorterer ut verdier dersom de overstiger kvalitets-cutoffen.
        non_valid_qualities.append(key)
"""

#df_trimmed_weather = df_trimmed_weather.drop(non_valid_qualities)

filter_condition = (
    (df_trimmed_emissions["år"] >= 2013) & 
    (df_trimmed_emissions["år"] <= 2024) & 
    (df_trimmed_emissions["komponent"] == "K11 Karbondioksid (CO2)") & 
    (df_trimmed_emissions["energiprodukt"] == "VT0 I alt")
)
df_trimmed_emissions = df_trimmed_emissions[filter_condition] #Bruker list comprehension til å sortere ut ikke-relevante data

df_trimmed_emissions.to_csv("raw_data/data/Utslippdata.csv")
df_trimmed_weather.to_csv("raw_data/data/Vaerdata.csv")


