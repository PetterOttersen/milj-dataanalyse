import pandas as pd

#Vi behandler dataframen sånn at vi kun får de dataene vi synes er relevante

df_weather = pd.read_csv("raw_data/data/csv_weather.csv")
df_emissions = pd.read_csv("raw_data/data/csv_emissions.csv")

df_weather.to_excel("Excelfil_ubeh.xlsx")
#df_emissions.to_excel("Utslipp.xlsx")

df_trimmed_weather = df_weather[["elementId", "value", "timeOffset", "qualityCode","referenceTime","sourceId"]]
df_trimmed_emissions = df_emissions[["kilde (aktivitet)", "energiprodukt", "komponent", "år", "statistikkvariabel", "13931: Klimagasser AR5, etter kilde (aktivitet), energiprodukt, komponent, år og statistikkvariabel"]]

year_dict = df_trimmed_emissions["år"].to_dict()
komponent_dict = df_trimmed_emissions["komponent"].to_dict()
energiprodukt_dict = df_trimmed_emissions["energiprodukt"]

non_valid_years = []

for key in year_dict:
    if (year_dict[key] <= 2012) or (year_dict[key] > 2024 or (komponent_dict[key] != "K11 Karbondioksid (CO2)") or (energiprodukt_dict[key] != "VT0 I alt")):
        df_trimmed_emissions = df_trimmed_emissions.drop(key)


cutoff_quality = 3


quality_dict = df_trimmed_weather["qualityCode"].to_dict()
non_valid_qualities = []
"""
non_valid_qualities = [
    key for key, value in quality_dict.items()
    if pd.isna(value) or not str(value).isdigit() or int(value) > cutoff_quality
]
"""
for key in quality_dict:
    try:
        int(quality_dict[key])

    except ValueError:  #Sorterer ut Nan-verdier
        non_valid_qualities.append(key)
        continue

    if (int(quality_dict[key]) > cutoff_quality):  #Sorterer ut verdier dersom de overstiger kvalitets-cutoffen.
        non_valid_qualities.append(key)

df_trimmed_weather = df_trimmed_weather.drop(non_valid_qualities)

df_trimmed_emissions.to_excel("Excelfil_utslipp.xlsx")
df_trimmed_weather.to_excel("Excelfil.xlsx")


