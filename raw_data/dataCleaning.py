import pandas as pd

#Vi behandler dataframen sånn at vi kun får de dataene vi synes er relevante

df_weather = pd.read_csv("raw_data/data/csv_weather.csv")
df_emissions = pd.read_csv("raw_data/data/csv_emissions.csv")

df_weather.to_excel("Excelfil_ubeh.xlsx")
#df_emissions.to_excel("Utslipp.xlsx")

df_trimmed_weather = df_weather[["index", "elementId", "value", "timeOffset", "qualityCode","referenceTime","sourceId"]]
df_trimmed_emissions = df_emissions[["kilde (aktivitet)", "energiprodukt", "komponent", "år", "statistikkvariabel", "13931: Klimagasser AR5, etter kilde (aktivitet), energiprodukt, komponent, år og statistikkvariabel"]]

year_dict = df_trimmed_emissions["år"].to_dict()
non_valid_years = []

for key in year_dict:
    print(key)

cutoff_quality = 3


quality_dict = df_trimmed_weather["qualityCode"].to_dict()
non_valid_qualities = []

for key in quality_dict:
    try:
        int(quality_dict[key])

    except ValueError:  #Sorterer ut Nan-verdier
        non_valid_qualities.append(key)
        continue

    if (int(quality_dict[key]) > cutoff_quality):  #Sorterer ut verdier dersom de overstiger kvalitets-cutoffen.
        non_valid_qualities.append(key)

df_trimmed_weather.drop(df_trimmed_weather[df_trimmed_weather.år > 2012].index, inplace = True)

df_trimmed_weather = df_trimmed_weather.drop(non_valid_qualities)
        
df_trimmed_weather.to_excel("Excelfil.xlsx")


