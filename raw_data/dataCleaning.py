import pandas as pd

#Vi behandler dataframen sånn at vi kun får de dataene vi synes er relevante

df_weather = pd.read_csv("raw_data/data/csv_weather.csv")
df_emissions = pd.read_csv("raw_data/data/csv_emissions.csv")

df_weather.to_csv("raw_data/data/Vaerdata_ubehandlet.csv", index = False) #Inkluderer index = False for å unngå få med indexen 

df_trimmed_weather = df_weather[["elementId", "value", "timeOffset", "qualityCode","referenceTime","sourceId"]]
df_trimmed_emissions = df_emissions[["kilde (aktivitet)", "energiprodukt", "komponent", "år", "statistikkvariabel", "13931: Klimagasser AR5, etter kilde (aktivitet), energiprodukt, komponent, år og statistikkvariabel"]]


def filter_weather(df_trimmed_weather):
    """
    Filtrerer dataen fra yr.no

    Parametere: Dataframe med data fra yr.no

    Returnerer Dataframe med data fra yr.no

    """

    cutoff_quality = 3 #Dataen fra FROST API kommer med en kvalitetsparameter. Her velger vi å beholde dataen hvis kvaliteten er under 3
    filter = []

    for idx, row in df_trimmed_weather.iterrows(): #Sorterer ut NA-verdier, ikke-eksisterende og for høye verdier fra kvalitetskolonna
        i = row["qualityCode"]
        filter.append(pd.isna(i) or str(i).isdigit() or int(i) > cutoff_quality)

    df_trimmed_weather = df_trimmed_weather[~pd.Series(filter, index=df_trimmed_weather.index)]
    #~ sin funksjon er at istedenfor å beholde den gitte kolonna, forkaster den kolonna. ~ er en bitwise-operator.

    return df_trimmed_weather


def filter_emissions(df_trimmed_emissions):
    """
    Filtrerer dataen fra yr.no

    Parametere: Dataframe med data fra yr.no

    Returnerer Dataframe med data fra yr.no

    """
    filter_condition = (
        (df_trimmed_emissions["år"].between(2013,2024)) &
        (df_trimmed_emissions["komponent"] == "K11 Karbondioksid (CO2)") & 
        (df_trimmed_emissions["energiprodukt"] == "VT0 I alt")
    )

    df_trimmed_emissions = df_trimmed_emissions[filter_condition] #Bruker list comprehension til å sortere ut ikke-relevante data

    return df_trimmed_emissions


df_trimmed_weather = filter_weather(df_trimmed_weather)
df_trimmed_emissions = filter_emissions(df_trimmed_emissions)

df_trimmed_emissions.to_csv("raw_data/data/Utslippdata.csv", index = False)
df_trimmed_weather.to_csv("raw_data/data/Vaerdata.csv", index = False)


