
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



FILE_PATH = "../raw_data/data/Vaerdata.csv"


def analyze_weather_data():
    
    # Leser data fra Excel-filen
    data = pd.read_csv(FILE_PATH)
    
    # Konverterer tiden
    data["referenceTime"] = pd.to_datetime(data["referenceTime"])
    data["timeOffset"] = pd.to_timedelta(data["timeOffset"])
    data["justertTid"] = data["referenceTime"] - data["timeOffset"]

    # Filtrer temperatur og nedbør
    temp_data = data[data["elementId"] == "mean(air_temperature P1D)"]
    nedbør_data = data[data["elementId"] == "sum(precipitation_amount P1D)"]

    # Henter ut verdiene og sorterer
    temperatur = temp_data["value"].values
    temp_tider = temp_data["justertTid"].values
    sortert_index = np.argsort(temp_tider)
    temp_tider_sortert = temp_tider[sortert_index]
    temperatur_sortert = temperatur[sortert_index]

    nedbør = nedbør_data["value"].values
    nedbør_tider = nedbør_data["justertTid"].values
    sortert_index_nedbør = np.argsort(nedbør_tider)
    nedbør_tider_sortert = nedbør_tider[sortert_index_nedbør]
    nedbør_sortert = nedbør[sortert_index_nedbør]

    # Korrelasjon mellom temperatur og nedbør
    temp_daglig = temp_data.groupby(pd.to_datetime(temp_data['justertTid']).dt.date)['value'].mean()
    nedbør_daglig = nedbør_data.groupby(pd.to_datetime(nedbør_data['justertTid']).dt.date)['value'].mean()
    korrelasjon = temp_daglig.corr(nedbør_daglig)

    # Statistikk for nedbør
    gjennomsnitts_nedbør = np.mean(nedbør)
    median_nedbør = np.median(nedbør)
    standardavvik_nedbør = np.std(nedbør)

    # Statistikk for temperatur
    gjennomsnitts_temp = np.mean(temperatur)
    median_temp = np.median(temperatur)
    standardavvik_temp = np.std(temperatur)

    # Resultater
    resultater = {
        "korrelasjon": korrelasjon,
        "nedbør_statistikk": {
            "gjennomsnitt": gjennomsnitts_nedbør,
            "median": median_nedbør,
            "standardavvik": standardavvik_nedbør
        },
        "temperatur_statistikk": {
            "gjennomsnitt": gjennomsnitts_temp,
            "median": median_temp,
            "standardavvik": standardavvik_temp
        },
        "sorterte_data": {
            "temperatur": {
                "tider": temp_tider_sortert,
                "verdier": temperatur_sortert
            },
            "nedbør": {
                "tider": nedbør_tider_sortert,
                "verdier": nedbør_sortert
            }
        },
        "temp_data": temp_data,
        "nedbør_data": nedbør_data
    }


    return resultater


def temperatur(resultater):
        #Data fra analyse
        temp_tider_sortert = resultater["sorterte_data"]["temperatur"]["tider"]
        temperatur_sortert = resultater["sorterte_data"]["temperatur"]["verdier"]
        gjennomsnitts_temp = resultater["temperatur_statistikk"]["gjennomsnitt"]
        median_temp = resultater["temperatur_statistikk"]["median"]
        standardavvik_temp = resultater["temperatur_statistikk"]["standardavvik"]
        temp_data = resultater["temp_data"]

        #Plot temperatur over tid
        plt.figure(figsize=(15,5))
        plt.plot(temp_tider_sortert,temperatur_sortert) 
        plt.title("Temperatur over tid")
        plt.xlabel("antall målinger")
        plt.ylabel("Temperatur (C)")
        plt.grid(True)
        plt.show()

        #Plot gjennomsnittstemperatur per år
        temp_årlig = temp_data.groupby(pd.to_datetime(temp_data['justertTid']).dt.year)['value'].mean()
        temp_årlig_uten_2012=temp_årlig[1:]
        plt.figure(figsize=(15,5))
        plt.bar(temp_årlig_uten_2012.index,temp_årlig_uten_2012.values)
        plt.xticks(temp_årlig_uten_2012.index)
        plt.title("Gjennomsnittstemperatur gjennom årene")
        plt.xlabel("År")
        plt.ylabel("Gjennomsnittstemperatur i °C")
        plt.grid(True)
        plt.show()
        #Print statistikk
        print("Gjennomsnittstemperaturen er:",round(gjennomsnitts_temp,2),"°C")
        print("Median temperaturen er:",round(median_temp,2),"°C")
        print("Standardavviket til temperaturen er:",round(standardavvik_temp),"°C")



def nedbør(resultater):
        #Data fra analyse
        nedbør_tider_sortert = resultater["sorterte_data"]["nedbør"]["tider"]
        nedbør_sortert = resultater["sorterte_data"]["nedbør"]["verdier"]
        gjennomsnitts_nedbør = resultater["nedbør_statistikk"]["gjennomsnitt"]
        median_nedbør = resultater["nedbør_statistikk"]["median"]
        standardavvik_nedbør = resultater["nedbør_statistikk"]["standardavvik"]
        nedbør_data = resultater["nedbør_data"]

        #Plot nedbør over tid
        plt.figure(figsize=(15,5))
        plt.plot(nedbør_tider_sortert,nedbør_sortert)  
        plt.title("Nedbørsdata over tid")
        plt.xlabel("antall målinger")
        plt.ylabel("Nedbør (mm)")
        plt.grid(True)
        plt.show()

        #Plot gjennomsnittsnedbør per år
        nedbør_årlig = nedbør_data.groupby(pd.to_datetime(nedbør_data['justertTid']).dt.year)['value'].mean()
        nedbør_årlig_uten_2012=nedbør_årlig[1:]
        plt.figure(figsize=(15,5))
        plt.bar(nedbør_årlig_uten_2012.index,nedbør_årlig_uten_2012.values)
        plt.xticks(nedbør_årlig_uten_2012.index)
        plt.title("Gjennomsnittsnedbør ulike årene")
        plt.xlabel("År")
        plt.ylabel("Gjennomsnittsnedbør i mm")
        plt.grid(True)
        plt.show()

        #Print statistikk
        print("Gjennomsnittsnedbøren er:",round(gjennomsnitts_nedbør,2),"mm")
        print("Median nedbøren er:",round(median_nedbør,2),"mm")
        print("Standardavviket til nedbøren er:",round(standardavvik_nedbør),"mm")





