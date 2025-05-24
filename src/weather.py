
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from ipywidgets import interact, widgets
from IPython.display import display




FILE_PATH = "../raw_data/data/Vaerdata.csv"

class analyse_og_visualisere:

    def __init__(self, file_path="../raw_data/data/Vaerdata.csv"):
        self.FILE_PATH = file_path

    def analyser_vaerdata(self):
    
    # Leser data fra Excel-filen
        data = pd.read_csv(self.FILE_PATH)
    
    # Konverterer tiden til datetime og justerer med hensyn på timeOffset
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

    # Resultater i et dictionary
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


    def temperatur(self,resultater):
        #Data fra analyze_weather_data
        temp_tider_sortert = resultater["sorterte_data"]["temperatur"]["tider"]
        temperatur_sortert = resultater["sorterte_data"]["temperatur"]["verdier"]
        gjennomsnitts_temp = resultater["temperatur_statistikk"]["gjennomsnitt"]
        median_temp = resultater["temperatur_statistikk"]["median"]
        standardavvik_temp = resultater["temperatur_statistikk"]["standardavvik"]
        temp_data = resultater["temp_data"]

        #Plot temperatur over tid
        plt.figure(figsize=(15,5))
        plt.plot(temp_tider_sortert,temperatur_sortert) 
        plt.title("Figur 1: Temperatur over tid")
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
        plt.title("Figur 2: Gjennomsnittstemperatur gjennom årene")
        plt.xlabel("År")
        plt.ylabel("Gjennomsnittstemperatur i °C")
        plt.grid(True)
        plt.show()
        #Print statistikk
        print("Gjennomsnittstemperaturen er:",round(gjennomsnitts_temp,2),"°C")
        print("Median temperaturen er:",round(median_temp,2),"°C")
        print("Standardavviket til temperaturen er:",round(standardavvik_temp),"°C")



    def nedbør(self,resultater):
        #Data fra analyze_weather_data
        nedbør_tider_sortert = resultater["sorterte_data"]["nedbør"]["tider"]
        nedbør_sortert = resultater["sorterte_data"]["nedbør"]["verdier"]
        gjennomsnitts_nedbør = resultater["nedbør_statistikk"]["gjennomsnitt"]
        median_nedbør = resultater["nedbør_statistikk"]["median"]
        standardavvik_nedbør = resultater["nedbør_statistikk"]["standardavvik"]
        nedbør_data = resultater["nedbør_data"]

        #Plot nedbør over tid
        plt.figure(figsize=(15,5))
        plt.plot(nedbør_tider_sortert,nedbør_sortert)  
        plt.title("Figur 3: Nedbørsdata over tid")
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
        plt.title("Figur 4: Gjennomsnittsnedbør ulike årene")
        plt.xlabel("År")
        plt.ylabel("Gjennomsnittsnedbør i mm")
        plt.grid(True)
        plt.show()

        #Print statistikk
        print("Gjennomsnittsnedbøren er:",round(gjennomsnitts_nedbør,2),"mm")
        print("Median nedbøren er:",round(median_nedbør,2),"mm")
        print("Standardavviket til nedbøren er:",round(standardavvik_nedbør),"mm")
        

    def sammenlign_temp_nedbør(self,resultater):
    # Data fra analyze_weather_data
        temp_data = resultater["temp_data"]
        nedbør_data = resultater["nedbør_data"]

    # Beregn årlig gjennomsnitt for temperatur og nedbør
        temp_årlig = temp_data.groupby(pd.to_datetime(temp_data['justertTid']).dt.year)['value'].mean()
        nedbør_årlig = nedbør_data.groupby(pd.to_datetime(nedbør_data['justertTid']).dt.year)['value'].mean()

    # Fjerner 2012 pga. kun data fra 1 dag
        temp_årlig = temp_årlig[1:]
        nedbør_årlig = nedbør_årlig[1:]

    # Beregn korrelasjon
        korrelasjon = temp_årlig.corr(nedbør_årlig)
        print(f"Korrelasjon mellom temperatur og nedbør: {korrelasjon:.2f}")

    # Visualiser sammenhengen
        plt.figure(figsize=(10, 6))
        plt.scatter(temp_årlig, nedbør_årlig, color='blue', alpha=0.7)
        plt.title("Figur 5: Sammenheng mellom temperatur og nedbør")
        plt.xlabel("Gjennomsnittstemperatur (°C)")
        plt.ylabel("Gjennomsnittsnedbør (mm)")
        plt.grid(True)


    

    def prediksjonsanalyse_nedbør_lineær(self, resultater):
        nedbør_data = resultater["nedbør_data"]
        
        #Håndterer manglende verdier
        if nedbør_data['value'].isna().any():
            print(f"Advarsel: {nedbør_data['value'].isna().sum()} manglende verdier blir fylt med årsmiddel")
            nedbør_data['value'] = nedbør_data.groupby('År')['value'].transform('mean').fillna(nedbør_data['value'])
        # Grupper etter år og beregn årlig gjennomsnitt
        nedbør_årlig = nedbør_data.groupby(pd.to_datetime(nedbør_data['justertTid']).dt.year)['value'].mean()
        # Fjerner 2012 pga. kun data fra 1 dag

        nedbør_årlig_uten_2012 = nedbør_årlig[1:]
        
        # Forbered data for modellering
        X = nedbør_årlig_uten_2012.index.values.reshape(-1, 1)
        y = nedbør_årlig_uten_2012.values  
        
        # Tren lineær regresjonsmodell
        model = LinearRegression()
        model.fit(X, y)
        
        y_pred = model.predict(X)

        siste_år = int(X[-1][0])
        
        def oppdater_plot(slutt_år):

            antall_fremtidige_år=slutt_år-siste_år if slutt_år>siste_år else 0

            plt.figure(figsize=(12, 6))
        
            # Plot faktisk nedbør
            plt.scatter(X, y, color='blue', label='Faktisk nedbør')
            
             
            fremtidige_år = np.array([siste_år + i for i in range(1, antall_fremtidige_år + 1)]).reshape(-1, 1)
            fremtidige_pred = model.predict(fremtidige_år)
            plt.scatter(fremtidige_år, fremtidige_pred, color='blue', marker='x', s=100, label='Fremtidige prediksjoner')

            # Plot regresjonslinje
            plt.plot(np.concatenate([X.flatten(),fremtidige_år.flatten()]), np.concatenate([y_pred, fremtidige_pred]), color='red', linewidth=2, label='Lineær regresjon')
            
            for år, pred in zip(fremtidige_år.flatten(), fremtidige_pred):
                plt.text(år, pred, f'{pred:.1f}', ha='right', va='bottom')
            
            plt.title('Figur 6: Årlig gjennomsnittlig nedbør med lineær regresjon')
            plt.xlabel('År')
            plt.ylabel('Gjennomsnittlig nedbør (mm)')
            plt.legend()
            plt.grid(True)
            #flatten gjør fra 2D til 1D, slik at det kan plottes i en graf
            plt.xticks(np.append(X.flatten(), fremtidige_år.flatten()),rotation=90)
            plt.show()
            
            for år, pred in zip(fremtidige_år.flatten(), fremtidige_pred):
                print(f"  År {år}: {pred:.1f} mm")

        # Interaktiv widget for antall fremtidige år
        interact(
        oppdater_plot, 
        slutt_år=widgets.IntSlider(
            value=siste_år+5,
            min=siste_år+1,
            max=siste_år+20,
            step=1,
            description='Velg år:',
            continuous_update=False
        )
    )



    def prediksjonsanalyse_temperatur_lineær(self, resultater):
        temp_data = resultater["temp_data"]
        

        if temp_data['value'].isna().any():
            print(f"Advarsel: {temp_data['value'].isna().sum()} manglende verdier blir fylt med årsmiddel")
            temp_data['value'] = temp_data.groupby('År')['value'].transform('mean').fillna(temp_data['value'])

        # Grupper etter år og beregn årlig gjennomsnitt
        temperatur_årlig = temp_data.groupby(pd.to_datetime(temp_data['justertTid']).dt.year)['value'].mean()
        # Fjerner 2012 pga. kun data fra 1 dag
        temperatur_årlig_uten_2012 = temperatur_årlig[1:]
        
        # Forbered data for modellering
        X = temperatur_årlig_uten_2012.index.values.reshape(-1, 1)
        y = temperatur_årlig_uten_2012.values  
        
        # Tren lineær regresjonsmodell
        model = LinearRegression()
        model.fit(X, y)
        
        y_pred = model.predict(X)

        siste_år = int(X[-1][0])

        def oppdater_plot(slutt_år):

            antall_fremtidige_år=slutt_år-siste_år if slutt_år>siste_år else 0
            plt.figure(figsize=(12, 6))
            
            # Plot faktisk temperatur
            plt.bar(X.flatten(), y, color='gray', label='Faktisk temperatur')
            
            
            fremtidige_år = np.array([siste_år + i for i in range(1, antall_fremtidige_år + 1)]).reshape(-1, 1)
            fremtidige_pred = model.predict(fremtidige_år)

            plt.bar(fremtidige_år.flatten(), fremtidige_pred, color='blue', label='Prediktert temperatur')
            
            # Plot regresjonslinje
            plt.plot(np.concatenate([X.flatten(), fremtidige_år.flatten()]), np.concatenate([y_pred, fremtidige_pred]), color='red', linewidth=2, label='Lineær regresjon')

            for år, pred in zip(fremtidige_år.flatten(), fremtidige_pred):
                plt.text(år, pred, f'{pred:.1f}', ha='center', va='bottom')
            
            plt.title('Figur 6: Årlig gjennomsnittlig temperatur med lineær regresjon')
            plt.xlabel('År')
            plt.ylabel('Gjennomsnittlig temperatur (°C)')
            plt.legend()
            plt.grid(True)
            #flatten gjør fra 2D til 1D, slik at det kan plottes i en graf
            plt.xticks(np.append(X.flatten(), fremtidige_år.flatten()),rotation=90)
            plt.show()
            
            for år, pred in zip(fremtidige_år.flatten(), fremtidige_pred):
                print(f"  År {år}: {pred:.1f} °C")
        
        # Interaktiv widget for antall fremtidige år
        interact(
        oppdater_plot, 
        slutt_år=widgets.IntSlider(
            value=siste_år+5,
            min=siste_år+1,
            max=siste_år+20,
            step=1,
            description='Velg år:',
            continuous_update=False
        )
    )