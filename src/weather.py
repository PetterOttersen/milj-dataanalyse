
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from ipywidgets import interact, widgets
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display, HTML



FILE_PATH = "../raw_data/data/Vaerdata.csv"

class analyse_og_visualisere:
    # Gir filstien til csv-filen
    def __init__(self, file_path="../raw_data/data/Vaerdata.csv"):
        self.FILE_PATH = file_path

    def analyser_vaerdata(self):

        """Funksjonen henter inn datasettet via parameteren self og lagrer den som variabelen data
        
        Parametre: 
        Self

        Returnerer:
        Analysen i en dictionary, Resultater
        """
    
    # Leser data fra csv-filen
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

        """Funksjonen tar inn resultatene fra analyser_vaerdata og plotter temperaturdataene
        
        Parametre:
        resultater: Resultatene fra analyser_vaerdata
        
        Returnerer:
        Gir ut to plot, et for temperatur over tid i et linjediagram og et for 
        gjennomsnittstemperatur per år i et stolpediagram"""

        #Data fra analyser_vaerdata
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

        fig = go.FigureWidget()
        
        # Legg til all data
        fig.add_trace(go.Bar(
            x=temp_årlig_uten_2012.index,
            y=temp_årlig_uten_2012.values.round(2),
            name='Gjennomsnittstemperatur',
            marker_color='Orange',
            text=[f'{value:.2f}°C' for value in temp_årlig_uten_2012.values],
            textposition='outside'
        ))

        # fikser på utseendet til stolpediagrammet
        fig.update_layout(
            title='Årlig gjennomsnittlig temperatur',
            xaxis_title='År',
            yaxis_title='Temperatur (°C)',
            hovermode='x unified',
            showlegend=True,
            xaxis=dict(tickangle=45),
            barmode='group',
            height=600
        )

        display(fig)
        #Print statistikk
        print("Gjennomsnittstemperaturen er:",round(gjennomsnitts_temp,2),"°C")
        print("Median temperaturen er:",round(median_temp,2),"°C")
        print("Standardavviket til temperaturen er:",round(standardavvik_temp),"°C")



    def nedbør(self,resultater):
        """Funksjonen tar inn resultatene fra analyser_vaerdata og plotter nedbørsdataene
        
        Parametre:
        resultater: Resultatene fra analyser_vaerdata
        
        Returnerer:
        gir ut to plot, et for nedbør over tid i et linjediagram og et for
        gjennomsnittsnedbør per år i et stolpediagram"""

        #Data fra analyser_vaerdata
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
        fig = go.FigureWidget()
        
        # Legg til all data
        fig.add_trace(go.Bar(
            x=nedbør_årlig_uten_2012.index,
            y=nedbør_årlig_uten_2012.values.round(2),
            name='Gjennomsnittsnedbør',
            marker_color='Blue',
            text=[f'{value:.2f}mm' for value in nedbør_årlig_uten_2012.values],
            textposition='outside'
        ))

        # Fikser på utseendet til stolpediagrammet
        fig.update_layout(
            title='Årlig gjennomsnittlig nedbør',
            xaxis_title='År',
            yaxis_title='Nedbør (mm)',
            hovermode='x unified',
            showlegend=True,
            xaxis=dict(tickangle=45),
            barmode='group',
            height=600
        )

        display(fig)

        #Print statistikk
        print("Gjennomsnittsnedbøren er:",round(gjennomsnitts_nedbør,2),"mm")
        print("Median nedbøren er:",round(median_nedbør,2),"mm")
        print("Standardavviket til nedbøren er:",round(standardavvik_nedbør),"mm")
        

    def sammenlign_temp_nedbør(self,resultater):
        """Funksjonen tar inn resultatene fra analyser_vaerdata og plotter sammenhengen mellom temperatur og nedbør
        
        Parametre:
        resultater: Resultatene fra analyser_vaerdata
        
        Returnerer:
        Gir ut et scatterplot som viser sammenhengen mellom temperatur og nedbør"""

    # Data fra analyser_vaerdata
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
        """Funksjonen tar inn resultatene fra analyser_vaerdata og utfører en 
        lineær regresjonsanalyse på nedbørsdataene. Deretter plottes historisk data og predektiv data i et scatterplot
        som oppdateres ved bruk av en slider for å velge fremtidige år.
        
        Parametre:
        resultater: Resultatene fra analyser_vaerdata
        
        Returnerer:
        gir ut et scatterplot med historiske nedbørsdata, en regresjonslinje og mulighet for å predikere fremtidig nedbør"""

        # Data fra analyser_vaerdata
        nedbør_data = resultater["nedbør_data"]
        
        # Håndterer manglende verdier
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

        # Opprett figur
        fig = go.FigureWidget()
        
        # Legg til scatterplot for historiske data i figuren
        fig.add_trace(go.Scatter(
            x=X.flatten(),
            y=y.round(2),
            mode='markers',
            name='Historisk nedbør',
            marker=dict(color='blue', size=10),
            text=[f'{value:.2f} mm' for value in y],
            hoverinfo='text'
        ))
        
        # Legger til et tomt spor for fremtidige prediksjoner
        fig.add_trace(go.Scatter(
            x=[],
            y=[],
            mode='markers',
            name='Fremtidig prediksjon',
            marker=dict(color='purple', size=12, symbol='diamond'),
            text=[],
            hoverinfo='text'
        ))
        
        # Regresjonslinje
        fig.add_trace(go.Scatter(
            x=[],
            y=[],
            mode='lines',
            name='Regresjon',
            line=dict(color='red', width=3)
        ))
        
        # fikser på utseendet til scatterplotet
        fig.update_layout(
            title='Årlig gjennomsnittlig nedbør med lineær regresjon',
            xaxis_title='År',
            yaxis_title='Nedbør (mm)',
            hovermode='x unified',
            showlegend=True,
            xaxis=dict(tickangle=45),
            height=600
        )
        
        #Gjør det enkelt å printe ut prediksjoner for riktig antall år
        prediksjon_output = widgets.Output()

        def oppdater_plot(slutt_år_nedbør):
            """Funksjonen oppdaterer scatterplotet når slideren endres
             
            Parametre: 
            slutt_år_nedbør: siste året vi har nedbørsdata
             
            Returnerer:
            Det oppdaterte scatterplotet """

            antall_fremtidige_år = slutt_år_nedbør - siste_år if slutt_år_nedbør > siste_år else 0
            
            # Beregn prediksjoner
            fremtidige_år = np.array([siste_år + i for i in range(1, antall_fremtidige_år + 1)]).reshape(-1, 1)
            fremtidige_pred = model.predict(fremtidige_år) if antall_fremtidige_år > 0 else np.array([])
            
            #gir riktig prediksjoner når det printes
            with prediksjon_output:
                prediksjon_output.clear_output()
                if antall_fremtidige_år > 0:
                    print("Predikert nedbør:")
                    for år, pred in zip(fremtidige_år.flatten(), fremtidige_pred):
                        print(f"  År {år}: {pred:.2f} mm")

            # Oppdater figuren
            with fig.batch_update():
                # Predikerte data
                if antall_fremtidige_år > 0:
                    fig.data[1].x = fremtidige_år.flatten()
                    fig.data[1].y = fremtidige_pred.round(2)
                    fig.data[1].text = [f'{value:.2f} mm' for value in fremtidige_pred]
                else:
                    fig.data[1].x = []
                    fig.data[1].y = []
                    fig.data[1].text = []
                
                # Regresjonslinje
                alle_år = np.concatenate([X.flatten(), fremtidige_år.flatten()]) if antall_fremtidige_år > 0 else X.flatten()
                alle_pred = np.concatenate([y_pred, fremtidige_pred]) if antall_fremtidige_år > 0 else y_pred
                fig.data[2].x = alle_år
                fig.data[2].y = alle_pred.round(2)
                
                # Oppdater x-akse
                fig.update_xaxes(tickvals=alle_år)

        # Oppretter slider
        slider = widgets.IntSlider(
            value=siste_år,
            min=siste_år,
            max=siste_år+20,
            step=1,
            description='Velg år:',
            continuous_update=False
        )
        
        # Viser slideren, figuren og output
        display(widgets.VBox([
            slider,
            fig,
            prediksjon_output
        ]))
        # Knytter slideren til oppdateringsfunksjonen
        widgets.interactive(oppdater_plot, slutt_år_nedbør=slider)
        
   
    
        
    
    def prediksjonsanalyse_temperatur_lineær(self, resultater):
        """Funksjonen tar inn resultatene fra analyser_vaerdata og utfører en 
        lineær regresjonsanalyse på temperaturdataene. Deretter plottes historisk data og predektiv data i et stolpediagram
        som oppdateres ved bruk av en slider for å velge fremtidige år.
         
        Parametre:
        resultater: Resultatene fra analyser_vaerdata
         
        Returnerer
        gir ut et stolpediagram med historiske temperaturdata, en regresjonslinje og mulighet for å predikere fremtidig temperatur """

        #henter data fra analyser_vaerdata
        temp_data = resultater["temp_data"]
    
        if temp_data['value'].isna().any():
            print(f"Advarsel: {temp_data['value'].isna().sum()} manglende verdier blir fylt med årsmiddel")
            temp_data['value'] = temp_data.groupby('År')['value'].transform('mean').fillna(temp_data['value'])

        # Grupper etter år og beregn årlig gjennomsnitt
        temperatur_årlig = temp_data.groupby(pd.to_datetime(temp_data['justertTid']).dt.year)['value'].mean()
        temperatur_årlig_uten_2012 = temperatur_årlig[1:]  # Fjerner 2012
        
        # Forbered data for modellering
        X = temperatur_årlig_uten_2012.index.values.reshape(-1, 1)
        y = temperatur_årlig_uten_2012.values  
        
        # Tren lineær regresjonsmodell
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        siste_år = int(X[-1][0])

        # Oppretter figur
        fig = go.FigureWidget()
        
        # Legger til historiske data i figuren
        fig.add_trace(go.Bar(
            x=X.flatten(),
            y=y.round(2),
            name='Historisk',
            marker_color='gray',
            text=[f'{value:.2f}°C' for value in y],
            textposition='outside'
        ))
        
        # Legger til et tomt spor for fremtidige prediksjoner
        fig.add_trace(go.Bar(
            x=[],
            y=[],
            name='Prediksjon',
            marker_color='purple',
            text=[],
            textposition='outside'
        ))
        # Regresjonslinje
        fig.add_trace(go.Scatter(
            x=[],
            y=[],
            mode='lines',
            name='Regresjon',
            line=dict(color='red', width=3)
        ))
        
        # fikser på utseendet til figuren
        fig.update_layout(
            title='Årlig gjennomsnittlig temperatur med lineær regresjon',
            xaxis_title='År',
            yaxis_title='Temperatur (°C)',
            hovermode='x unified',
            showlegend=True,
            xaxis=dict(tickangle=45),
            barmode='group',
            height=600
        )
        
        # Gjør det enkelt å printe ut prediksjoner for riktig antall år
        prediksjon_output = widgets.Output()

        def oppdater_plot(slutt_år_temp):
            """Funksjonen oppdaterer stolpediagrammet når slideren endres
             
            Parametre: 
            slutt_år_temp: siste året vi har temperaturdata
             
            Returnerer:
            Det oppdaterte stolpediagrammet """

            antall_fremtidige_år = slutt_år_temp - siste_år if slutt_år_temp > siste_år else 0
            
            # Beregn prediksjoner
            fremtidige_år = np.array([siste_år + i for i in range(1, antall_fremtidige_år + 1)]).reshape(-1, 1)
            fremtidige_pred = model.predict(fremtidige_år) if antall_fremtidige_år > 0 else np.array([])
            
            # Gjør det enkelt å printe ut data til riktig år
            with prediksjon_output:
                prediksjon_output.clear_output()
                if antall_fremtidige_år > 0:
                    print("Predikerte temperaturer:")
                    for år, pred in zip(fremtidige_år.flatten(), fremtidige_pred):
                        print(f"  År {år}: {pred:.2f} °C")

            # Oppdater figuren
            with fig.batch_update():
                # Predikerte data
                if antall_fremtidige_år > 0:
                    fig.data[1].x = fremtidige_år.flatten()
                    fig.data[1].y = fremtidige_pred.round(2)
                    fig.data[1].text = [f'{value:.2f}°C' for value in fremtidige_pred]
                else:
                    fig.data[1].x = []
                    fig.data[1].y = []
                    fig.data[1].text = []
                
                # Regresjonslinje
                alle_år = np.concatenate([X.flatten(), fremtidige_år.flatten()]) if antall_fremtidige_år > 0 else X.flatten()
                alle_pred = np.concatenate([y_pred, fremtidige_pred]) if antall_fremtidige_år > 0 else y_pred
                fig.data[2].x = alle_år
                fig.data[2].y = alle_pred.round(2)
                
                # Oppdater x-akse
                fig.update_xaxes(tickvals=alle_år)

        # Oppretter slider
        slider = widgets.IntSlider(
            value=siste_år,
            min=siste_år,
            max=siste_år+20,
            step=1,
            description='Velg år:',
            continuous_update=False
        )
        
        # Viser slideren, figuren og output
        display(widgets.VBox([
            slider,
            fig,
            prediksjon_output
        ]))
        # Knytter slideren til oppdateringsfunksjonen
        widgets.interactive(oppdater_plot, slutt_år_temp=slider)
        
        