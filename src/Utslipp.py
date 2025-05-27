#Henter inn relevante bibloteker

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from ipywidgets import interact, widgets




#Henter csv filen og leser den

file_path = "raw_data/data/Utslippdata.csv"
df = pd.read_csv(file_path) 




def analyze_clean_utslipp_data(df):
    """
    Rydder datasettet, beholder relevante kolonner og fjerner rader for å gjøre videre datahåndtering lettere
    
    Parametre:
    df : Datasettet
    
    Returnerer:
    Renset dataset
    """

     
    #Dataen inneholder både heltall og tekstrenger. Datasettet ligger i grupper
    
    df.columns = ['kilde', 'energiprodukt', 'komponent', 'år', 'statistikkvariabel', 'verdi'] 

    df = df[['kilde', 'energiprodukt', 'komponent', 'år', 'verdi']] 

    #Fjerner alle kilder for å gjøre statistikk beregninger lettere 
    
    all_sources = df[df['kilde'].str.contains("0 Alle kilder", na=False)].index 
    df = df.drop(all_sources)
    return df

#Oppretter en klasse for statitiske plots


    


class statitics_plot: 

    #Funskjon som henter inn dataen inn i en dataframe, og lagrer den 
    def __init__(self, df): 
        """
        Funskjon som henter inn dataen inn i en dataframe, og lagrer den som self.df inn i klassen
        
        Parametre:
        Self : .....
        df : datafilen

        Returnerer:
        Datafilen inn i klassen
        """

        self.df = df 


    def plot_co2_per_year_mean(self):
        """
        Finner gjennomsnittlig utslipp per år og plotter det 
        
        Parametre: 
        Self : Et objekt i klassen

        Returnerer:
        Plot av gjennomsnittlige utslipp per år 
        """

        #Grupperer radene i dataframe etter verdier i kolonnen år
        #Mean() beregner gjennomsnitt for hver verdi inennfor gruppe hvert år gruppe
        co2_per_year_mean = self.df.groupby('år')['verdi'].mean() 

        
        #Plotter figuren
        plt.figure(figsize=(10, 6))
        co2_per_year_mean.plot(kind='bar', title="Figur 1: CO2-utslipp over tid (gjennomsnitt)")
        plt.ylabel("Utslipp (1000 tonn CO2-ekv.)")
        plt.xlabel("År")
        plt.tight_layout()
        plt.grid()
        plt.show()

        return co2_per_year_mean
        


    def plot_co2_per_year_median(self):

        """
        Finner median av utslipp per år og plotter det 
        
        Parametre: 
        Self : Et objekt i klassen

        Returnerer:
        Plot av median av utslipp per år 
        """

        co2_per_year_median = self.df.groupby('år')['verdi'].median().reset_index()

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=co2_per_year_median, x = 'år', y = 'verdi', marker = 'o', color = 'green')
        plt.title('Figur 2: Co2 utslipp over tid i median')
        plt.ylabel("Utslipp (1000 tonn CO2-ekv.)")
        plt.xlabel("År")
        plt.grid()
        plt.tight_layout()
        plt.show()


    def plot_co2_per_source_mean(self):
        """
        Finner gjennomsnittlige utslipp i hele perioden per kilde og plotter det 
        
        Parametre: 
        Self : Et objekt i klassen

        Returnerer:
        Plottet figur av gjennomsnittlige utslipp i hele perioden per kilde
        """

        co2_per_source_median = self.df.groupby('kilde')['verdi'].mean() 
        
        plt.figure(figsize=(10, 6))
        co2_per_source_median.plot(kind='bar', title="Figur 3: CO2-utslipp per kilde gjennomsnitt")
        plt.ylabel("Utslipp (1000 tonn CO2-ekv.)")
        plt.xlabel("Kilde")
        plt.tight_layout()
        plt.show()




    def plot_co2_per_year_std(self):
        """
        Finner standardavvik av utslipp i hele perioden per kilde plotter det 
        
        Parametre: 
        Self : Et objekt i klassen

        Returnerer:
        Plottet figur av standardavvik utslipp i hele perioden per kilde
        """
        
        co2_per_year_std = self.df.groupby('kilde')['verdi'].std() 
        
        plt.figure(figsize=(10, 6))
        co2_per_year_std.plot(kind='bar', title="Figur 4: CO2-utslipp per kilde (standardavvik)")
        plt.ylabel("Utslipp (1000 tonn CO2-ekv.)")
        plt.xlabel("Kilde")
        plt.tight_layout()
        plt.show()
        
    
   

        #Varmekart visualisering
    def plot_co2_source_year_hm(self):
        
        """
        Lager et heatmap av CO2 per kilde per år
        
        Parametre: 
        Self : Et objekt i klassen

        Returnerer:
        Plottet figur av heatmap utslipp 
        """
        sns.set_theme()
        #Lager en kopi av datasettet 
        df_log = self.df.copy()

        #Transfomerer verdier i ['verdi'] ved logaritme 
        df_log["verdi_log"] = np.log10(df_log["verdi"].replace(0, np.nan))
        
        co2_source_year_hm = (df_log.pivot(index="kilde", columns="år", values="verdi_log"))

        
        f, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(co2_source_year_hm, annot=True, fmt=".2f", linewidths=.5,cmap = "YlGnBu", ax=ax,cbar_kws={'label': 'log10(Utslipp i tonn)'})
        plt.xlabel("År",size = 11)        
        plt.ylabel("CO2 utslipp kilder", size = 11) 
        plt.title("Figur 5: Varmekart over kilder, år og mengden av utslipp",size = 16)
        plt.show()
        


#Oppretter en ny klasse for regresjonsanalyse

class plots_part_2: 

    def __init__(self, df):     
        self.df = df #ai
    

    def linreg_train_test(self):
        """
        Lineær regresjon ved bruk av to metoder training 50%, og training på 100 % av datasettet
        
        Parametre: 
        Self : Et objekt i klassen

        Returnerer:
        scaler_full : Standardrisesere hele datasettet 
        model_full : Regresjonsmodell som brukes til å lage prediksjoner 
        X :  år kolonnen
        X_scaled_full : Skalert type av X brukt for hele datasettet

        ...
        """
        
        #Aggregerer data og henter inn år og utslippsmengde i grupper
        df_groupby = self.df.groupby('år')['verdi'].mean().reset_index() #ai
        
        #Definerer funksjonene X og variabelen y 
        
        def regression_model():
            """
            Lineær regresjon 
            
            Parametre: 
            Self : Et objekt i klassen

            Returnerer: 
            X: år kolonnen
            X_test : Tester 50 % av dataen 
            y_test : Ekte målingen fra datasettet
            y_test_pred : Predikerte verdier 50 % at datasettet, viser hvordan modellen forutsier data.
            y_full_pred : Predikerte verdier for 100 % av datasettet
            scaler_full : Standardrisesere hele datasettet 
            model_full : Regresjonsmodell som brukes til å lage prediksjoner 
            X_scaled_full : Skalert type av X brukt for hele datasettet

            """

        
            X =  df_groupby[["år"]] 
            y = df_groupby["verdi"] 


            #Deler regresjonen i train og test, hvor testdata er 50 % av dataen 
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)

            #Skalerer dataen og tranformerer treningsdataen og testdataen
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)


            #Trener modellen 
            model_train = LinearRegression()
            model_train.fit(X_train_scaled, y_train)
            y_test_pred = model_train.predict(X_test_scaled)

            #Skalerer slik den tester hele 100% av dataen 
            scaler_full = StandardScaler() #ai
            X_scaled_full = scaler_full.fit_transform(X)
            
            #Trener 100% av dataen
            model_full = LinearRegression()
            model_full.fit(X_scaled_full, y)
            y_full_pred = model_full.predict(X_scaled_full)

            #Returnerer for fremtidig implementasjon av disse av følgende variabler
            return X, X_test, y_test, y_test_pred, y_full_pred, scaler_full, model_full, X_scaled_full
      
       #Kaller på variablene slik at linreg_train_test gjenkjenner de
        X, X_test, y_test, y_test_pred, y_full_pred, scaler_full, model_full, X_scaled_full = regression_model()

        
        #Plotter
        plt.plot(X_test, y_test_pred, color="green", label=" 50 % Prediction")
        plt.scatter(X_test, y_test, label="Test verdier")
        plt.plot(X, y_full_pred, color="red", label="100 % Prediksjon")
        plt.xlabel("År")
        plt.ylabel("Verdi")
        plt.title("Figur 6: Lineær regresjon")
        plt.legend()
        plt.grid(True)
        plt.show()
        
        #Beregner R2-score 
        r2 = r2_score(y_test, y_test_pred)
        print("r2 = ",r2)
        
        #Returnerer nødvendige variabler
        return scaler_full, model_full, X, X_scaled_full

    
    def barplot(self):
        """
        Lineær regresjon trent på 100 % på datasettet. 
            
        Parametre: 
        Self : Et objekt i klassen

        Returnerer: 
        Regresjon av kilde og verdi fremstilt som barplot 

        """
        
        
        df_groupby = self.df.groupby('kilde')['verdi'].mean().reset_index()#mean
        
        #Bruker pd.get_dummies for å transfomere tekst til binære tall
        X =  pd.get_dummies(df_groupby[["kilde"]]) #ai
        y = df_groupby["verdi"] 

        
        model = LinearRegression()
        model.fit(X, y)

        y_pred = model.predict(X)

        plt.figure(figsize=(12, 6))
        sns.barplot(x="kilde",y= y_pred,data = df_groupby,color = "magenta") 
        plt.xticks(rotation=45, ha = "right")
        plt.xlabel("Kilde")
        plt.ylabel("Verdi")
        plt.title("Figur 7: Linreg av Kilde og Verdi")
        plt.tight_layout()
        plt.grid(True)
        plt.show()
    

    def futureplot(self,scaler_full, model_full, X, X_scaled_full):
        """
        Bruker regresjon til å fremstille fremtiden 
        
        Parametre: 
        Self : Et objekt i klassen
        scaler_full : Standardrisesere hele datasettet
        model_full : Regresjonsmodell som brukes til å lage prediksjoner
        X : år kolonnen
        X_scaled_full :  Skalert type av X brukt for hele datasettet

        Returnerer:
        Blir en barplot 

        """


        #Aggregerer data og henter inn år og utslippsmengde i grupper
        #Bruker reset index slik sckit kan lese dataen 
        df_groupby = self.df.groupby('år')['verdi'].mean().reset_index() #ai
            
        #Finner det siste / høyeste året i kolonnen år
        siste_år_data = df_groupby["år"].max()  


        def oppdater_plot(slutt_år):
            """

            Lager en fremstilling av hitorisk data samt fremtidige prediksjoner 
            Parametre: 
            Self : Et objekt i klassen
            

            Returnerer:
            Viser en graf med historisk data samt fremtidige prediksjoner 
            """

            #Regner ut år i fremtiden som skal predikeres

            antall_fremtidige_år=slutt_år-siste_år_data if slutt_år-siste_år_data>0 else 0 #ai

            #Lager array med år i fremtiden 
            future_years = np.arange(siste_år_data + 1, siste_år_data + antall_fremtidige_år+1)#ai

            #Lager en dataframe til prediskjon
            future_df = pd.DataFrame({"år": future_years})#ai

            #Bruker skaler som ble brukt tidligere
            future_scaled = scaler_full.transform(future_df)#ai

            #Predikerer fremtidige år ved bruk av predefinert model
            future_preds = model_full.predict(future_scaled)#ai

            plt.figure(figsize=(10, 6))
            plt.plot(X, model_full.predict(X_scaled_full), color="green", label="Historisk trend") 

            #Bruker betingelse for å finne om det skal printes neste år eller ikke 
            if antall_fremtidige_år>=0: #ai
                plt.plot(future_df, future_preds, color="red", linestyle="--", marker="x", label="Fremtidsprediksjon")

            plt.axvline(x=max(X["år"]), linestyle=":", color="gray")
            plt.xlabel("År")
            plt.ylabel("Verdi")
            plt.title("Figur 8: Lineær regresjon med fremtidige prediksjoner")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        #Interaktiv widget for antall fremtidige år
        interact(
        oppdater_plot, 
        slutt_år=widgets.IntSlider(
            value=siste_år_data+1,
            min=siste_år_data+1,
            max=siste_år_data+10,
            step=1,
            description='Velg år:',
            continuous_update=False
        )
    )


#Lager en kopi av datasetett
df_with_nans = df.copy()
#Bruker random 
np.random.seed(42)

#Erstatter verdier med nan
missing_idx = np.random.choice(df_with_nans.index, size=int(0.3 * len(df_with_nans)), replace=False)
df_with_nans.loc[missing_idx, "verdi"] = np.nan

class MissingValues:
    def __init__(self, df, legg_til_nans=False, andel_nans=0.3, seed=42): 
        """
        Bearbeider datasettet med nanverdier
        
        Parametre: 
        self : Et objekt i klassen
        df : datafilen
        legg_til _nans=False : legger til Nan verdier i datasettet 
        andel_nans=0.3 : Prosentandel av dataen som skal ersattes med nan verdier
        seed=42 : Startverdi for tilfeldig tallgenerator

        Returnerer:
        Dataframe med muligens nanverdier
        """
        #Lagrer datasettet
        self.df = df.copy()
        if legg_til_nans:
            np.random.seed(seed)
            missing_idx = np.random.choice(
                self.df.index, size=int(andel_nans * len(self.df)), replace=False
            )
            self.df.loc[missing_idx, "verdi"] = np.nan

    def remove_random_data(self, andel=0.5, seed=None):
        """
        Bearbeider datasettet med nanverdier
        
        Parametre: 
        self : Et objekt i klassen
        andel = 0.5 : 50 % av datasettet med nan verdier
        seed = none : Startverdi for tilfeldig tallgenerator
        
        Returnerer:
        Bearbeidet datasett med nanverdier
        """
        #Sjekker riktig andel for nan verdier
        if not 0 < andel < 1:
            raise ValueError("Andel må være mellom 0 og 1.")
        #Velger en viss andel data og fjerner denne dataen til nanverdier
        self.df = self.df.drop(self.df.sample(frac=andel, random_state=seed).index)
        return self.df

    def plot_missing_data(self, verdi_kolonne='verdi', år_kolonne='år'):
        """
        Bearbeider datasettet med nanverdier
        
        Parametre: 
        self : Et objekt i klassen
        verdi_kolonne = 'verdi' : 50 % av datasettet med nan verdier
        seed = none : Startverdi for tilfeldig tallgenerator
        
        Returnerer:
        Bearbeidet datasett med nanverdier
        """
        #
        if verdi_kolonne not in self.df.columns or år_kolonne not in self.df.columns:
            raise KeyError(f"DataFrame må inneholde kolonnene '{verdi_kolonne}' og '{år_kolonne}'.")

        # Del opp datasettet
        complete_cases = self.df.dropna(subset=[verdi_kolonne])
        incomplete_cases = self.df[self.df[verdi_kolonne].isnull()]

        # Imputer manglende verdier med gjennomsnitt
        imputer = SimpleImputer(strategy='mean')
        df_imputed = self.df.copy()
        df_imputed[[verdi_kolonne]] = imputer.fit_transform(df_imputed[[verdi_kolonne]])

        # Lineær regresjon
        X = df_imputed[[år_kolonne]]
        y = df_imputed[verdi_kolonne]
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        plt.figure(figsize=(10, 6))
        plt.scatter(complete_cases[år_kolonne], complete_cases[verdi_kolonne],
                    label='Fullstendige rader', color='blue')
        
        if not incomplete_cases.empty:
            imputert = df_imputed.loc[incomplete_cases.index]
            plt.scatter(
                incomplete_cases[år_kolonne],
                imputert[verdi_kolonne],
                label='Imputerte verdier',
                color='orange',
                marker='x',
                s=100
            )
        #Plotting
        plt.plot(X, y_pred, label='Lineær regresjon', color='green')
        plt.xlabel("År")
        plt.ylabel("Verdi")
        plt.title("Figur 9: Lineær regresjon og imputering av manglende data")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

