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


#Rydder datasettet, beholder relevante kolonner og fjerner rader for å gjøre videre datahåndtering lettere

def analyze_clean_utslipp_data(df):
     
     #ai #Dataen inneholder både heltall og tekstrenger. Datasettet ligger i grupper
    
    df.columns = ['kilde', 'energiprodukt', 'komponent', 'år', 'statistikkvariabel', 'verdi'] 

    df = df[['kilde', 'energiprodukt', 'komponent', 'år', 'verdi']] #ai

    #Fjerner alle kilder for å gjøre statistikk beregninger lettere 
    
    all_sources = df[df['kilde'].str.contains("0 Alle kilder", na=False)].index 
    df = df.drop(all_sources)
    return df

#Oppretter en klasse for statitiske plots

class plots: 

    #Funskjon som henter inn dataen inn i en dataframe, og lagrer den ai
    def __init__(self, df): 
        self.df = df 


    def plot_co2_per_year_mean(self):

        #Grupperer radene i dataframe etter verdier i kolonnen år
        #Mean() beregner gjennomsnitt for hver verdi inennfor gruppe hvert år gruppe
        co2_per_year_mean = self.df.groupby('år')['verdi'].mean() 
        
        #Plotter figuren
        plt.figure(figsize=(10, 6))
        co2_per_year_mean.plot(kind='bar', title="Figur 1: CO2-utslipp over tid (gjennomsnitt)")
        plt.ylabel("Utslipp (1000 tonn CO2-ekv.)")
        plt.xlabel("År")
        plt.tight_layout()
        plt.show()
        return co2_per_year_mean


    def plot_co2_per_year_median(self):

        co2_per_year_median = self.df.groupby('år')['verdi'].median().reset_index()

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=co2_per_year_median, x = 'år', y = 'verdi', marker = 'o', color = 'green')
        plt.title('Co2 utslipp over tid i median')
        plt.ylabel("Utslipp (1000 tonn CO2-ekv.)")
        plt.xlabel("År")
        plt.grid()
        plt.tight_layout()
        plt.show()


    def plot_co2_per_source_mean(self):
        

        co2_per_source_median = self.df.groupby('kilde')['verdi'].mean() 
        
        plt.figure(figsize=(10, 6))
        co2_per_source_median.plot(kind='bar', title="CO2-utslipp per kilde gjennomsnitt")
        plt.ylabel("Utslipp (1000 tonn CO2-ekv.)")
        plt.xlabel("Kilde")
        plt.tight_layout()
        plt.show()




    def plot_co2_per_year_std(self):
        
        #
        co2_per_year_std = self.df.groupby('år')['verdi'].std() 
        
        plt.figure(figsize=(10, 6))
        co2_per_year_std.plot(kind='bar', title="CO2-utslipp per kilde (standardavvik)")
        plt.ylabel("Utslipp (1000 tonn CO2-ekv.)")
        plt.xlabel("Kilde")
        plt.tight_layout()
        plt.show()
        return co2_per_year_std
    
    #Henter inn to nye parametre for å beregne forholdet mellom standarvavvik og gjennomsnitt
    def comparisons(self,co2_per_year_std, co2_per_year_mean):
        

        CV = co2_per_year_std / co2_per_year_mean
        
        plt.figure(figsize=(10, 3))
        plt.grid(True)
        CV.plot(title="CO2-utslipp over tid (median)")
        plt.ylabel("Utslipp (1000 tonn CO2-ekv.)")
        plt.xlabel("Kilde")
        plt.show()


    #Varmekart visualisering
    def plot_co2_source_year_hm(self):
        
        sns.set_theme()


        #Lager en kopi av datasettet 
        df_log = self.df.copy()

        #Transfomerer verdier i ['verdi] ved logaritme 
        df_log["verdi_log"] = np.log10(df_log["verdi"].replace(0, np.nan))
        
        
        co2_source_year_hm = (df_log.pivot(index="kilde", columns="år", values="verdi"))

        
        f, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(co2_source_year_hm, annot=True, fmt=".0f", linewidths=.5, ax=ax)
        plt.xlabel("År",size = 11)        
        plt.ylabel("CO2 utslipp kilder", size = 11) 
        plt.title("Varmekart over kilder, år og mengden av utslipp",size = 16)
        plt.show()


#Oppretter en ny klasse for regresjonsanalyse

class plots_part_2: 

    def __init__(self, df): #ai
        self.df = df #ai
    
    def linreg_train_test(self):
        #
        df_groupby = self.df.groupby('år')['verdi'].mean().reset_index() #ai
        
        X =  df_groupby[["år"]] #ai
        y = df_groupby["verdi"] 
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
       
        model_train = LinearRegression()
        model_train.fit(X_train_scaled, y_train)
       
        y_train_pred = model_train.predict(X_test_scaled)


        model = LinearRegression()
        model.fit(X, y)

        y_pred = model.predict(X)

        plt.plot(X, y_pred, color="green", label="Prediction")
        plt.scatter(X_test, y_test, label="Test data")
        plt.plot(X_test, y_train_pred, color="red", label="Prediction")
        plt.xlabel("År")
        plt.ylabel("Verdi")
        plt.title("Lineær regresjon")
        plt.legend()
        plt.grid(True)
        plt.show()

        r2 = r2_score(y_test, y_train_pred)
        print("r2 = ",r2)
   

    def barplot(self):
        df_groupby = self.df.groupby('kilde')['verdi'].mean().reset_index()#mean
        
        
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
        plt.title("Linreg av Kilde og Verdi")
        plt.tight_layout()
        plt.grid(True)
        plt.show()
    
    def futureplot(self):
        df_groupby = self.df.groupby('år')['verdi'].mean().reset_index() #ai
        
        X =  df_groupby[["år"]] #ai
        y = df_groupby["verdi"] 
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
       
        model_train = LinearRegression()
        model_train.fit(X_train_scaled, y_train)
        y_train_pred = model_train.predict(X_test_scaled)


        scaler_full = StandardScaler() #ai
        X_scaled_full = scaler_full.fit_transform(X)
        model_full = LinearRegression()
        model_full.fit(X_scaled_full, y)

        siste_år_data = df_groupby["år"].max()  # Henter det siste året fra dataene

        def oppdater_plot(slutt_år):

            antall_fremtidige_år=slutt_år-siste_år_data if slutt_år-siste_år_data>0 else 0
            future_years = np.arange(siste_år_data + 1, siste_år_data + antall_fremtidige_år+1)

            future_df = pd.DataFrame({"år": future_years})
            future_scaled = scaler_full.transform(future_df)
            future_preds = model_full.predict(future_scaled)#ai

            plt.figure(figsize=(10, 6))
            plt.plot(X, model_full.predict(X_scaled_full), color="green", label="Historisk trend")
            if antall_fremtidige_år>0:
                plt.plot(future_df, future_preds, color="red", linestyle="--", marker="x", label="Fremtidsprediksjon")

            plt.axvline(x=max(X["år"]), linestyle=":", color="gray")
            plt.xlabel("År")
            plt.ylabel("Verdi")
            plt.title("Lineær regresjon med fremtidige prediksjoner")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

         # Interaktiv widget for antall fremtidige år
        interact(
        oppdater_plot, 
        slutt_år=widgets.IntSlider(
            value=siste_år_data+1,
            min=siste_år_data+1,
            max=siste_år_data+20,
            step=1,
            description='Velg år:',
            continuous_update=False
        )
    )

from sklearn.impute import SimpleImputer


class missing_values:
    def __init__(self, df): 
        self.df = df 

    
    def remove_random_data(self, andel = 0.1,seed = None):#ai
        if  not 0 < andel < 1:
            raise ValueError("Andel må være mellom 0 og 1.")
    
        df_renset = self.df.drop(self.df.sample(frac=andel, random_state=seed).index)
        self.df = df_renset
        return self.df
    
    #ai



    def plot_missing_data(self):
        
        df_groupby = self.df.groupby('år')['verdi'].mean().reset_index() #ai

        complete_cases = self.df.dropna()
        incomplete_cases = self.df[df.isnull().any(axis=1)]

        imputer = SimpleImputer(strategy='mean')
        df_imputed  = self.df.copy()
        df_imputed[['verdi']] = imputer.fit_transform(df_imputed[["verdi"]])
 
        X = df_imputed[['år']]
        y = df_imputed['verdi']
       

        model = LinearRegression()
        model.fit(X ,y)
        y_pred = model.predict(X)


        plt.figure(figsize=(10, 6))
        plt.scatter(complete_cases['år'], complete_cases['verdi'], label='Fullstendige rader', color='blue')
        
        if not incomplete_cases.empty:
            imputert = model.predict(incomplete_cases[['år']])
            plt.scatter(incomplete_cases['år'], imputert, label='Imputerte verdier', color='orange')

        plt.plot(X, y_pred, label="Imputerte verdier",color="green")
        plt.axvline(x=max(X["år"]), linestyle=":", color="gray")
        plt.xlabel("År")
        plt.ylabel("Verdi")
        plt.title("Lineær regresjon med fremtidige prediksjoner")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


class research:
    
    def two_source_comparison(self, source1,source2, year):
        #source_year_value = self.df.groupby(['kilde','år'])['verdi'].mean()
        filtered = self.df[(self.df['år'] == year) & (self.df['kilde'].isin([source1, source2]))] #ai
        
        result = filtered.groupby('kilde')['verdi'].mean()

        return result 
    """
    def sammenligne_brukervalg(self):
        print("Tilgjengelige kilder:\n")
        for kilde in sorted(self.df['kilde'].unique()):
            print("-", kilde)

        source1 = input('Velg første kilde')
        source2 = input('Velg andre kilde')

        print(self.df('kilde')).unique()
        filtered = self.df[(self.df['år'] == year) & (self.df['kilde'].isin([source1, source2]))] #ai

        if filtered.empty:
            return("Ingen data funnet")
    
        result = filtered.groupby('kilde')['verdi'].mean()


        sns.set_style("whitegrid")
        plt.figure(figsize=(8, 5))
        sns.barplot(x=result.index, y=result.values)
        plt.title(f"CO₂-utslipp i {year}: {source1} vs {source2}")
        plt.ylabel("Gjennomsnittlig utslipp (1000 tonn CO₂-ekvivalenter)")
        plt.xlabel("Kilde")
        plt.tight_layout()
        plt.show()

        return result
"""





    
    









        

        

















"""
   
   
    def linreg_test(self):
        df_groupby = self.df.groupby('år')['verdi'].mean().reset_index() #ai
        X =  df_groupby[["år"]] #ai
        y = df_groupby["verdi"]
        scaler = StandardScaler()
        model = LinearRegression()
        model.fit(X, y)

        y_pred = model.predict(X)

        plt.plot(X, y_pred, color="red", label="Prediction")
        plt.xlabel("År")
        plt.ylabel("Verdi")
        plt.title("Lineær regresjon")
        plt.legend()
        plt.grid(True)
        plt.show()

"""

    
 
    


    











