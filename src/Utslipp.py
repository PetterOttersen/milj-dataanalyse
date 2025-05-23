import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
from sklearn import datasets



file_path = "raw_data/data/Utslippdata.csv"
df = pd.read_csv(file_path) #Last inn data



def analyze_clean_utslipp_data(df):
    df.columns = ['kilde', 'energiprodukt', 'komponent', 'år', 'statistikkvariabel', 'verdi'] #ai #Dataen inneholder både heltall og tekstrenger. Datasettet ligger i grupper 

    df = df[['kilde', 'energiprodukt', 'komponent', 'år', 'verdi']] #ai
    alle_kilder = df[df['kilde'].str.contains("0 Alle kilder", na=False)].index #retting av ai
    df = df.drop(alle_kilder)
    return df

#Rydder datasettet, beholder relevante kolonner og fjerner rader for å gjøre videre datahåndtering lettere

class plots: 

    def __init__(self, df): #ai
        self.df = df #ai

    def plot_co2_per_year_mean(self):
        
        co2_per_year_mean = self.df.groupby('år')['verdi'].mean() #gjennomsnittlige utslipp per år #ai
        
        plt.figure(figsize=(10, 6))
        co2_per_year_mean.plot(kind='bar', title="CO2-utslipp over tid (gjennomsnitt)")
        plt.ylabel("Utslipp (1000 tonn CO2-ekv.)")
        plt.xlabel("År")
        plt.tight_layout()
        plt.show()
        return co2_per_year_mean


    def plot_co2_per_year_median(self):
        
        co2_per_year_median = self.df.groupby('år')['verdi'].median() #median av utslipp per år
        
        plt.figure(figsize=(10, 6))
        co2_per_year_median.plot(kind='bar', title="CO2-utslipp over tid (median)")
        plt.ylabel("Utslipp (1000 tonn CO2-ekv.)")
        plt.xlabel("År")
        plt.tight_layout()
        plt.show()


    def plot_co2_per_source_median(self):
        
        co2_per_source_median = self.df.groupby('kilde')['verdi'].mean() #medi
        
        plt.figure(figsize=(10, 6))
        co2_per_source_median.plot(kind='bar', title="CO2-utslipp per kilde (median)")
        plt.ylabel("Utslipp (1000 tonn CO2-ekv.)")
        plt.xlabel("Kilde")
        plt.tight_layout()
        plt.show()

    def plot_co2_per_year_std(self):
        
        co2_per_year_std = self.df.groupby('år')['verdi'].std() 
        
        plt.figure(figsize=(10, 6))
        co2_per_year_std.plot(kind='bar', title="CO2-utslipp per kilde (standardavvik)")
        plt.ylabel("Utslipp (1000 tonn CO2-ekv.)")
        plt.xlabel("Kilde")
        plt.tight_layout()
        plt.show()
        return co2_per_year_std

    def comparisons(self,co2_per_year_std, co2_per_year_mean):
        
        CV = co2_per_year_std / co2_per_year_mean
        
        plt.figure(figsize=(10, 3))
        plt.grid(True)
        CV.plot(title="CO2-utslipp over tid (median)")
        plt.ylabel("Utslipp (1000 tonn CO2-ekv.)")
        plt.xlabel("Kilde")
        plt.show()

    def plot_co2_source_year_hm(self):
        
        sns.set_theme()
        co2_source_year_hm = (self.df.pivot(index="kilde", columns="år", values="verdi"))
        
        f, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(co2_source_year_hm, annot=True, fmt=".0f", linewidths=.5, ax=ax)
        plt.xlabel("År",size = 11)        
        plt.ylabel("CO2 utslipp kilder", size = 11) 
        plt.title("Varmekart over kilder, år og mengden av utslipp",size = 16)
        plt.show()



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score



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
        df_groupby = self.df.groupby('kilde')['verdi'].mean().reset_index() #mean
        
        X =  df_groupby[["kilde"]] #ai
        y = df_groupby["verdi"] 

        model = LinearRegression()
        model.fit(X, y)

        y_pred = model.predict(X)
       
        plt.plot(x = X, y = y,kind = "bar")
        plt.xlabel("År")
        plt.ylabel("Verdi")
        plt.title("Linreg")
        plt.legend()
        plt.grid(True)
        plt.show()
   
   
   
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

    
 
    


    











