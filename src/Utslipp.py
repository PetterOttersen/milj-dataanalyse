import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


file_path_utslipp = "../Excelfil_utslipp.xlsx"
df = pd.read_excel(file_path_utslipp)

def analyze_clean_utslipp_data():
    df.columns = ['index', 'kilde', 'energiprodukt', 'komponent', 'år', 'statistikkvariabel', 'verdi'] #ai
    df = df[['kilde', 'energiprodukt', 'komponent', 'år', 'verdi']] #ai
    alle_kilder = df[df['kilde'].str.contains("0 Alle kilder", na=False)].index #retting av ai
    df = df.drop(alle_kilder)



def descriptive_statistics():
    co2_per_year_mean = df.groupby('år')['verdi'].mean() #gjennomsnittlige utslipp per år #ai
    co2_per_year_median = df.groupby('år')['verdi'].median() #median av utslipp per år
    co2_per_source_median = df.groupby('kilde')['verdi'].mean() #medi
    co2_per_year_std = df.groupby('år')['verdi'].std() 
    
    return co2_per_year_mean, co2_per_year_median, co2_per_source_median, co2_per_year_std


class plots: 

def plot_co2_per_year_mean():
    plt.figure(figsize=(10, 6))
    co2_per_year_mean.plot(kind='bar', title="CO2-utslipp over tid (gjennomsnitt)")
    plt.ylabel("Utslipp (1000 tonn CO2-ekv.)")
    plt.xlabel("År")
    plt.tight_layout()
    plt.show()


def plot_co2_per_year_median():
    plt.figure(figsize=(10, 6))
    co2_per_source_median.plot(kind='bar', title="CO2-utslipp over tid (median)")
    plt.ylabel("Utslipp (1000 tonn CO2-ekv.)")
    plt.xlabel("År")
    plt.tight_layout()
    plt.show()


def plot_co2_per_source_median():
    plt.figure(figsize=(10, 6))
    co2_per_source_median.plot(kind='bar', title="CO2-utslipp per kilde (gjennomsnitt)")
    plt.ylabel("Utslipp (1000 tonn CO2-ekv.)")
    plt.xlabel("Kilde")
    plt.tight_layout()
    plt.show()

def plot_co2_per_year_std():
    plt.figure(figsize=(10, 6))
    co2_per_år_std.plot(kind='bar', title="CO2-utslipp per kilde (gjennomsnitt)")
    plt.ylabel("Utslipp (1000 tonn CO2-ekv.)")
    plt.xlabel("Kilde")
    plt.tight_layout()
    plt.show()


#Last inn data
#Dataen inneholder både heltall og tekstrenger. Datasettet ligger i grupper 
#print(df)
#Rydder datasettet, beholder relevante kolonner og fjerner rader for å gjøre videre datahåndtering lettere
