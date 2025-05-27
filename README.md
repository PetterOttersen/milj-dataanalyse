# Prosjektbeskrivelse

I denne oppgaven fremstilles data hentet fra yr og ssb, dataen er hentet fra hele Norge. Værdataene er hentet fra værstasjoner i Kristiansand og Trondheim. I dette programmet vil vi se på data fra 2013 til 2024, og gi en prediktiv analyse av hvordan tallene kan se ut i fremtiden. 

## Installasjonsbeskrivelse

For å installere skriv "pip install -r requirements.txt"


## Bruksinstruks
For best mulig gjennomkjøring bruk VS code eller jupiter notebook, da interaktive visualiseringer ikke vil vises i github.

1. Kjør først [getData.py](raw_data/getData.py) og [dataCleaning.py](raw_data/dataCleaning.py)
2. Deretter kan src filene kjøres [Utslipp.py](src/Utslipp.py) og [weather.py](src/weather.py)
3. For å få opp visualiseringene må [Fremstilling_Utslipp.ipynb](processed_data/Fremstilling_Utslipp.ipynb) og
   [Fremstilling_weather.ipynb](processed_data/Fremstilling_weather.ipynb) kjøres

## Kilder

yr - https://frost.met.no/observations/v0.jsonld

ssb - https://data.ssb.no/api/v0/dataset/832678.csv?lang=no

