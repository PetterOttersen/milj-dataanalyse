import pandas as pd

#Vi behandler dataframen sånn at vi kun får de dataene vi synes er relevante

df = pd.read_csv("raw_data/data/csv_df.csv")

df.to_excel("Excelfil_ubeh.xlsx")

df_trimmed = df[["index", "elementId", "value", "timeOffset", "qualityCode", "sourceId"]]

cutoff_quality = 3

quality_dict = df_trimmed["qualityCode"].to_dict()
non_valid_qualities = []

for key in quality_dict:
    try:
        int(quality_dict[key])

    except ValueError:  #Sorterer ut Nan-verdier
        non_valid_qualities.append(key)
        continue

    if (int(quality_dict[key]) > cutoff_quality):  #Sorterer ut verdier dersom de overstiger kvalitets-cutoffen.
        non_valid_qualities.append(key)

df_trimmed = df_trimmed.drop(non_valid_qualities)
        
df_trimmed.to_excel("Excelfil.xlsx")


