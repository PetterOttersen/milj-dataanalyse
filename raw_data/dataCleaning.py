from getData import df

#Vi behandler dataframen sånn at vi kun får de dataene vi synes er relevante

df.to_excel("Excelfil.xlsx")

df_trimmed = df[["index", "elementId", "value", "timeOffset", "qualityCode", "sourceId"]]

