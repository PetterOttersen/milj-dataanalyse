from getData import df

#Vi behandler dataframen sånn at vi kun får de dataene vi synes er relevante

df_trimmed = df[["index", "elementId", "value", "timeOffset", "qualityCode", "sourceId"]]

df_trimmed.to_excel("Excelfil.xlsx")





