import matplotlib.pyplot as plt
import pandas as pd

file_path = "../Excelfil.xlsx"

# Leser data fra Excel
data = pd.read_excel(file_path)

# Antar at 'value' er kolonnen du ønsker å bruke
nedbør = data["value"].tolist()

# Lage en liste med dager basert på lengden på dataen
dager = list(range(len(nedbør)))

# Plotter diagrammet
plt.bar(dager, nedbør)
plt.show()











