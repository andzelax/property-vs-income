import pandas as pd
import numpy as np
import folium
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# K-MEANS (Czysty Python)
# ==============================================================================
def dist(p, q):
    return sum((pp - qq) ** 2 for pp, qq in zip(p, q))

def center(points):
    if len(points) == 0:
        return [0.0, 0.0]
    return [sum(coords) / len(points) for coords in zip(*points)]

def assign(points, centroids):
    assignments = []
    for p in points:
        dists = [dist(p, c) for c in centroids]
        assignments.append(dists.index(min(dists)))
    return assignments

def kmeans_powiatowy(points, k, max_iter=100, eps=1e-4):
    import random
    centroids = random.sample(points, k)
    for _ in range(max_iter):
        assignments = assign(points, centroids)
        clusters = [[] for _ in range(k)]
        for i, p in enumerate(points):
            clusters[assignments[i]].append(p)
        
        new_centroids = [center(cluster) for cluster in clusters]
        if max(dist(c, nc) for c, nc in zip(centroids, new_centroids)) < eps:
            break
        centroids = new_centroids
    return clusters, centroids, assignments

def czysc_nazwe(tekst):
    if not isinstance(tekst, str):
        return ""
    t = tekst.lower()
    for skrot in ['powiat ', 'm. st. ', 'm. ']:
        t = t.replace(skrot, '')
    return t.strip()


# ==============================================================================
# 1. PRZETWARZANIE DANYCH GUS (Transakcje, Metraże, Wynagrodzenia)
# ==============================================================================
print("Integracja baz danych GUS dla powiatów...")
df_lokale = pd.read_csv('RYNE_3783_CTAB_20260606230330.csv', delimiter=';')
df_powierzchnia = pd.read_csv('RYNE_3785_CTAB_20260606230435.csv', delimiter=';')
df_wynagrodzenia = pd.read_csv('WYNA_2497_CTAB_20260531120135.csv', delimiter=';')

df_lokale = df_lokale[(df_lokale['Kod'] != 0) & (~df_lokale['Nazwa'].str.isupper())].copy()
df_powierzchnia = df_powierzchnia[(df_powierzchnia['Kod'] != 0) & (~df_powierzchnia['Nazwa'].str.isupper())].copy()
df_wynagrodzenia = df_wynagrodzenia[(df_wynagrodzenia['Kod'] != 0) & (~df_wynagrodzenia['Nazwa'].str.isupper())].copy()

rok_szt = 'ogółem;ogółem;2024;[szt.]'
rok_m2 = 'ogółem;ogółem;2024;[m2]'
rok_wyn = 'ogółem;2024;[zł]'

df_l = df_lokale[['Kod', 'Nazwa', rok_szt]].copy()
df_l['liczba_transakcji'] = df_l[rok_szt].astype(float)

df_p = df_powierzchnia[['Kod', rok_m2]].copy()
df_p['laczna_powierzchnia'] = (df_p[rok_m2].astype(str)
                               .str.replace(r'\s+', '', regex=True)
                               .str.replace(',', '.', regex=False)
                               .astype(float))

df_w = df_wynagrodzenia[['Kod', rok_wyn]].copy()
df_w['wynagrodzenie_netto'] = (df_w[rok_wyn].astype(str)
                               .str.replace(r'\s+', '', regex=True)
                               .str.replace(',', '.', regex=False)
                               .astype(float) * 0.72).round(2)

df_powiaty = pd.merge(df_l[['Kod', 'Nazwa', 'liczba_transakcji']], df_p[['Kod', 'laczna_powierzchnia']], on='Kod')
df_powiaty = pd.merge(df_powiaty, df_w[['Kod', 'wynagrodzenie_netto']], on='Kod')

df_powiaty = df_powiaty[df_powiaty['liczba_transakcji'] >= 10].copy()
df_powiaty['sredni_metraz'] = df_powiaty['laczna_powierzchnia'] / df_powiaty['liczba_transakcji']

# NOWY WSKAŹNIK: Stosunek rocznego dochodu netto mieszkańca do średniego metrażu kupowanego w powiecie
df_powiaty['affordability_index'] = ((df_powiaty['wynagrodzenie_netto'] * 12) / df_powiaty['sredni_metraz']).round(2)

df_powiaty = df_powiaty.dropna().reset_index(drop=True)


# ==========================================
# 2. MODELOWANIE K-MEANS Z LOGARYTMEM
# ==========================================
print("Uruchamianie autorskiego algorytmu K-means...")
df_powiaty['transakcje_log'] = np.log1p(df_powiaty['liczba_transakcji'])

X_raw = df_powiaty[['transakcje_log', 'sredni_metraz']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

import random
random.seed(42)
klastry_scaled, centroidy_scaled, assignments = kmeans_powiatowy(X_scaled.tolist(), k=3)
df_powiaty['cluster'] = assignments

srednie_transakcje = df_powiaty.groupby('cluster')['liczba_transakcji'].mean().sort_values()

mapowanie_nazw = {
    srednie_transakcje.index[0]: "Rynki Kameralne (Niska płynność)",
    srednie_transakcje.index[1]: "Rynki Rozwijające się",
    srednie_transakcje.index[2]: "Metropolitalne Centra Obrotu"
}
df_powiaty['cluster_mapped'] = df_powiaty['cluster'].map(mapowanie_nazw)
df_powiaty['Nazwa_czysta'] = df_powiaty['Nazwa'].apply(czysc_nazwe)

df_powiaty_nodup = df_powiaty.drop_duplicates(subset=['Nazwa_czysta'])
dane_tekstowe_dict = df_powiaty_nodup.set_index('Nazwa_czysta').to_dict(orient='index')

# ==============================================================================
# 3. GENEROWANIE WYKRESU
# ==============================================================================
print("Generowanie czystego wykresu rozrzutu...")
centroidy_realne = scaler.inverse_transform(centroidy_scaled)

plt.figure(figsize=(11, 7))

paleta_kolorow = {
    "Metropolitalne Centra Obrotu": "#e41a1c",           # Czerwony
    "Rynki Rozwijające się": "#377eb8",                  # Niebieski
    "Rynki Kameralne (Niska płynność)": "#4daf4a"         # Zielony
}

# Rysujemy powiaty (używamy skali logarytmicznej na osi X, aby wykres był czytelny)
sns.scatterplot(
    data=df_powiaty,
    x='liczba_transakcji',
    y='sredni_metraz',
    hue='cluster_mapped',
    palette=paleta_kolorow,
    s=90,
    edgecolor='black',
    alpha=0.7,
    zorder=3
)
plt.xscale('log') # Logarytmiczna skala osi płynności rynkowej

# Nanosimy centroidy
# Ponieważ model uczył się na logarytmie transakcji, pierwszą współrzędną centroidu musimy odlogarytmować (np.expm1)
centroids_log_x = centroidy_realne[:, 0]
centroids_real_x = np.expm1(centroids_log_x)
centroids_real_y = centroidy_realne[:, 1]

plt.scatter(
    centroids_real_x,
    centroids_real_y,
    marker='*',
    s=500,
    color='black',
    label='Autorskie Centroidy (Środki klastrów)',
    edgecolor='white',
    linewidth=2,
    zorder=4
)

top_powiaty = df_powiaty.sort_values('liczba_transakcji', ascending=False).head(6)
for _, row in top_powiaty.iterrows():
    plt.text(
        x=row['liczba_transakcji'] * 1.15, # Przesunięcie w skali logarytmicznej
        y=row['sredni_metraz'],
        s=row['Nazwa'].replace('Powiat m. st. ', '').replace('Powiat m. ', '').replace('Powiat ', ''),
        fontsize=9, weight='bold', zorder=5
    )

plt.title('Segmentacja wszystkich powiatów w Polsce za pomocą WŁASNEGO algorytmu K-means (2024)', fontsize=13, weight='bold', pad=15)
plt.xlabel('Roczna liczba sprzedanych mieszkań (Płynność rynku - skala logarytmiczna)', fontsize=11)
plt.ylabel('Średni metraż zakupionego lokalu (m²)', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5, zorder=1)
plt.legend(title="Profil strukturalny powiatu (GUS)", loc='upper left', fontsize=10)
plt.tight_layout()

plt.savefig('wlasny_kmeans_powiaty_scatter.png', bbox_inches='tight', dpi=150)
plt.close()
print("\n-> Sukces! Wygenerowano bezbłędny wykres powiatowy: wlasny_kmeans_powiaty_scatter.png")

# ==========================================
# 3. WCZYTYWANIE GEOMETRII I INTEGRACJA W TOOLTIPIE
# ==========================================
print("Generowanie interaktywnej mapy Folium...")

with open('powiaty-medium.geojson', 'r', encoding='utf-8') as f:
    geojson_powiaty = json.load(f)

for feature in geojson_powiaty['features']:
    nazwa_mapowa = feature['properties'].get('nazwa', '')
    nazwa_czysta = czysc_nazwe(nazwa_mapowa)
    
    info = dane_tekstowe_dict.get(nazwa_czysta, {})
    if not info:
        for klucz, wartosc in dane_tekstowe_dict.items():
            if nazwa_czysta in klucz or klucz in nazwa_czysta:
                info = wartosc
                break
                
    if info:
        feature['properties']['wyswietl_profil'] = str(info['cluster_mapped'])
        feature['properties']['wyswietl_transakcje'] = f"{int(info['liczba_transakcji'])} szt."
        feature['properties']['wyswietl_metraz'] = f"{info['sredni_metraz']:.2f} m²"
        feature['properties']['wyswietl_zarobki'] = f"{info['wynagrodzenie_netto']:.2f} zł netto"
        feature['properties']['wyswietl_indeks'] = f"{info['affordability_index']:.1f} pkt (Dochód roczny / m² produktu)"
    else:
        feature['properties']['wyswietl_profil'] = "Odrzucony (Płynność < 10 szt.)"
        feature['properties']['wyswietl_transakcje'] = "Brak danych"
        feature['properties']['wyswietl_metraz'] = "Nie dotyczy"
        feature['properties']['wyswietl_zarobki'] = "Brak danych"
        feature['properties']['wyswietl_indeks'] = "Nie dotyczy"

mapa = folium.Map(location=[52.0, 19.0], zoom_start=6, tiles="cartodbpositron")

kolory_klastrow = {
    "Metropolitalne Centra Obrotu": "#e41a1c",
    "Rynki Rozwijające się": "#377eb8",
    "Rynki Kameralne (Niska płynność)": "#4daf4a"
}

def stylizuj_powiat(feature):
    nazwa_mapowa = feature['properties'].get('nazwa', '')
    nazwa_czysta = czysc_nazwe(nazwa_mapowa)
    
    info = dane_tekstowe_dict.get(nazwa_czysta, {})
    if not info:
        for klucz, wartosc in dane_tekstowe_dict.items():
            if nazwa_czysta in klucz or klucz in nazwa_czysta:
                info = wartosc
                break
                
    cluster_name = info.get('cluster_mapped', '')
    kolor = kolory_klastrow.get(cluster_name, "#d9d9d9")
    
    opacity = 0.7 if cluster_name else 0.3
    weight = 1.2 if cluster_name else 0.6
    
    return {
        'fillColor': kolor,
        'color': '#2c3e50',
        'weight': weight,
        'fillOpacity': opacity
    }

folium.GeoJson(
    geojson_powiaty,
    style_function=stylizuj_powiat,
    tooltip=folium.GeoJsonTooltip(
        fields=['nazwa', 'wyswietl_profil', 'wyswietl_transakcje', 'wyswietl_metraz', 'wyswietl_zarobki', 'wyswietl_indeks'],
        aliases=['Powiat:', 'Profil K-means:', 'Roczny obrót lokali:', 'Średni metraż lokalu:', 'Miesięczna pensja netto:', 'Wskaźnik Dostępności Dochodowej:'],
        localize=True,
        style="font-family: Arial, sans-serif; font-size: 13px; padding: 10px; background-color: white; border-radius: 4px;"
    )
).add_to(mapa)

legenda_html = """
     <div style="position: fixed; 
     bottom: 40px; left: 40px; width: 270px; height: 120px; 
     border:2px solid #555; z-index:9999; font-size:12px;
     background-color:white; opacity: 0.95; padding: 12px;
     font-family: Arial, sans-serif; border-radius: 8px;
     box-shadow: 3px 3px 5px rgba(0,0,0,0.2);">
     <b>Segmentacja Twojego K-means (2024):</b><br style="margin-bottom:5px;">
     <i class="fa fa-square" style="color:#e41a1c"></i> Metropolitalne Centra Obrotu<br>
     <i class="fa fa-square" style="color:#377eb8"></i> Rynki Rozwijające się<br>
     <i class="fa fa-square" style="color:#4daf4a"></i> Rynki Kameralne (Niska płynność)<br>
     </div>
     """
mapa.get_root().html.add_child(folium.Element(legenda_html))

mapa.save("mapa_wlasny_kmeans_powiaty.html")
print("\n-> Sukces! Wygenerowano ostateczną wersję mapy: mapa_wlasny_kmeans_powiaty.html")