# Opis Projektu: Property vs Income

## 1. Problem
Rynek nieruchomości w Polsce jest bardzo zróżnicowany. Problem z kupnem własnego mieszkania dotyka obecnie wiele młodych osób. Chcieliśmy sprawdzić, czy ceny mieszkań wynikają z realnych zarobków mieszkańców, czy może z innych mechanizmów.

## Datasety
WYNA_2497_CTAB_20260531120135.csv - Przeciętne miesięczne wynagrodzenia brutto 2024r. dla powiatów i województw, GUS

apartment-prices-in-poland - Zbiór danych zawiera oferty sprzedaży mieszkań z 15 największych miast Polski (Warszawa, Łódź, Kraków, Wrocław, Poznań, Gdańsk, Gdynia, Radom, Szczecin, Bydgoszcz, Lublin, Katowice, Białystok, Częstochowa, Rzeszów). Dane pochodzą z lokalnych stron internetowych z ofertami sprzedaży mieszkań. Aby lepiej uchwycić okolicę każdego mieszkania, każda oferta została rozszerzona o dane z Open Street Map z odległościami do punktów POI. Dane są gromadzone co miesiąc i obejmują okres od sierpnia 2023 r. do czerwca 2024 r.

RYNE_3783_CTAB_20260606230330.csv - Oficjalne statystyki GUS dotyczące liczby zawartych transakcji na rynku nieruchomości w 2024 roku dla każdego powiatu w Polsce.

RYNE_3785_CTAB_20260606230435.csv - Informacje o całkowitym metrażu lokali sprzedanych w poszczególnych powiatach w 2024 roku.

powiaty-medium.geojson - Granice wszystkich powiatów w Polsce.

## 2. Co zrobiliśmy?
Projekt składa się z dwóch części:
* **Analiza miast:** Sprawdziliśmy 14 największych miast w Polsce, porównując ceny mieszkań do przeciętnych pensji.
* **Analiza powiatowa:** Sprawdziliśmy wszystkie powiaty w Polsce, badając płynność rynku (liczbę transakcji) i średni metraż.

## 3. Jak to zrobiliśmy?
* Użyliśmy danych z GUS oraz datasetu z Kaggle z portali z ogłoszeniami.
* Stworzyliśmy własny **Wskaźnik Dostępności**, który pokazuje, ile pensji trzeba odłożyć, aby kupić 1 m² mieszkania.
* Do podziału rynków na grupy (np. drogie centra vs tańsze przedmieścia) użyliśmy algorytmu **K-means**. W części powiatowej napisaliśmy ten algorytm całkowicie od zera, aby w pełni zrozumieć, jak działa matematyka podziału danych.

## 4. Wnioski
* **Miasta:** W największych ośrodkach (Warszawa, Kraków, Wrocław) mieszkania są trudniej dostępne dla przeciętnego pracownika.
* **Powiaty:** Obserwujemy zjawisko ucieczki z drogich centrów na przedmieścia – ludzie szukają większej przestrzeni w powiatach wokół dużych miast, gdzie relacja ceny do metrażu jest bardziej korzystna.