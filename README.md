# Segmentacja Rynku Nieruchomości w Polsce (Algorytm K-means)

## Cel
Celem projektu jest weryfikacja dostępności mieszkań w Polsce poprzez zderzenie danych o transakcjach z realnymi zarobkami mieszkańców. Projekt dzieli się na dwie skale analizy:
- Analiza miast: Segmentacja 14 największych metropolii przy użyciu biblioteki scikit-learn.
- Analiza powiatowa: Segmentacja wszystkich powiatów w Polsce przy użyciu autorskiej implementacji algorytmu K-means.

## Środowisko i Wymagania

Projekt napisany jest w języku Python. Do poprawnego działania wymagane jest zainstalowanie bibliotek do analizy danych i mapowania.

1. Sklonuj repozytorium i przejdź do głównego katalogu.
    ```bash
    git clone https://github.com/andzelax/property-vs-income.git    
    ```
2. Utwórz i aktywuj środowisko wirtualne:
    ```bash
   python -m venv venv
   source venv/bin/activate
   
3. Zainstaluj zależności:
    ```
    pip install -r requirements.txt
    ```