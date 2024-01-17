import json
from collections import Counter
import matplotlib.pyplot as plt

json_bestand_pad = "/data/volume_2/dev/data/MIMIC/medication/patients_full.json"

def haal_medicatie_gegevens(json_bestand_pad):
    try:
        with open(json_bestand_pad, 'r') as json_bestand:
            inhoud = json_bestand.read()
            objecten = inhoud.strip().split('\n')
            
            gegevens = []
            for obj in objecten:
                try:
                    gegevens.append(json.loads(obj))
                except json.decoder.JSONDecodeError as e:
                    print(f"Skipping invalid JSON object: {e}")
            
            return gegevens
    except FileNotFoundError:
        print(f"The file {json_bestand_pad} was not found.")
        return None

def analyseer_medicatiegegevens(gegevens, min_telling=5000):
    if gegevens is None:
        return Counter()

    # Filter medicatie met 0.0 en minder dan 'min_telling' keer voorkomen uit de lijst
    medicatie_lijst = [medicatie for patient in gegevens for medicatie in patient.get('ndc_list', []) if medicatie != 0.0]
    medicatie_telling = Counter(medicatie_lijst)

    # Filter medicatie met minder dan 'min_telling' keer voorkomen
    medicatie_telling = {medicatie: telling for medicatie, telling in medicatie_telling.items() if telling >= min_telling}

    return medicatie_telling

def toon_top_medicatie(medicatie_telling, top_n=15):
    # Toon de top medicatie
    top_medicatie = Counter(medicatie_telling).most_common(top_n)
    print(f"Top {top_n} meest gebruikte medicatie:")
    for medicatie, telling in top_medicatie:
        print(f"Medicatie: {medicatie}, Aantal keer: {telling}")    

def plot_medicatiegrafiek(medicatie_telling):
    # Sorteer de medicatiecodes op basis van het aantal keer dat ze voorkomen (in aflopende volgorde)
    gesorteerde_medicatie = sorted(medicatie_telling.items(), key=lambda x: x[1], reverse=True)

    # Haal de medicatiecodes en de bijbehorende tellingen op
    medicatiecodes, tellingen = zip(*gesorteerde_medicatie)

    # Bepaal de maximale telling voor de y-as
    max_telling = max(tellingen)

    # Maak een staafdiagram
    plt.figure(figsize=(30, 15))  # Grootte van de grafiek aanpassen
    plt.bar(range(len(medicatiecodes)), tellingen, align='center')
    plt.xticks(range(len(medicatiecodes)), medicatiecodes, rotation='vertical', fontsize=8)  # Aanpassingen voor duidelijkheid
    plt.xlabel('Medication codes')
    plt.ylabel('Amount of times used')
    plt.title('Most used medication codes')
    plt.ylim(0, 66500) 

    # Aangepaste y-ticks instellen
    plt.yticks(range(0, 66500, 1000), fontsize=8)

    afbeeldingsbestand = '/data/volume_2/dev/myrthe/aae-recommender/myrthe-venv/medicatiegrafiek.png'
    plt.savefig(afbeeldingsbestand)

    print(f"Grafiek opgeslagen als: {afbeeldingsbestand}")

    # Toon het diagram
    plt.show()


    
def main():
    print("Loading data from", json_bestand_pad)
    medicatie_gegevens = haal_medicatie_gegevens(json_bestand_pad)
    medicatie_telling = analyseer_medicatiegegevens(medicatie_gegevens, min_telling=5000)
    
    # Toon de resultaten
    for medicatie, telling in medicatie_telling.items():
        print(f"Medicatie: {medicatie}, Aantal keer: {telling}")

    unieke_codes = len(medicatie_telling)
    print(f"Aantal unieke medicatiecodes: {unieke_codes}") #4084

    toon_top_medicatie(medicatie_telling)

    plot_medicatiegrafiek(medicatie_telling)

if __name__ == "__main__":
    main()
