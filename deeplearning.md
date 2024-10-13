## VetA competitieuitslagen te voorspellen met Deep Learning

**Disclaimer:** Dit script is ontwikkeld om wedstrijduitslagen te voorspellen, gewoon omdat het kan. Het is een voorbeeld van een implementatie van een deep learning concept. De trainingsdataset is over het algemeen te klein om zinvolle resultaten te verwachten. De resultaten zijn kwalitatief onvoldoende voor serieuze toepassingen en dus vooral grappig. Ga er dus met de nodige voorzichtigheid mee om.

## Inhoudsopgave

1. [Functionele Werking van het Script](#functionele-werking-van-het-script)
    - [Implementatie van het Deep Learning Concept](#implementatie-van-het-deep-learning-concept)
    - [Uitgevoerde Analyses](#uitgevoerde-analyses)
    - [Verwerking van Input naar Output](#verwerking-van-input-naar-output)
2. [Nadere Uitleg om de Output van het Script te Begrijpen](#nadere-uitleg-om-de-output-van-het-script-te-begrijpen)
3. [Uitleg Waarom Resultaten Kunnen Afwijken](#uitleg-waarom-resultaten-kunnen-afwijken)
4. [Gebruik van het Script](#gebruik-van-het-script)

---

## Functionele Werking van het Script

### Implementatie van het Deep Learning Concept

Het `deeplearning.py` script maakt gebruik van een deep neuraal netwerk (Deep Learning) om de verwachte uitkomsten van veldhockeywedstrijden te voorspellen. Het model is ontworpen om zowel het aantal doelpunten voor het thuisteam (`HomeGoals`) als het uitteam (`AwayGoals`) te voorspellen. Hieronder volgt een overzicht van hoe het deep learning concept is geïmplementeerd:

1. **Data Voorbereiding:**
    - **Lees historische wedstrijduitslagen:** Het script leest een CSV-bestand met kolommen zoals `HomeTeam`, `AwayTeam`, `HomeGoals`, en `AwayGoals`.
    - **Label Encoding:** De teamnamen worden omgezet naar numerieke waarden met behulp van `LabelEncoder`.
    - **One-Hot Encoding:** Beide teams (thuis en uit) worden één-hot gecodeerd om categorische gegevens geschikt te maken voor het model.
    - **Feature Scaling:** De gecombineerde features worden geschaald met `StandardScaler` om het model te helpen sneller en effectiever te trainen.

2. **Model Architectuur:**
    - **Input Layer:** Het model ontvangt geschaalde één-hot gecodeerde teamgegevens.
    - **Verborgen Lagen:** Twee verborgen lagen met respectievelijk 128 en 64 neuronen, elk gevolgd door een Dropout-laag om overfitting te voorkomen.
    - **Output Layers:** Twee afzonderlijke output lagen voorspellen respectievelijk het aantal doelpunten voor het thuisteam en het uitteam.

3. **Training:**
    - **Loss Functie:** Mean Squared Error (MSE) wordt gebruikt voor beide outputs.
    - **Optimizer:** Adam optimizer wordt gebruikt voor efficiënte training.
    - **Early Stopping:** Om overfitting te voorkomen, stopt het model trainen als de validatieverlies niet verbetert gedurende 10 opeenvolgende epochs.

### Uitgevoerde Analyses

Naast de primaire taak van het voorspellen van doelpunten, voert het script aanvullende analyses uit om de voorspellingen te verfijnen en inzicht te krijgen in de uitkomsten:

- **Thuisvoordeel Berekening:** Het script berekent het thuisvoordeel op basis van historische gegevens, wat helpt bij het corrigeren van de voorspellingen.
- **Poisson Distributie:** Gebruikt om de waarschijnlijkheid van verschillende scorelijnen te berekenen, wat bijdraagt aan de berekening van de meest waarschijnlijke uitslag en de bijbehorende waarschijnlijkheden.
- **Winstkansen Analyse:** Het script berekent de kans op winst voor het thuisteam, het uitteam en een gelijkspel, wat resulteert in een gedetailleerd inzicht in de verwachte uitkomst.

### Verwerking van Input naar Output

Het proces van het omzetten van invoer naar uitvoer verloopt als volgt:

1. **Invoer:**
    - Het script accepteert een invoerbestand (`uitslagen_h30.txt`) met historische wedstrijduitslagen.
    - De data wordt ingelezen en voorbereid door encoding en scaling.

2. **Model Predictie:**
    - Het getrainde model voorspelt het aantal doelpunten voor zowel het thuisteam als het uitteam.
    - Deze voorspellingen worden vervolgens gebruikt om de waarschijnlijkheden van verschillende uitkomsten te berekenen via de Poisson distributie.

3. **Output:**
    - De voorspellingen worden opgeslagen in een uitvoerbestand (`dl_voorspellingen_h30.csv`).
    - Elke voorspelling bevat gedetailleerde informatie zoals de verwachte doelpunten, de meest waarschijnlijke uitslag, de waarschijnlijkheid van die uitslag, de algemene voorspelling, en de winstkansen.

---

## Nadere Uitleg om de Output van het Script te Begrijpen

Het `dl_voorspellingen_h30.csv` bestand bevat meerdere kolommen die elk een specifiek aspect van de voorspellingen weergeven. Hieronder volgt een gedetailleerde uitleg van elke kolom:

| **Kolomnaam**                   | **Beschrijving**                                                                                                                                                  |
|---------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Wedstrijd**                   | De twee teams die aan een specifieke wedstrijd deelnemen, weergegeven als "ThuisTeam vs UitTeam".                                                                   |
| **Verwachte Doelpunten**        | De voorspelde gemiddelde aantal doelpunten voor het thuisteam en het uitteam, weergegeven als "ThuisDoelpunten - UitDoelpunten".                                      |
| **Meest Waarschijnlijke Uitslag**| De specifieke scorelijn die de hoogste waarschijnlijkheid heeft om zich voor te doen, berekend via Poisson distributie.                                           |
| **Waarschijnlijkheid Uitslag**   | De waarschijnlijkheid (in procenten) van de meest waarschijnlijke uitslag.                                                                                        |
| **Voorspelling**                 | De algehele voorspelling van het model voor de wedstrijd, zoals "TeamA wint", "TeamB wint" of "Gelijkspel".                                                        |
| **Waarschijnlijkheid Voorspelling**| De waarschijnlijkheid (in procenten) van de algehele voorspelling.                                                                                                  |
| **Winstkansen**                  | De geschatte winstkansen voor het thuisteam, een gelijkspel en het uitteam, weergegeven als "ThuisWin% - Gelijkspel% - UitWin%".                                      |

### Voorbeeld van een Output Rij

| Wedstrijd         | Verwachte Doelpunten | Meest Waarschijnlijke Uitslag | Waarschijnlijkheid Uitslag | Voorspelling | Waarschijnlijkheid Voorspelling | Winstkansen               |
|-------------------|----------------------|-------------------------------|----------------------------|--------------|-------------------------------|---------------------------|
| TeamA vs TeamB    | 2 - 1                | 2 - 1                         | 30.00%                     | TeamA wint   | 40.00%                        | 40.0% - 30.0% - 30.0%     |
| TeamC vs TeamD    | 0 - 3                | 0 - 3                         | 25.00%                     | TeamD wint   | 50.00%                        | 25.0% - 20.0% - 55.0%     |

### Kolommen Uitleg

#### 1. Wedstrijd

**Beschrijving:**
Deze kolom geeft de twee teams weer die aan een specifieke wedstrijd deelnemen. Het formaat is altijd "ThuisTeam vs UitTeam".

**Voorbeeld:**
- `TeamA vs TeamB`
- `TeamC vs TeamD`

#### 2. Verwachte Doelpunten

**Beschrijving:**
Hier worden de voorspelde aantal doelpunten voor zowel het thuisteam als het uitteam weergegeven. Het formaat is "ThuisDoelpunten - UitDoelpunten".

**Voorbeeld:**
- `2 - 1` (TeamA verwacht 2 doelpunten, TeamB verwacht 1 doelpunt)
- `0 - 3` (TeamC verwacht 0 doelpunten, TeamD verwacht 3 doelpunten)

#### 3. Meest Waarschijnlijke Uitslag

**Beschrijving:**
Dit is de meest waarschijnlijke uitkomst van de wedstrijd op basis van de voorspelde doelpunten. Het geeft de specifieke score weer met de hoogste waarschijnlijkheid.

**Voorbeeld:**
- `2 - 1` (Het meest waarschijnlijke resultaat is dat TeamA 2 doelpunten scoort en TeamB 1 doelpunt)
- `0 - 3` (Het meest waarschijnlijke resultaat is dat TeamC 0 doelpunten scoort en TeamD 3 doelpunten)

#### 4. Waarschijnlijkheid Uitslag

**Beschrijving:**
Deze kolom toont de waarschijnlijkheid (in procenten) van de meest waarschijnlijke uitslag. Het geeft aan hoe zeker het model is over de specifieke score die hierboven is vermeld.

**Voorbeeld:**
- `30.00%` (Er is een 30% kans dat het meest waarschijnlijke resultaat zich voordoet)
- `25.00%` (Er is een 25% kans dat het meest waarschijnlijke resultaat zich voordoet)

#### 5. Voorspelling

**Beschrijving:**
Dit is de algehele voorspelling van het model voor de wedstrijd, gebaseerd op de berekende winstkansen. Het geeft aan welk team verwacht wordt te winnen of of er een gelijkspel zal zijn.

**Mogelijke Waarden:**
- `TeamA wint`
- `TeamB wint`
- `Gelijkspel`

**Voorbeelden:**
- `TeamA wint` (Het model voorspelt dat TeamA de wedstrijd zal winnen)
- `Gelijkspel` (Het model voorspelt dat de wedstrijd gelijk zal eindigen)

#### 6. Waarschijnlijkheid Voorspelling

**Beschrijving:**
Deze kolom geeft de waarschijnlijkheid (in procenten) van de voorspelling zoals vermeld in de vorige kolom. Het toont hoe zeker het model is over de uitkomst van de wedstrijd.

**Voorbeelden:**
- `40.00%` (Er is een 40% kans dat TeamA wint)
- `50.00%` (Er is een 50% kans dat TeamD wint)
- `30.00%` (Er is een 30% kans op een gelijkspel)

#### 7. Winstkansen

**Beschrijving:**
Deze kolom toont de geschatte winstkansen voor het thuisteam, het uitspel, en een gelijkspel, in die volgorde. Het formaat is "ThuisWin% - Gelijkspel% - UitWin%".

**Voorbeelden:**
- `40.0% - 30.0% - 30.0%` (40% kans op winst voor TeamA, 30% kans op gelijkspel, en 30% kans op winst voor TeamB)
- `25.0% - 20.0% - 55.0%` (25% kans op winst voor TeamC, 20% kans op gelijkspel, en 55% kans op winst voor TeamD)

---

## Uitleg Waarom Resultaten Kunnen Afwijken

De resultaten van het `deeplearning.py` script kunnen afwijken van de werkelijke uitkomsten om verschillende redenen. Hieronder worden de belangrijkste factoren besproken die de nauwkeurigheid van de voorspellingen beïnvloeden.

### 1. Kleine Trainingsdataset

**Beschrijving:**
- Een beperkte hoeveelheid trainingsdata kan leiden tot overfitting of underfitting van het model.
- Met te weinig data kan het model niet voldoende patronen leren, waardoor de voorspellingen onnauwkeurig worden.

**Impact:**
- Minder nauwkeurige voorspellingen.
- Hoge variabiliteit in resultaten.

### 2. Complexiteit van het Model

**Beschrijving:**
- Een complex model met veel lagen en neuronen kan gevoelig zijn voor overfitting, vooral met beperkte data.
- Eenvoudige modellen kunnen mogelijk niet voldoende complexe patronen herkennen.

**Impact:**
- Overfitting: Het model presteert goed op de trainingsdata maar slecht op nieuwe data.
- Underfitting: Het model kan belangrijke patronen in de data niet leren.

### 3. Feature Engineering

**Beschrijving:**
- Onvoldoende of niet-relevante features kunnen de prestaties van het model beperken.
- Belangrijke factoren zoals teamvorm, blessures, of weersomstandigheden zijn mogelijk niet meegenomen.

**Impact:**
- Het model mist cruciale informatie die de uitkomsten beïnvloedt.
- Lagere nauwkeurigheid van voorspellingen.

### 4. Poisson Distributie Assumpties

**Beschrijving:**
- De Poisson-distributie gaat uit van de aanname dat doelpunten onafhankelijk en met een constant gemiddelde optreden.
- In werkelijkheid kunnen doelpunten afhangen van factoren zoals verdediging, aanvalsdynamiek, en speltempo.

**Impact:**
- Mogelijk onnauwkeurige berekeningen van waarschijnlijkheden van specifieke scorelijnen.
- De meest waarschijnlijke uitslag kan afwijken van de realiteit.

### 5. Dynamische Teamprestatie

**Beschrijving:**
- Teamprestaties kunnen variëren door seizoenen heen, afhankelijk van factoren zoals teamchemie, coaching, en spelersveranderingen.
- Het model gebruikt historische data die mogelijk niet representatief is voor de huidige teamvorm.

**Impact:**
- Verouderde data kan leiden tot onnauwkeurige voorspellingen.
- Het model kan moeite hebben om recente veranderingen in teamdynamiek te weerspiegelen.

### 6. Randomness in Sportwedstrijden

**Beschrijving:**
- Sportwedstrijden bevatten een element van onzekerheid en toevalligheid.
- Onverwachte gebeurtenissen zoals blessures, rode kaarten, of plotse doelpunten beïnvloeden de uitkomst.

**Impact:**
- Zelfs met een perfect model kunnen willekeurige gebeurtenissen de voorspellingen verstoren.
- Het model kan geen rekening houden met onverwachte gebeurtenissen tijdens een wedstrijd.

### 7. Data Kwaliteit

**Beschrijving:**
- Incomplete, onnauwkeurige of verouderde data kan de prestaties van het model negatief beïnvloeden.
- Fouten in de data kunnen leiden tot verkeerde voorspellingen.

**Impact:**
- Verkeerde voorspellingen door onjuiste data.
- Verminderde betrouwbaarheid van het model.

### 8. Parameter Instellingen van het Model

**Beschrijving:**
- Hyperparameters zoals het aantal lagen, neuronen per laag, leerpercentage, en batchgrootte kunnen de prestaties van het model sterk beïnvloeden.
- Onjuiste hyperparameter instellingen kunnen leiden tot suboptimale training.

**Impact:**
- Slechte convergentie tijdens training.
- Lagere nauwkeurigheid en betrouwbaarheid van voorspellingen.

### 9. Uitdagingen bij One-Hot Encoding

**Beschrijving:**
- One-hot encoding kan leiden tot hoge dimensionaliteit, vooral met veel teams, wat het model trager kan maken en mogelijk de prestaties kan beïnvloeden.

**Impact:**
- Hogere computationele kosten.
- Mogelijke verminderde prestaties bij beperkte data.

---

## Gebruik van het Script

Het `deeplearning.py` script is ontworpen om voorspellingen te genereren voor veldhockeywedstrijden op basis van historische data. Hieronder volgt een stapsgewijze handleiding voor het gebruik van het script.

### 1. Voorbereiding

**Benodigdheden**

- **Python 3.10** of hoger geïnstalleerd op je systeem.
- **Vereiste Python-pakketten:** `pandas`, `numpy`, `scipy`, `scikit-learn`, `tensorflow`.

### 2. Invoerbestand Voorbereiden

Zorg ervoor dat je een invoerbestand hebt met historische wedstrijduitslagen. Het bestand moet de volgende kolommen bevatten:

- `HomeTeam`: Naam van het thuisteam.
- `AwayTeam`: Naam van het uitteam.
- `HomeGoals`: Aantal doelpunten gescoord door het thuisteam.
- `AwayGoals`: Aantal doelpunten gescoord door het uitteam.

Het script zal op basis van de input ook bepalen welke wedstrijden er nog gespeeld moeten worden. 

**Voorbeeld van `uitslagen_h30.txt`:**

```csv
HomeTeam,AwayTeam,HomeGoals,AwayGoals
TeamA,TeamB,2,1
TeamC,TeamD,0,3
TeamA,TeamC,1,1
etc.
```
         
### 3. Uitvoeren van het script

Run deeplearning.bat nadat je de namen van de invoer en uitvoerbestanden aangepast hebt naar je wensen.

### 4. Analyseren van de Uitvoer
Na het uitvoeren van het script, genereert het een CSV-bestand met voorspellingen. Gebruik een spreadsheetprogramma om de resultaten te analyseren en inzicht te krijgen in de voorspellingen.
Uitleg over hoe de output te lezen hierboven.

Bijvoorbeeld:   
- Controleer de nauwkeurigheid: Vergelijk de voorspellingen met daadwerkelijke uitslagen om de prestaties van het model te evalueren.  
- Iteratieve Verbetering: Pas het model aan op basis van de resultaten en probeer het opnieuw met verbeterde data of modelparameters.  


