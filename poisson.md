## Functionele Werking van het Script

Dit script is ontwikkeld om uitslagen van veldhockeywedstrijden te voorspellen op basis van historische wedstrijdgegevens. Het maakt gebruik van statistische methoden en AI-concepten, zoals de Poisson-verdeling, om de kans op verschillende wedstrijduitslagen te berekenen. De gebruiker kan een CSV-bestand met historische wedstrijduitslagen invoeren, waarna het script voorspellingen genereert voor de nog niet gespeelde wedstrijden.

## Implementatie van AI Concepten

Het script maakt gebruik van een eenvoudige vorm van AI, namelijk voorspellende modellering met behulp van statistiek. De **Poisson-verdeling** wordt gebruikt om het aantal gescoorde doelpunten te voorspellen, waarbij rekening wordt gehouden met de aanvalskracht en verdedigingssterkte van de teams. Hoewel het geen complexe machine learning gebruikt, wordt toch een vorm van voorspellende analyse toegepast om resultaten te genereren op basis van bestaande data.

## Uitgevoerde Analyses

Het script voert een aantal analyses uit om voorspellingen te kunnen doen:

1. **Berekening van Thuisvoordeel**: De verhouding tussen het aantal gescoorde doelpunten van thuisspelende teams en uitspelende teams wordt berekend.
2. **Teamstatistieken**: Voor elk team worden statistieken berekend, zoals het gemiddelde aantal gescoorde en tegendoelpunten per wedstrijd.
3. **Aanvalskracht en Verdedigingssterkte**: Op basis van de gemiddelde doelpunten wordt een maat berekend voor de aanvalskracht en verdedigingssterkte van elk team.
4. **Voorspelling van Verwachte Doelpunten**: De verwachte doelpunten voor beide teams worden voorspeld met behulp van de aanvalskracht, verdedigingssterkte, het thuisvoordeel en het gemiddelde aantal doelpunten in de competitie.

## Verwerking van Input naar Output

Het script neemt een CSV-bestand als invoer, waarin de historische wedstrijdgegevens staan, zoals de teams en het aantal gescoorde doelpunten. Op basis van deze gegevens worden statistieken berekend die vervolgens worden gebruikt om de uitslag van nog te spelen wedstrijden te voorspellen. De uiteindelijke voorspellingen, inclusief de verwachte doelpunten, de kans op winst/gelijkspel/verlies, en de meest waarschijnlijke uitslag, worden opgeslagen in een nieuwe CSV.

## Nadere Uitleg om de Output van het Script te Begrijpen

De output van het script bevat voorspellingen voor elke nog te spelen wedstrijd. Per wedstrijd worden de volgende gegevens weergegeven:

- **Verwachte Doelpunten**: Het aantal doelpunten dat elk team waarschijnlijk zal scoren. Dit is een gemiddelde waarde die aangeeft hoeveel doelpunten een team waarschijnlijk zal maken op basis van de berekende aanvalskracht, verdedigingssterkte en andere factoren.
- **Meest Waarschijnlijke Uitslag**: De specifieke uitslag (zoals 2-1 of 1-1) die de hoogste waarschijnlijkheid heeft volgens de Poisson-verdeling. Dit is de combinatie van doelpuntenaantallen voor beide teams die het meest waarschijnlijk is.
- **Waarschijnlijkheid van de Uitslag**: Hoe groot de kans is dat deze specifieke uitslag optreedt.
- **Voorspelling**: Welke ploeg waarschijnlijk zal winnen of dat het een gelijkspel zal worden.
- **Winstkansen**: De kansen (in percentages) voor een overwinning van het thuisteam, een gelijkspel, of een overwinning van het uitteam.

## Uitleg Waarom Resultaten Kunnen Afwijken

Voorspellingen zijn gebaseerd op statistische gemiddelden en historische gegevens, waardoor afwijkingen mogelijk zijn. Factoren zoals blessures, weersomstandigheden, vorm van de dag, of wijzigingen in tactiek worden niet meegenomen in het model. Bovendien zijn veldhockeywedstrijden soms moeilijk te voorspellen vanwege hun dynamische karakter en de mogelijkheid dat kleine gebeurtenissen grote invloed kunnen hebben op de uitkomst.

## Gebruik van het Script

Het script kan worden gebruikt door analisten, coaches, of liefhebbers die inzicht willen krijgen in de verwachte uitslagen van veldhockeywedstrijden. Door historische gegevens in te voeren, kan men voorspellingen krijgen voor toekomstige wedstrijden. Dit kan bijvoorbeeld nuttig zijn voor het maken van strategieën of voor het verkennen van verschillende scenario's.

## Tunen van het Model

Het model kan verder getuned worden om de nauwkeurigheid van de voorspellingen te verbeteren:

- **Aanpassen van de Poisson-Parameters**: De parameters van de Poisson-verdeling, zoals de gemiddelde doelpunten per wedstrijd, kunnen worden aangepast om beter overeen te komen met de specifieke kenmerken van veldhockey. Deze parameters bevinden zich in de sectie waarin de statistieken worden berekend, zoals de aanvalskracht en verdedigingssterkte van de teams. In de code bevinden deze parameters zich bijvoorbeeld in de functie `prepare_poisson_data()`, waar de **gemiddelde doelpunten per team** en **league average** worden berekend.
- **Meer Gegevens Toevoegen**: Door meer gegevens over teams en wedstrijden toe te voegen, zoals recente vorm of blessure-informatie, kan de voorspellende waarde van het model worden verhoogd. Deze gegevens kunnen worden toegevoegd aan de dataset die als invoer wordt gebruikt. Dit betekent dat de CSV-invoer uitgebreid kan worden met aanvullende kolommen, die vervolgens gebruikt kunnen worden bij het berekenen van de teamstatistieken.
- **Gebruik van Machine Learning**: Een volgende stap zou kunnen zijn om meer geavanceerde AI-technieken, zoals machine learning, te gebruiken om patronen te leren die niet eenvoudig met statistiek te vangen zijn. Dit vereist het implementeren van een nieuw algoritme en het trainen van een model met behulp van de historische wedstrijdgegevens. De implementatie daarvan is te vinden in de [deep learning versie van het competitie voorspellingsmodel](deeplearning.md)
