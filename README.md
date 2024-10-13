# VetA Voorspellingen
In deze GitHub repository vind je code en een beschrijving om zelf met behulp van AI de uitslagen van VetA-wedstrijden te voorspellen. Er zijn twee varianten beschikbaar: een Poisson-model en een uitgebreider deep learning model. Deze implementaties zijn bedoeld als voorbeeld voor werkende AI modellen.

# Kan ik ook ...?
De modellen zijn universeel toepasbaar op een KNHB competitie. Ze worden gevoed met de uitslagen van alle tot nu toe in een competitie gespeelde wedstrijden. 
Op basis daarvaan wordt uitgewerkt welke wedstrijden nog gespeeld moeten worden en daarvoor worden dan voorspellingen gedaan.  
Je kunt de modellen dus eenvoudig toepassen op de competitie van een ander team door ze te voeden met data van die andere competitie. 
Uitslagen van het lopende seizoen zijn voor iedere competitie op de KNHB site te vinden.

# Zijn ze goed? Nee.
Omdat de inputdatasets te beperkt van omvang zijn, zijn de resultaten discutabel. Het deep learning model is zelfs behoorlijk overkill voor deze toepassing. Op dit moment worden voor deze toepassing en deze datasets de beste resultaten behaald met het Poisson-model. 
Tot nu is het resultaat dat er per wedstrijd voor 1 partij maximaal 2 doelpunten te weinig voorspeld worden, over het algemeen zonder het winst of verlies resultaat te beïnvloeden. Voor serieuze toepassingen is dat natuurlijk gruwelijk slecht.
Ik heb aangegeven hoe je zelf met tuning parameters de modellen nog verder kunt tunen.
 
# Hmmm. Okay. Hoe...?
Per model vind je een uitleg en een uitgebreide beschrijving, en natuurlijk de Python-code om de betreffende AI-modellen op je eigen computer te kunnen draaien. Experimenteer gerust met de code en datasets om de voorspellingen verder te verbeteren! Ik heb aangegeven per model hoe je dat kunt doen.
  
Het Poisson-model is een goede start, is ook de basis voor bookmakers e.d. maar wordt bijv. ook in verkeersmanagement en gezondheidszorg gebruikt om workloads te voorspellen. Het deep learning model is gemaakt met TensorFlow, dat heel populair is nu. De uitgangspunten van Poisson worden gebruikt maar nu wordt de data gegenereerd aan de hand van een getraind neuraal netwerk ipv. een statistisch model. 
 
[Uitleg over de implementatie van het toegepaste Poisson model](poisson.md)   
[Uitleg over de implementatie van het toegepaste Deep Learning model](deeplearning.md)

Om de modellen uit te voeren op je pc heb je Python en de aangegeven libraries nodig. 

Ik heb geen intenties deze code verder te onderhouden, het meeste gedoe is de installatie van de libraries maar Google en Chatgpt zijn je vrienden in deze.
