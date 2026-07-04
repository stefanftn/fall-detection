# Predlog projekta

**Računarska inteligencija**

**Stefan Ilić SV12/2023**

## Naziv teme

Multimodelni sistem za detekciju pada u realnom vremenu primenom mobilnih senzorskih uređaja

## Definicija problema

- Problem koji ovaj projekat rešava jeste automatska, pravovremena i pouzdana detekcija pada kod rizičnih grupa stanovništva, prvenstveno starijih osoba, pacijenata na postoperativnom oporavku i osoba sa neurološkim poremećajima.
- Najveći rizik kod padova starijih osoba nije nužno sam udarac, već vreme koje osoba provede povređena na podu bez pomoći. Ako pomoć ne stigne u roku od jednog sata, rizik od ozbiljnih komplikacija ili smrtnog ishoda raste za 50%.
- Sa globalnim starenjem populacije, broj hospitalizacija usled padova raste. Automatizovani nadzor smanjuje pritisak na medicinsko osoblje u bolnicama i domovima za stare.

## Skup podataka

U projektu će se koristiti **SisFall** dataset, jedan od najpriznatijih i najobimnijih javno dostupnih skupova podataka za ovu oblast.

Link do skupa podataka:
- [Kaggle - SisFall Enhanced Dataset](https://www.kaggle.com/datasets/nvnikhil0001/sisfall-enhanced)
- [Originalni istraživački izvor](https://www.kaggle.com/datasets/nvnikhil0001/sis-fall-original-dataset)

**Broj instanci:** Dataset sadrži preko 3.000 pojedinačnih fajlova (zapisa kretanja). Testirano je 38 subjekata (23 mlade osobe koje su simulirale padove i 15 starijih osoba koje su izvodile svakodnevne aktivnosti).

**Atributi (Sirovi podaci):** Podaci su zabeleženi sa tri senzora (dva akcelerometra i jedan žiroskop) sa frekvencijom uzorkovanja od 200Hz (200 očitavanja u sekundi).

Kroz proces pretprocesiranja biće izvučeni sledeći ključni atributi:

- *Vektorska magnituda ubrzanja:* $VM = \sqrt{X^2 + Y^2 + Z^2}$ za eliminaciju problema orijentacije telefona.
- *Maksimalni udarac:* Najviša vrednost VM u vremenskom prozoru.
- *Standardna devijacija i varijansa:* za nivo haotičnosti u pokretu.

**Ciljno obeležje:** Atribut Activity_Type (binarna klasa)

- **0 (ADL - Activities of Daily Living):** Normalne aktivnosti (hodanje, trčanje, brzo sedanje, ležanje, penjanje uz stepenice).
- **1 (FALL):** Stvarni padovi (pad napred, pad unazad, bočni pad, pad usled sinkope/gubitka svesti).

## Metodologija

### A. Pretprocesiranje podataka i „Sliding Window" tehnika

Sirovi podaci iz vremenskih serija biće isečeni na vremenske prozore dužine 3 sekunde (sa preklapanjem od 50% kako se ne bi izgubila granica samog pada). Padovi traju kratko (obično oko 0.5 do 1 sekunde), pa je prozor od 3 sekunde idealan da uhvati stanje *pre pada*, *trenutak udarca* i *stanje mirovanja nakon pada*. Za svaki prozor računaće se gore navedene statističke karakteristike i formirati konačna tabela za ML modele.

### B. Implementacija i poređenje modela

U projektu će biti implementirana i upoređena tri različita algoritma kako bi se analizirale njihove performanse i ponašanje:

- **Gaussian Naive Bayes:** Predstavnik verovatnosnih modela sa predavanja. Služi kao bazični model koji brzo računa uslovne verovatnoće za pad na osnovu statistika prozora.
- **kNN:** Predstavnik metričkih modela. Klasifikovaće novi prozor na osnovu Euklidske udaljenosti u odnosu na najbliže profile kretanja iz trening seta. Biće izvršena optimizacija hiperparametra k.
- **Neuronska mreža (MLP):** Predstavnik dubokog učenja. Arhitektura će se sastojati od ulaznog sloja, dva skrivena sloja sa ReLU aktivacionim funkcijama i jednim izlaznim neuronom sa Sigmoid funkcijom za binarnu klasifikaciju.

### C. Demonstracija i integracija sa Android mobilnim uređajem

Za potrebe demonstracije sistema u realnom vremenu, biće kreirana klijent-server arhitektura sa nativnom ili hibridnom Android aplikacijom.

## Način evaluacije

Evaluacija modela se neće vršiti isključivo na osnovu ukupne tačnosti (*Accuracy*), jer je dataset debalansiran (ima više primera hodanja nego padova). Koristiće se sledeće metrike:

- **Matrica konfuzije (Confusion Matrix):** Prikaz tačnih i pogrešnih klasifikacija za sve tri metode.
- **Osetljivost (Sensitivity / Recall):** Ključna metrika za ovaj projekat. Meri procenat stvarnih padova koje je model uspešno detektovao. U medicinskim aplikacijama cilj je da Recall bude što bliži 100% (bolje je imati lažnu uzbunu, nego promašiti pad pacijenta).
- **F1-Score:** Harmonijska sredina preciznosti i odziva, koja daje realnu ocenu stabilnosti modela.
- **Inference Time (Vreme izvršavanja):** Izmeriće se vreme u milisekundama koje je potrebno svakom modelu da donese odluku, što je ključno za aplikacije koje rade u realnom vremenu.

## Tehnologije

- **Biblioteke za obradu podataka i ML:** Pandas, NumPy, Scipy.
- **Algoritmi računarske inteligencije:** Scikit-learn (za Naive Bayes i kNN), TensorFlow / Keras (za izradu i treniranje MLP neuronske mreže).
- **Vizuelizacija i Evaluacija:** Matplotlib, Seaborn.
- **Deployment i API:** FastAPI
- **Mobilni razvoj:** Android SDK, Android Sensor.TYPE_ACCELEROMETER i Sensor.TYPE_GYROSCOPE.

## Primeri

- **Apple Watch Fall Detection:** Komercijalno rešenje integrisano u pametne satove koje koristi slične algoritme za prepoznavanje naglog ubrzanja i slanje SOS poruka.
- **Fall Detection sa pametnih uređaja:** Primer istraživačkog koda koji implementira mašinsko učenje nad akcelerometarskim podacima. [Link do GitHub repozitorijuma](https://github.com/tudorbaranai/guardian-fall-companion)

## Relevantna literatura

- Sakorn Mekruksavanich, Anuchit Jitpattanaku: [Enhancing Wearable-Based Elderly Activity Recognition Through a Hybrid Deep Residual Network](https://www.mdpi.com/2504-4990/8/4/107)
- Yue Shi, Yuanchun Shi, Xia Wang: [Fall Detection on Mobile Phones Using Features from a Five-Phase Model](https://ieeexplore.ieee.org/document/6332111)