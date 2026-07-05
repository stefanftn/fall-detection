# TEHNIČKA SPECIFIKACIJA
## Multimodelni sistem za detekciju pada u realnom vremenu primenom mobilnih senzorskih uređaja

---

## 1. Definicija Problema i Cilj Modelovanja

### 1.1. Medicinsko-statistički kontekst

Problem koji ovaj projekat rešava jeste automatska, pravovremena i pouzdana detekcija pada kod rizičnih grupa stanovništva, prvenstveno starijih osoba, pacijenata na postoperativnom oporavku i osoba sa neurološkim poremećajima.

Najveći rizik kod padova starijih osoba nije nužno sam udarac, već vreme koje osoba provede povređena na podu bez pomoći. Ukoliko pomoć ne stigne u roku od jednog sata, rizik od ozbiljnih komplikacija ili smrtnog ishoda raste za 50%. Sa globalnim trendom starenja populacije, broj hospitalizacija usled padova konstantno raste, zbog čega automatizovani nadzor ima direktan uticaj na smanjenje pritiska na medicinsko osoblje u bolnicama i domovima za stare.

### 1.2. Formulacija problema u domenu mašinskog učenja

Cilj projekta je razvoj i poređenje modela računarske inteligencije koji problem detekcije pada tretiraju kao zadatak **binarne klasifikacije vremenskih serija u realnom vremenu**. Na osnovu kontinuiranog strimovanja podataka sa senzora, modeli moraju sa minimalnim kašnjenjem i visokim stepenom pouzdanosti da klasifikuju prozor ponašanja subjekta u jedno od dva stanja: normalna aktivnost ili pad.

---

## 2. Specifikacija Skupa Podataka (Data Specification)

Za potrebe razvoja i validacije modela koristi se **SisFall** dataset, jedan od najpriznatijih i najobimnijih javno dostupnih skupova podataka u ovoj oblasti.

### 2.1. Struktura i demografija skupa podataka

* **Ukupan broj instanci:** Dataset sadrži preko 3.000 pojedinačnih fajlova koji predstavljaju zapise kretanja.
* **Subjekti:** Eksperiment je sproveden nad ukupno 38 subjekata podeljenih u dve grupe:
    * **23 mlade osobe** koje su vršile simulaciju različitih tipova padova.
    * **15 starijih osoba** koje su izvodile isključivo bezbedne, svakodnevne životne aktivnosti.

### 2.2. Senzorski modaliteti i karakteristike sirovih podataka

Podaci su zabeleženi u visokoj rezoluciji pomoću tri hardverska senzora integrisana u uređaj:
* **Akcelerometar 1** (merenje linearnog ubrzanja).
* **Akcelerometar 2** (sekundarni senzor ubrzanja radi redundantnosti i preciznosti).
* **Žiroskop** (merenje ugaone brzine/rotacije).

Svi senzori rade na frekvenciji uzorkovanja od **200 Hz** (200 očitavanja u sekundi po svakoj osi X, Y, Z).

---

## 3. Pretprocesiranje i Inženjering Obeležja (Feature Engineering)

### 3.1. Segmentacija vremenskih serija (Sliding Window)

Sirovi podaci se ne mogu direktno proslediti tradicionalnim algoritmima mašinskog učenja, već se primenjuje tehnika **klizećeg prozora (Sliding Window)**:
* **Dužina prozora:** 3 sekunde (300 očitavanja po senzorskoj osi).
* **Preklapanje (Overlap):** 50% (1.5 sekundi) kako bi se osiguralo da kritične prelazne tačke na granici prozora ne budu izgubljene.

> **Opravdanost dinamike:** Padovi su tranzijentni događaji koji traju kratko (obično između 0.5 i 1 sekunde). Vremenski prozor od 3 sekunde je optimalan jer uspešno obuhvata kompletan fiziološki profil događaja: stanje *pre pada*, sam *trenutak udarca* i kritično *stanje mirovanja nakon pada*.

### 3.2. Ekstrakcija matematičkih obeležja (Features)

Iz svakog segmentisanog prozora od 3 sekunde ekstrahuju se ključni statistički i fizički indikatori koji transformišu sirove vremenske serije u konačnu tabelu za treniranje modela:

1. **Vektorska magnituda ubrzanja ($VM$):** Računa se sa ciljem eliminacije uticaja orijentacije samog uređaja u prostoru. Formula glasi:

   $$VM = \sqrt{X^2 + Y^2 + Z^2}$$

2. **Maksimalni udarac ($VM_{max}$):** Najviša registrovana vrednost vektorske magnitude unutar posmatranog vremenskog prozora, koja detektuje kinetički udar o podlogu.
3. **Standardna devijacija ($\sigma$) i Varijansa ($\sigma^2$):** Računaju se u svrhu kvantifikacije nivoa haotičnosti i dinamičnosti u pokretu subjekta.

### 3.3. Specifikacija ciljnog obeležja (Target Label)

Ciljno obeležje je predstavljeno atributom **Activity_Type** koji ima binarnu strukturu:

| Klasa | Naziv | Opis aktivnosti obuhvaćenih datasetom |
| :--- | :--- | :--- |
| **0** | **ADL** (Activities of Daily Living) | Normalne aktivnosti: hodanje, trčanje, brzo sedanje, ležanje, penjanje uz stepenice. |
| **1** | **FALL** | Stvarni padovi: pad napred, pad unazad, bočni pad, pad usled sinkope / gubitka svesti. |

---

## 4. Specifikacija i Konfiguracija ML Modela

U projektu se implementiraju i komparativno analiziraju tri različita algoritma koji predstavljaju tri različite paradigme računarske inteligencije:

### 4.1. Gaussian Naive Bayes (Probabilistički model)

* **Uloga:** Služi kao bazični model (baseline) za brzu uslovnu verovatnoću.
* **Princip rada:** Pretpostavlja Gausovu raspodelu neprekidnih atributa i računa uslovne verovatnoće za klase na osnovu statistika izvučenih iz vremenskog prozora. Odlikuje se izuzetno niskom računarskom složenošću.

### 4.2. k-Nearest Neighbors / kNN (Metrički model)

* **Uloga:** Klasifikacija na osnovu geometrijske bliskosti u prostoru obeležja.
* **Princip rada:** Novi vremenski prozor se klasifikuje analizom $k$ najbližih profila kretanja iz trening skupa, primenom **Euklidske udaljenosti**.
* **Optimizacija:** U sklopu projekta biće izvršena pretraga i optimizacija hiperparametra $k$.

### 4.3. Višeslojni Perceptron / MLP Neuronska Mreža (Model dubokog učenja)

* **Uloga:** Nelinearno modelovanje kompleksnih interakcija između senzorskih atributa.
* **Arhitektura:**
    * **Ulazni sloj:** Prihvata vektor ekstrahovanih statističkih obeležja iz klizećeg prozora.
    * **Skriveni slojevi:** Dva potpuno povezana skrivena sloja sa **ReLU** aktivacionim funkcijama za uvođenje nelinearnosti.
    * **Izlazni sloj:** Jedan izlazni neuron sa **Sigmoid** aktivacionom funkcijom koji generiše verovatnoću pripadnosti klasi za binarnu klasifikaciju.

---

## 5. Metodologija Evaluacije i Validacije Modela

### 5.1. Strategija rešavanja debalansa klasa

Dataset je prirodno debalansiran, s obzirom na to da postoji znatno više primera hodanja i svakodnevnih aktivnosti nego samih padova. Zbog toga se **ukupna tačnost (Accuracy) eksplicitno odbacuje** kao primarna metrika, jer može dati lažno optimistične rezultate.

### 5.2. Metrike uspešnosti (Performance Metrics)

Za evaluaciju svih modela koristiće se sledeći set metrika:

* **Matrica konfuzije (Confusion Matrix):** Prikaz tačnih i pogrešnih klasifikacija za sve tri metode.
* **Osetljivost / Odziv (Sensitivity / Recall):** **Ključna metrika projekta**. Meri procenat stvarnih padova koje je sistem uspešno prepoznao. U ovom domenu primene, cilj je maksimizovati Recall (težiti ka 100%), jer je sa medicinskog aspekta daleko prihvatljivije imati lažnu uzbunu (False Positive) nego propustiti stvarni pad pacijenta (False Negative).
* **F1-Score:** Harmonijska sredina preciznosti i odziva, koja pruža realnu i balansiranu ocenu stabilnosti modela na debalansiranom skupu podataka.

### 5.3. Evaluacija računarske efikasnosti (Real-time Constraints)

* **Inference Time (Vreme izvršavanja):** Eksperimentalno će se izmeriti vreme u milisekundama koje je potrebno svakom modelu da donese odluku nad jednim vremenskim prozorom. S obzirom na to da sistem treba da radi u realnom vremenu, brzina donošenja odluke predstavlja ključni faktor pri izboru konačnog modela.

---

## 6. Tehnološki Stek za Razvoj Modela

Sve faze istraživanja, pretprocesiranja, treniranja i evaluacije modela biće implementirane korišćenjem sledećeg ekosistema:

* **Obrada podataka i ekstrakcija obeležja:** `Pandas`, `NumPy`, `SciPy`.
* **Algoritmi računarske inteligencije:**
    * `Scikit-learn` (za implementaciju Gaussian Naive Bayes i kNN algoritama, kao i za evaluaciju).
    * `TensorFlow` / `Keras` (za izradu i treniranje MLP neuronske mreže).
* **Vizuelizacija rezultata i matrica:** `Matplotlib`, `Seaborn`.