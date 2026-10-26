# Metodologia ML

## Significato limitato dei target

- **Purchase** = almeno un ordine `completed` futuro in 30 giorni.
- **Churn** = un evento futuro `synthetic_churn` in 30 giorni. È indipendente dagli ordini e non si deriva dal silenzio di acquisto.

Il generatore usa un seed predefinito `20261025` e formule probabilistiche visibili nel codice. Le feature sintetiche sono costruite come fotografia storica al tempo `observation_at`; gli eventi di outcome cadono dopo tale data. Le formule sono solo generatori di esempi e non descrivono distribuzioni reali.

## Training e leakage

Ogni target ha pipeline separata. I periodi distinti sono ordinati; il 70% iniziale è train, il successivo 15% validation e l'ultimo 15% test. Il preprocessing (imputazione, scaling numerico e one-hot categorico) è fit all'interno della pipeline solo sul training set. Logistic Regression è il baseline. Validation è mantenuto separato e non utilizzato per tuning nella demo; il test non guida la selezione.

Un modello non viene scritto/promosso se i dati sono insufficienti o uno split non contiene entrambe le classi. Le feature devono seguire lo schema versionato `1.0`. Gli artefatti vengono scritti nel volume dedicato, con sostituzione atomica del file.

## Metriche

La classe positiva è `1`; la soglia 0.5 è il comportamento predefinito del classificatore, non una soglia aziendale. Si registrano precision, recall, F1, matrice di confusione, ROC-AUC e average precision sul solo test set cronologico. Class weights bilanciati sono parte del baseline. Le metriche descrivono esclusivamente l'esecuzione sul dataset sintetico e non la performance su dati reali.

Gli artefatti joblib possono eseguire codice durante la deserializzazione: l'applicazione li carica esclusivamente dal volume locale controllato, mai da upload o fonti esterne. Il volume non deve contenere file non attendibili.
