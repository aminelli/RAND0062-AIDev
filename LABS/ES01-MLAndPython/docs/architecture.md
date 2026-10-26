# Architettura

## Servizi e flussi

```text
Browser host -> Streamlit (127.0.0.1:8501)
                    | HTTP, rete Compose
                    v
              FastAPI (127.0.0.1:8000) -> MongoDB (rete interna, 27017)
                    |
                    +-> volume model_data (artefatti approvati)

Comando esplicito scripts/train_models.py -> MongoDB + model_data
```

Compose definisce i servizi `mongo`, `api`, `ui` e una rete bridge `backend`. Solo API e UI sono pubblicate su loopback host; MongoDB non ha porte host. I volumi nominati `mongo_data` e `model_data` persistono rispettivamente database e artefatti.

La UI invia richieste HTTP a FastAPI e non contiene credenziali/database client. L'API è l'unico punto di accesso applicativo per feature e predizioni. Il processo di training è separato dal lifecycle dell'API/UI: si avvia con un comando one-shot esplicito.

## Responsabilità

- `database.py`: connessione, collection e indici MongoDB.
- `synthetic.py`: dataset deterministico con eventi futuri e target sintetici.
- `training.py`: join delle feature con etichette, split temporale, pipeline e metriche.
- `api.py`: health, consultazione limitata, inferenza e persistenza della predizione.
- `ui.py`: interfaccia browser, esclusivamente via API.
- `scripts/`: bootstrap, generazione, validazione e training invocabili.

MongoDB usa un utente applicativo con ruolo `readWrite` solo sul database della demo. L'utente root è usato unicamente dal bootstrap MongoDB. Le credenziali arrivano dall'ambiente e non sono versionate.

## Disponibilità ed errori

MongoDB ha un health check autenticato; API e UI attendono le dipendenze sane tramite `depends_on`. Il driver ha un timeout di selezione server. API health indica la disponibilità del processo; `/health/database` verifica separatamente il DB. Errori di modello, database e input non vengono trasformati in risultati di successo.

Gli artefatti joblib sono generati localmente dal training controllato e mai accettati da input dell'utente. La promozione è registrata nei metadati solo dopo split, training e calcolo di metriche valido. L'inferenza verifica target, versione schema e versione dei metadati.

## Limiti

Ambiente locale didattico, senza autenticazione applicativa, orchestrazione distribuita o alta disponibilità. Non esporre API/UI a reti non fidate. I dati e i target sintetici non hanno significato aziendale.
