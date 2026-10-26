# Customer Purchase & Churn — demo locale

Applicazione didattica Python, MongoDB, FastAPI e Streamlit per mostrare due pipeline di classificazione distinte. Tutti i record generati, le etichette e le predizioni sono **sintetici**: non rappresentano clienti, definizioni aziendali o risultati validi su dati reali.

## Definizioni sintetiche adottate

Per consentire una demo funzionante senza inventare requisiti di business, i target hanno significato esclusivamente tecnico:

- **Purchase**: esiste almeno un ordine sintetico con stato `completed` nei 30 giorni successivi all'osservazione.
- **Churn**: esiste un evento sintetico `synthetic_churn` nei 30 giorni successivi. È generato separatamente dagli ordini: la mancanza di acquisti non significa churn.
- L'etichetta è disponibile il giorno dopo la fine della finestra di outcome. Le feature sono una fotografia al tempo di osservazione e non usano gli eventi futuri.
- Il generatore usa formule e seed documentati in `src/customer_ml/synthetic.py`. Le formule sono un meccanismo di simulazione, non conoscenza di dominio.

Per un impiego aziendale occorre sostituire queste definizioni, regole di generazione e dati sintetici con requisiti e dati approvati.

## Prerequisiti

- Windows 11 con Docker Desktop installato e avviato in modalità Linux containers.
- Docker Compose v2 (incluso in Docker Desktop).
- PowerShell per i comandi qui sotto.
- Non è necessario installare Python, MongoDB o le dipendenze ML sull'host.

Verifica da PowerShell:

```powershell
docker --version
docker compose version
docker info
```

L'applicazione non richiede account cloud o servizi esterni. Il download iniziale delle immagini Docker e delle dipendenze Python durante la build richiede accesso ai registry.

## Struttura

```text
compose.yaml                         Servizi MongoDB, API e UI
Dockerfile                           Immagine Python condivisa
pyproject.toml                       Dipendenze versionate
scripts/                             Inizializzazione, dati, validazione, training
src/customer_ml/                     Persistenza, generazione, ML, API e UI
tests/                               Test unitari, API e integrazione facoltativa
docs/                                Architettura, dati, metodologia e operazioni
```

## Avvio su Windows

Esegui i comandi dalla directory principale del progetto in PowerShell.

1. Crea la configurazione locale ed edita `.env`. Sostituisci entrambi i placeholder password con password casuali differenti; non inserire `.env` nel controllo versione.

   ```powershell
   Copy-Item .env.example .env
   notepad .env
   ```

2. Controlla la configurazione e costruisci/avvia i servizi:

   ```powershell
   docker compose config
   docker compose up --build -d
   docker compose ps
   ```

3. Inizializza collection e indici, genera i dati e controlla i due target:

   ```powershell
   docker compose run --rm api python scripts/init_database.py
   docker compose run --rm api python scripts/generate_demo_data.py
   docker compose run --rm api python scripts/validate_dataset.py
   ```

   Lo stesso seed riproduce gli stessi documenti; il caricamento ripetuto usa ID stabili e upsert. `DEMO_SEED` e `DEMO_CUSTOMERS` sono configurabili nel file `.env`.

4. Esegui esplicitamente il training dei due modelli:

   ```powershell
   docker compose run --rm api python scripts/train_models.py
   ```

   Il training usa split cronologico 70/15/15 per periodo di osservazione; addestra solo sul primo segmento e richiede entrambe le classi in train, validation e test. Non viene eseguito automaticamente all'avvio dell'API o della UI. I modelli pubblicati sono artefatti prodotti localmente, valutati sul test set sintetico e identificati nei metadati.

5. Apri l'interfaccia locale:

   - UI Streamlit: <http://localhost:18501> (configurato nel `.env` locale perché la porta predefinita 8501 può essere occupata)
   - API e OpenAPI: <http://localhost:18000/docs> (configurato nel `.env` locale perché la porta predefinita 8000 era occupata)
   - Stato API: <http://localhost:18000/health>
   - Stato database: <http://localhost:18000/health/database>

   Le porte host attive sono `API_HOST_PORT=18000` e `UI_HOST_PORT=18501` nel `.env` locale; i valori predefiniti di `.env.example` restano 8000 e 8501. MongoDB non pubblica porte sull'host. API e UI sono associate a `127.0.0.1`; i container comunicano tramite rete Docker e nome servizio.

## API principali

| Metodo | Percorso | Descrizione |
|---|---|---|
| `GET` | `/health` | Stato del processo API |
| `GET` | `/health/database` | Ping MongoDB; HTTP 503 se non disponibile |
| `GET` | `/models` | Stato e metadati dei modelli sintetici |
| `GET` | `/metrics` | Metriche del test set riportate nei metadati |
| `GET` | `/customers?limit=100` | Elenco limitato di clienti sintetici |
| `GET` | `/customers/{customer_id}` | Feature dell'ultimo snapshot sintetico |
| `POST` | `/predictions/{purchase\|churn}/{customer_id}` | Predice e registra l'esito |
| `GET` | `/predictions?target=purchase&limit=100` | Storico sintetico delle predizioni |

Input non valido, cliente mancante, database non disponibile o modello mancante/incompatibile producono errori HTTP espliciti. Una predizione richiede sia il file artefatto sia metadati approvati e coerenti.

## Test

L'immagine contiene pytest e le dipendenze di test. Dalla radice:

```powershell
docker compose run --rm api pytest -q
```

I test unitari e API non richiedono un database. I test che richiedono MongoDB usano solo un database di test dedicato configurato tramite `MONGO_TEST_URI`; non usare l'URI del database demo. Al momento l'integrazione completa richiede MongoDB avviato e configurazione esplicita di tale URI.

## Predizioni e persistenza

Le richieste di inferenza passano da FastAPI, caricano un artefatto validato compatibile con lo schema delle feature e persistono il risultato in `predictions`. Streamlit interroga solo le API, mai MongoDB direttamente. L'API non inventa output se modello o snapshot non sono disponibili.

Le metriche mostrate sono del test set sintetico per la specifica esecuzione. Precision, recall, F1, matrice di confusione, ROC-AUC e average precision non sono stime di performance reale. La soglia 0.5 è la soglia predefinita tecnica del classificatore, non una soglia operativa approvata.

## Arresto, persistenza, backup e ripristino

Arresta i servizi mantenendo i dati:

```powershell
# Comando di arresto e clean
docker compose down
# Comando di arresto e clean con cancellazione volumi
docker compose down -v
```

Riavvia mantenendo i volumi:

```powershell
docker compose up -d
```

Il database e gli artefatti sono in volumi Docker nominati (`mongo_data`, `model_data`). **Non** usare `docker compose down -v` se si vogliono conservare dati o artefatti: la rimozione dei volumi è distruttiva.

Backup logico del database (verifica prima che i Database Tools siano presenti):

```powershell
docker compose exec mongo mongodump --version
docker compose exec mongo sh -c 'mongodump --username "$MONGO_INITDB_ROOT_USERNAME" --password "$MONGO_INITDB_ROOT_PASSWORD" --authenticationDatabase admin --db "$MONGO_INITDB_DATABASE" --out /tmp/customer-backup'
docker compose cp mongo:/tmp/customer-backup .\customer-backup
docker compose cp api:/app/models .\model-backup
```

Per il restore, ferma prima API e UI e conserva una copia dello stato corrente. I comandi seguenti **sovrascrivono le collection della demo**; usali solo dopo aver verificato la directory del backup:

```powershell
docker compose stop api ui
docker compose cp .\customer-backup mongo:/tmp/customer-restore
docker compose exec mongo sh -c 'mongorestore --username "$MONGO_INITDB_ROOT_USERNAME" --password "$MONGO_INITDB_ROOT_PASSWORD" --authenticationDatabase admin --drop --dir /tmp/customer-restore --nsInclude "$MONGO_INITDB_DATABASE.*"'
docker compose cp .\model-backup\models api:/app/models
docker compose start api ui
```

Se `mongodump`/`mongorestore` non sono inclusi nell'immagine MongoDB selezionata, usa MongoDB Database Tools compatibili da un container di tools nella rete Compose. Backup, restore e migrazione non sono stati eseguiti in questa sessione; prova il ripristino su una copia isolata prima di farvi affidamento. In un ambiente reale, predisporre backup cifrati e verifiche periodiche.

Al primo avvio MongoDB crea l'utente applicativo con privilegi `readWrite` sul solo database della demo. Le variabili di creazione vengono applicate solo quando il volume MongoDB è vuoto; cambiare `.env` non ruota automaticamente le credenziali su un volume già inizializzato.

## Risoluzione problemi

- **Docker daemon non raggiungibile**: avvia Docker Desktop, attendi che sia pronto e riprova `docker info`.
- **Credenziali mancanti**: controlla `.env` e che i valori non siano i placeholder di `.env.example`.
- **Database non disponibile**: controlla `docker compose ps` e `docker compose logs mongo api`.
- **Modelli non disponibili**: verifica che i dati siano stati caricati/validati ed esegui `train_models.py`.
- **Training respinto per classi insufficienti**: non pubblicare un modello; aumenta il numero dei dati sintetici o correggi i parametri del generatore mantenendo esplicita la natura artificiale.
- **Dati non aggiornati dopo modifica password MongoDB**: l'utente nel volume esistente deve essere ruotato tramite procedura amministrativa MongoDB; il bootstrap non viene rieseguito automaticamente.

## Limiti e sicurezza

La demo non implementa autenticazione utente e non è destinata a reti pubbliche o produzione. Le porte pubblicate sono vincolate a loopback; MongoDB resta interno alla rete Compose. Le credenziali di esempio devono essere sostituite localmente. Non inserire dati personali reali. Le feature sintetiche e gli eventi generati non costituiscono un modello di clienti.
