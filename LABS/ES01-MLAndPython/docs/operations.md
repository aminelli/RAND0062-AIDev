# Operazioni

I comandi sono eseguiti dalla directory principale del progetto in PowerShell. La sequenza di installazione, avvio, seed, training e test si trova nel README.

## Configurazione

`compose.yaml` richiede variabili non vuote per gli utenti MongoDB. Copiare `.env.example` in `.env`, impostare password locali robuste e tenerlo fuori dal controllo versione. Cambiare credenziali dopo l'inizializzazione di un volume richiede una procedura di rotazione MongoDB; il bootstrap init non viene rieseguito sul volume esistente.

La UI e API sono raggiungibili solo dall'host locale (`127.0.0.1`). MongoDB è raggiungibile solo dai servizi nella rete `backend`. La porta API predefinita è 8000, UI 8501; entrambe sono configurabili nell'env file. Non configurare un binding pubblico per la demo.

## Dati e training

`init_database.py` verifica con ping, crea le collection e indici. `generate_demo_data.py` accetta seed e numero di clienti da `.env`; gli upsert sono idempotenti. `validate_dataset.py` controlla presenza di target e split cronologici. `train_models.py` allena entrambi i casi d'uso e registra metriche test, versioni e campioni.

Gli artefatti (`purchase.joblib`, `churn.joblib`) risiedono nel volume `model_data`; i metadati stanno in MongoDB. Il sistema di inferenza richiede corrispondenza tra file, metadati approvati e schema feature. Training non parte all'avvio del container.

## Arresto e dati persistenti

`docker compose down` arresta/rimuove i container ma preserva i volumi nominati. `docker compose down -v` elimina i volumi e quindi **cancella irreversibilmente i dati locali**. Evitare quest'ultimo comando per l'uso normale.

Backup logico: verificare `docker compose exec mongo mongodump --version`, quindi `mongodump` nel container e `docker compose cp` per trasferire `/tmp/customer-backup` all'host. Esportare anche gli artefatti con `docker compose cp api:/app/models .\model-backup`. Il README riporta i comandi Windows completi. Per il restore fermare API/UI e ripristinare solo dopo aver salvato lo stato esistente: il comando `mongorestore --drop` sostituisce collection e non va usato sul database sbagliato. Verificare la presenza di Database Tools e testare il restore in ambiente isolato; questi comandi non sono stati eseguiti nella sessione. Il backup dei volumi raw richiede MongoDB arrestato e procedure Docker adeguate.

## Troubleshooting

- `docker info` fallisce: avviare Docker Desktop e verificare Linux containers.
- API non pronta: controllare health e log con `docker compose ps` / `docker compose logs api mongo`.
- `503` database: verificare credenziali e inizializzazione volume; password aggiornate non cambiano utenti preesistenti.
- `503` modello: generare dataset, validarlo e avviare esplicitamente il training.
- Training rifiutato: le partizioni devono contenere entrambe le classi; non ignorare l'errore né pubblicare un artefatto non validato.

La configurazione corrente è per una singola macchina locale e non include autenticazione API, alta disponibilità, audit o backup schedulato.
