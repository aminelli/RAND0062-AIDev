# Modello dati MongoDB

I documenti di dominio sintetici hanno `synthetic: true`; i metadati dei modelli hanno `synthetic_only: true`. Non sono acquisiti dati personali o esterni. Le date sono UTC (`datetime` BSON); gli ID stabili consentono upsert ripetibili per lo stesso seed.

| Collection | Campi principali | Vincoli/indici |
|---|---|---|
| `customers` | `_id`/`customer_id` (stringa), `created_at` (UTC), `synthetic` (bool) | `customer_id` univoco |
| `orders` | `_id`, `customer_id`, `occurred_at`, `status`, `amount`, `synthetic` | indice cliente + data; gli ordini demo validi sono `completed` |
| `customer_features` | `_id`/`snapshot_id`, `customer_id`, `observation_at`, `feature_schema_version`, `features` | coppia cliente + istante univoca |
| `training_labels` | `_id`, `snapshot_id`, `customer_id`, `target`, `label`, `observation_at`, `outcome_start/end`, `available_at`, `synthetic` | target `purchase`/`churn`, label binaria; snapshot + target univoco |
| `churn_events` | `_id`, `customer_id`, `occurred_at`, `event_type`, `synthetic` | indice cliente + data; evento di simulazione esplicito |
| `predictions` | ID Mongo, cliente, target, versione modello, classe, probabilità, timestamp, flag sintetico | indice cliente + timestamp |
| `model_metadata` | target/versione, schema feature, parametri, seed, date, conteggi, metriche, artifact, stato demo | indice target + data; `_id` target/versione |

`customer_features.features` contiene `age_years`, `orders_365d`, `days_since_last_order`, `avg_order_value`, `support_tickets_90d` e `plan_tier`. La media ordine è `null` senza ordini storici. La collection label riferisce la snapshot via `snapshot_id`; non si fa affidamento su foreign key MongoDB.

La validazione applicativa è affiancata da validator MongoDB `$jsonSchema` al momento della creazione delle collection. Le collection create in precedenza non vengono modificate automaticamente dal bootstrap; le regole server-side e retention vanno evolute con migrazioni esplicite. Le collection hanno indici per unicità e interrogazioni principali. L'inserimento di predizioni non sovrascrive la storia.

## Disponibilità etichette

La finestra sintetica inizia all'istante di osservazione, dura 30 giorni e `available_at` è un giorno dopo `outcome_end`. Training include soltanto target marcati sintetici e verifica la coerenza temporale. La simulazione genera direttamente gli eventi futuri coerenti con i label; tali eventi non sono osservati dalle feature dello stesso snapshot.
