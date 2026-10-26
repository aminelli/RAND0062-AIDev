import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st

API_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")


def api_call(path: str, method: str = "GET"):
    request = Request(f"{API_URL}{path}", method=method)
    try:
        with urlopen(request, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        try:
            message = json.loads(exc.read().decode("utf-8")).get("detail", str(exc))
        except (ValueError, AttributeError):
            message = str(exc)
        raise RuntimeError(f"API returned HTTP {exc.code}: {message}") from exc
    except URLError as exc:
        raise RuntimeError(f"Cannot reach the API at {API_URL}: {exc.reason}") from exc


st.set_page_config(page_title="Customer ML demo", page_icon="📊")
st.title("Customer Purchase & Churn — demo sintetica")
st.warning(
    "Dati, target, metriche e predizioni sono sintetici e solo didattici; "
    "non rappresentano clienti o regole aziendali reali."
)

try:
    health = api_call("/health")
    st.caption(f"API: {health['status']}")
    models = api_call("/models")
    customers = api_call("/customers?limit=500")
    history = api_call("/predictions?limit=100")
except RuntimeError as error:
    st.error(str(error))
    st.stop()

model_by_target = {model["target"]: model for model in models}
target = st.selectbox("Caso d'uso", ["purchase", "churn"])
customer_ids = [customer["customer_id"] for customer in customers]
if not customer_ids:
    st.info("Nessun cliente sintetico. Genera i dati dal terminale.")
    st.stop()

customer_id = st.selectbox("Cliente sintetico", customer_ids)
try:
    snapshot = api_call(f"/customers/{customer_id}")
    st.subheader("Feature all'istante di osservazione")
    st.json(snapshot["features"])
except RuntimeError as error:
    st.error(str(error))

if model_by_target[target]["available"]:
    if st.button("Richiedi predizione"):
        try:
            result = api_call(f"/predictions/{target}/{customer_id}", method="POST")
            st.success(f"Classe prevista: {result['predicted_class']}")
            if result["probability"] is not None:
                st.metric("Probabilità stimata", f"{result['probability']:.1%}")
            st.caption(f"Versione modello: {result['model_version']}")
        except RuntimeError as error:
            st.error(str(error))
else:
    st.info("Modello non disponibile. Inizializza i dati e avvia esplicitamente il training.")

metadata = model_by_target[target]["metadata"]
if metadata:
    st.subheader("Metriche sul test set cronologico")
    st.json(
        {
            "metriche_test": metadata["metrics"],
            "campioni": metadata["sample_counts"],
            "versione": metadata["model_version"],
        }
    )
    st.caption("Risultati specifici del dataset sintetico, non validazione su dati reali.")
else:
    st.info("Metriche non disponibili: nessun modello validato per questa demo.")

st.subheader("Predizioni recenti registrate")
st.dataframe([row for row in history if row["target"] == target], use_container_width=True)
