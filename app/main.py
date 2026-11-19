"""
WEEK 7 STARTER (part 2 of 2) — serve the model over HTTP.

The model is a file on disk. This file is what turns it into something a care
management tool can call. Two endpoints, one model load, no surprises.

Fill in every section marked TODO. When you are done:

    uvicorn app.main:app --reload --port 8000
    open http://localhost:8000/docs

Check your work:
    pytest tests/test_api.py -v
"""

import joblib
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pathlib import Path

from app.schemas import MemberFeatures, PredictionResponse
from config import HIGH_COST_THRESHOLD_USD

MODEL_PATH = Path("model.joblib")
DECISION_THRESHOLD = 0.5  

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    TODO 1 — load the model once, at startup.

    Set the module-level `model` (you will need `global model`).

    Why here and not inside the endpoint: joblib.load() on every request would
    add hundreds of milliseconds and re-read the file each time. Load once,
    answer many.

    Fail loudly if MODEL_PATH does not exist: a service that starts up happily
    and then 500s on every request is worse than one that refuses to boot.
    Raise RuntimeError with a message that tells the operator to run train.py.

    Everything before `yield` runs at startup, everything after at shutdown.
    """
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"{MODEL_PATH} not found. Run `python train.py` to create it.")
    model = joblib.load(MODEL_PATH)
    yield
    model = None


app = FastAPI(
    title="Claims Risk API",
    description="Predicts whether a member is likely to exceed the plan-paid threshold.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """
    TODO 2 — return {"status": "ok", "model_loaded": <bool>}.

    Render polls this endpoint to decide whether your service is alive, so it
    must not do any real work. Report whether the model actually loaded rather
    than hard-coding True; "the process is up" and "the service can serve" are
    different claims.
    """
    # check if model loaded well
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: MemberFeatures):
    """
    TODO 3 — score one member.

    Steps:
      1. If the model somehow is not loaded, raise HTTPException(status_code=503).
      2. Turn `features` into a one-row DataFrame.
      3. Get the probability of the positive class.
      4. Compare against DECISION_THRESHOLD to get the bool.
      5. Return a PredictionResponse, rounding the probability to 4 places and
         echoing HIGH_COST_THRESHOLD_USD.

    Resist the urge to "fix up" the row on the way in — no rescaling, no unit
    conversion, no reordering columns. The pipeline you saved has its
    preprocessing inside it, and anything you do here happens on top of that.
    Errors of this kind are silent: no exception, just a different probability
    than the one your notebook reported for the same member. That is train/serve
    skew, and tests/test_api.py checks for it by comparing your endpoint against
    a direct model call across a spread of members.

    Also decide what a caller should get when the input is valid JSON but
    nonsense (pharmacy_claims greater than total_claims, say). FastAPI will not
    catch that for you.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    # Valid JSON can still describe an impossible member, so reject it rather than score it.
    if (features.inpatient_claims > features.medical_claims
            or features.specialty_drug_fills > features.pharmacy_claims
            or features.unique_diagnosis_codes > features.medical_claims):
        raise HTTPException(status_code=422, detail="Sub-counts cannot exceed their parent claim counts.")

    # One row, passed in as-is, because the pipeline does its own scaling and encoding.
    row = pd.DataFrame([features.model_dump()])
    proba = float(model.predict_proba(row)[0, 1])

    return PredictionResponse(
        is_high_cost=proba >= DECISION_THRESHOLD,
        probability=round(proba, 4),
        threshold_dollars=HIGH_COST_THRESHOLD_USD,
    )
