"""Minimal physical-only service. No database credentials or annotation mounts."""
from contextlib import asynccontextmanager
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from forecast.inference import H7Adapter, MODEL_ID

adapter = None


@asynccontextmanager
async def lifespan(app):
    global adapter
    adapter = H7Adapter(os.getenv('H7_MODEL_DIR', '/model'), os.getenv('H7_CONFIG', '/config/care_h7.json'))
    yield


app = FastAPI(title='WindScope CARE H7 inference', lifespan=lifespan)


class Sample(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    source_time: str
    values: dict[str, float | None]


class Input(BaseModel):
    model_config = ConfigDict(extra='forbid')
    asset_id: str
    source_time: str
    history: list[Sample] = Field(min_length=1, max_length=1009)


@app.get('/health')
def health():
    return {'ready': adapter is not None, 'model_id': MODEL_ID, 'horizon_hours': 168}


@app.post('/predict')
def predict(value: Input):
    try:
        return adapter.predict(value.asset_id, value.source_time, [s.model_dump() for s in value.history])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
