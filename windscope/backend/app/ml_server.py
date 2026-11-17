"""No DB credentials, raw files, or annotations are available in this service."""
import os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel,ConfigDict
from baseline.inference import Measurement,ScadaXGBoostAdapter

adapter=None


@asynccontextmanager
async def lifespan(app):
    global adapter
    adapter=ScadaXGBoostAdapter(Path(os.environ['MODEL_PATH']))
    yield


app=FastAPI(title='WindScope model adapter',lifespan=lifespan)


class Input(BaseModel):
    model_config=ConfigDict(extra='forbid')
    source_time:str
    asset_id:str
    row_id:int
    partition:str
    scenario_id:str
    values:dict[str,float|None]


@app.get('/health')
def health():return {'ready':adapter is not None}


@app.post('/predict')
def predict(value:Input):
    return adapter.predict(Measurement(**value.model_dump())).to_dict()
