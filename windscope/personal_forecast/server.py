from contextlib import asynccontextmanager
import os,math
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,ConfigDict,Field,field_validator
from personal_forecast.inference import PersonalAdapter,MODEL_ID

adapter=None


@asynccontextmanager
async def lifespan(app):
    global adapter
    adapter=PersonalAdapter(os.getenv('PERSONAL_REGISTRY','/registry'))
    yield


app=FastAPI(title='WindScope personal early-warning inference',lifespan=lifespan)


class Sample(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    source_time:str
    values:dict[str,float|None]

    @field_validator('values',mode='before')
    @classmethod
    def physical_numbers(cls,values):
        if not isinstance(values,dict) or any(v is not None and (isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v)) for v in values.values()):raise ValueError('Only finite physical measurements are accepted')
        return values


class Input(BaseModel):
    model_config=ConfigDict(extra='forbid')
    source_id:str
    asset_id:str
    source_time:str
    history:list[Sample]=Field(min_length=1,max_length=49)


@app.get('/health')
def health():return dict(ready=adapter is not None,model_id=MODEL_ID,horizon_hours=24,sources=list(adapter.models) if adapter else [])


@app.post('/predict')
def predict(value:Input):
    try:return adapter.predict(value.source_id,value.asset_id,value.source_time,[r.model_dump() for r in value.history])
    except ValueError as exc:raise HTTPException(status_code=422,detail=str(exc)) from exc
