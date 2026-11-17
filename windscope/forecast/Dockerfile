FROM python:3.12.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
COPY forecast/requirements.txt forecast/requirements.lock.txt /opt/
RUN pip install --no-cache-dir -r /opt/requirements.lock.txt && pip check
COPY project/scada_pdm /work/reference/scada_pdm
COPY forecast /work/forecast
ENV PYTHONPATH=/work:/work/reference
WORKDIR /work
USER 10001:10001
CMD ["uvicorn", "forecast.server:app", "--host", "0.0.0.0", "--port", "8002"]
