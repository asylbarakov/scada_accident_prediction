FROM wind-scada-baseline:1.0
USER root
COPY backend/requirements.txt backend/requirements.lock.txt /opt/
RUN pip install --no-cache-dir -r /opt/requirements.lock.txt && pip check
COPY baseline /work/baseline
COPY backend /work/backend
WORKDIR /work/backend
ENV PYTHONPATH=/work:/work/backend
USER 10001:10001
ENTRYPOINT []
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]
