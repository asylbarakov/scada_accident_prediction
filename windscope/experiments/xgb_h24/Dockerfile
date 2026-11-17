FROM wind-scada-baseline:1.0 AS openmp
FROM windscope-h7:1.0
USER root
COPY --from=openmp /usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0 /usr/lib/x86_64-linux-gnu/libgomp.so.1
COPY .wheelhouse/xgboost_cpu-3.1.2-py3-none-manylinux_2_28_x86_64.whl /tmp/wheels/
COPY .wheelhouse/scikit_learn-1.7.2-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl /tmp/wheels/
RUN pip install --no-index --no-deps /tmp/wheels/*.whl \
    && pip check \
    && rm /tmp/wheels/*.whl
USER 10001:10001
WORKDIR /experiment
CMD ["python", "run.py", "train"]
