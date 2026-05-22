FROM ultralytics/ultralytics:latest

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace/src

ENTRYPOINT ["python", "src/train.py"]
