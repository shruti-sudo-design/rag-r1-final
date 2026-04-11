FROM python:3.10-slim

WORKDIR /app

# Install CPU-only PyTorch (smallest possible — no CUDA)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

RUN mkdir -p corpora

ENV ENABLE_WEB_INTERFACE=TRUE

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
