FROM python:3.8-slim
WORKDIR /app
ENV PYTHONPATH=/app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple
COPY . /app

CMD ["python", "main.py"]
