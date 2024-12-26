FROM python:3.10-slim

WORKDIR /app
COPY . /app

# Install dependencies in one step to reduce layers
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt werkzeug==2.2.2

# Expose the necessary port
EXPOSE 5000

CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "remove_bg:remove_bg"]
