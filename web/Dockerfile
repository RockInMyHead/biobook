FROM node:22-bookworm-slim AS css
WORKDIR /src
COPY package.json package-lock.json ./
RUN npm ci
COPY tailwind.config.js ./
COPY src ./src
COPY templates ./templates
RUN npm run build:css

FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends bubblewrap ffmpeg fonts-dejavu-core && rm -rf /var/lib/apt/lists/* \
  && pip install --no-cache-dir -r requirements.txt

COPY . .
COPY --from=css /src/static/tailwind.css /app/static/tailwind.css

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000"]
