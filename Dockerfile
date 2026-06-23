FROM python:3.12-slim

WORKDIR /app
RUN apt-get update \
    && apt-get install --yes --no-install-recommends gdal-bin libgdal-dev \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --no-cache-dir ".[geo,ml]"
COPY . .
CMD ["geowatch", "doctor", "--strict"]
