FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY server.py .
COPY config.json .

# Create directory for GeoIP databases
RUN mkdir -p /app/data

# Set environment variables
ENV GEOIP_CITY_DB=/app/data/GeoLite2-City.mmdb
ENV GEOIP_ASN_DB=/app/data/GeoLite2-ASN.mmdb
ENV GEOIP_COUNTRY_DB=/app/data/GeoLite2-Country.mmdb
ENV GEOIP_CACHE_TTL=3600

# Expose port (if needed for HTTP interface)
EXPOSE 8000

# Run the server
CMD ["python", "server.py"]
