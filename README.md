# GeoIP MCP Server

A Model Context Protocol (MCP) server providing IP geolocation and ASN lookup services using MaxMind GeoIP2 databases.

---

## Features

- **Single & Bulk IP Geolocation:** Lookup for IPv4 and IPv6 addresses.
- **ASN Information:** Retrieve Autonomous System Number and organization.
- **Comprehensive Location Data:** Country, city, subdivision, postal code, latitude/longitude, and network info.
- **Distance Calculation:** Compute distance between two coordinates.
- **Caching:** In-memory cache for fast repeated lookups.
- **Flexible Output:** JSON, summary, and CSV formats.
- **Health Checks & Performance Monitoring:** Tools for server and database health.
- **MCP Protocol:** Compatible with MCP clients.

---

## Setup

### 1. Install Dependencies

```sh
pip install -r requirements.txt
```

### 2. Download MaxMind GeoIP2 Databases

- Register for a free MaxMind account: https://www.maxmind.com/
- Download `GeoLite2-City.mmdb` and `GeoLite2-ASN.mmdb`.
- Place them in a known directory (e.g., `~/Downloads/`).

### 3. Set Environment Variables

```sh
export GEOIP_CITY_DB="~/Downloads/GeoLite2-City.mmdb"
export GEOIP_ASN_DB="~/Downloads/GeoLite2-ASN.mmdb"
# Optional:
export GEOIP_COUNTRY_DB="~/Downloads/GeoLite2-Country.mmdb"
export GEOIP_CACHE_TTL=3600
```

---

## Usage

### Run the Server

```sh
python server.py
```

The server communicates via MCP protocol (stdio by default).

---

## Available Tools

### 1. `geolocate_ip`
Get geolocation for a single IP.

**Parameters:**
- `ip_address` (string, required)
- `include_asn` (boolean, default: true)
- `output_format` (json|summary|csv, default: json)
- `use_cache` (boolean, default: true)

---

### 2. `geolocate_multiple_ips`
Batch geolocation for multiple IPs.

**Parameters:**
- `ip_addresses` (array of strings, required)
- `include_asn` (boolean, default: true)
- `output_format` (json|summary|csv, default: json)
- `use_cache` (boolean, default: true)

---

### 3. `get_asn_info`
Get ASN info for an IP.

**Parameters:**
- `ip_address` (string, required)

---

### 4. `calculate_distance`
Calculate distance between two coordinates.

**Parameters:**
- `lat1`, `lon1` (float, required): Point 1
- `lat2`, `lon2` (float, required): Point 2
- `unit` (km|mi, default: km)

---

### 5. `server_management`
Server operations.

**Parameters:**
- `action` (clear_cache|get_stats|reload_databases, required)

---

## Example Output

```json
{
  "ip_address": "203.0.113.0",
  "location": {
    "country": {"iso_code": "US", "name": "United States"},
    "city": {"name": "Minneapolis"},
    "location": {"latitude": 44.9733, "longitude": -93.2323}
  },
  "asn": {
    "autonomous_system_number": 1221,
    "autonomous_system_organization": "Telstra Pty Ltd"
  }
}
```

---

## Error Handling

- Invalid IP address formats
- Missing or unreadable database files
- IP not found in database
- Network/database errors

---

## Health & Monitoring

- Use `health_check.py` for system/database health.
- Use `performance_monitor.py` for performance metrics.

---

## Docker

A [Dockerfile](Dockerfile) and [docker-compose.yaml](docker-compose.yaml) are provided.

```sh
docker-compose up --build
```

---

## License

This project is licensed under the [GNU GPL v3](LICENSE).

---

## Credits