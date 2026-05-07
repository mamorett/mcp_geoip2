# GeoIP MCP Server

A Model Context Protocol (MCP) server providing IP geolocation and ASN lookup services using MaxMind GeoIP2 databases.

---

## Features

- **Single & Bulk IP Geolocation:** Lookup for IPv4 and IPv6 addresses, with batch concurrency (configurable via `GEOIP_CONCURRENCY`).
- **ASN Information:** Retrieve Autonomous System Number and organization.
- **Country-only Lookup:** Efficient country-level lookup tool.
- **Comprehensive Location Data:** Country, city, subdivision, postal code, latitude/longitude, and network info.
- **Distance Calculation:** Compute distance between two coordinates.
- **Caching:** In-memory cache for fast repeated lookups (configurable via `GEOIP_CACHE_TTL`).
- **Flexible Output:** JSON, summary, and CSV formats.
- **Health Checks & Performance Monitoring:** Tools for server and database health.
- **Prometheus Metrics:** Optional exporter via `PROMETHEUS_PORT`.
- **MCP Protocol:** Compatible with MCP clients.

---

## Setup

### Option 1: Automatic Download (Recommended)

The server can automatically download and update the required databases if you provide a MaxMind License Key.

1.  **Get a License Key**: Register for a free MaxMind account at https://www.maxmind.com/en/geolite2/signup and generate a license key.
2.  **Set Environment Environment Variable**:
    ```sh
    export MAXMIND_LICENSE_KEY="your_license_key_here"
    ```
3.  **Run the Server**: The server will download the databases to `~/.local/share/mcp_geoip2/` on startup.

### Option 2: Manual Installation

1.  **Download Databases**: Download `GeoLite2-City.mmdb`, `GeoLite2-ASN.mmdb`, and optionally `GeoLite2-Country.mmdb` from MaxMind.
2.  **Place Files**: Move the `.mmdb` files to the default data directory:
    ```sh
    mkdir -p ~/.local/share/mcp_geoip2/
    mv *.mmdb ~/.local/share/mcp_geoip2/
    ```

### Configuration (Optional)

You can override the default paths if needed:

```sh
export GEOIP_CITY_DB="/path/to/GeoLite2-City.mmdb"
export GEOIP_ASN_DB="/path/to/GeoLite2-ASN.mmdb"
export GEOIP_COUNTRY_DB="/path/to/GeoLite2-Country.mmdb"
export GEOIP_CACHE_TTL=3600
export GEOIP_CONCURRENCY=20
export PROMETHEUS_PORT=9000
```

---

## Usage

### Run with `uvx` (Recommended)

You can run the server directly without installing dependencies manually:

```sh
uvx geoip2-mcp-server
# OR
uvx mcp_geoip2
```

### Local Dev Setup

```sh
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the server
python -m geoip2_mcp_server.server
```

The server communicates via MCP protocol (stdio by default).

---

### Claude Code Configuration

Add the GeoIP MCP server to your Claude Code configuration.

**Option A: Global config** (`~/.claude/claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "geoip2": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/mamorett/mcp_geoip2",
        "mcp_geoip2"
      ],
      "env": {
        "MAXMIND_LICENSE_KEY": "your_license_key_here"
      }
    }
  }
}
```

**Option B: Project config** (`.mcp.json` in your project root)

```json
{
  "mcpServers": {
    "geoip2": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/mamorett/mcp_geoip2",
        "mcp_geoip2"
      ],
      "env": {
        "MAXMIND_LICENSE_KEY": "your_license_key_here",
        "GEOIP_CACHE_TTL": "3600",
        "GEOIP_CONCURRENCY": "20"
      }
    }
  }
}
```

**Option C: Local dev (from source)**

```json
{
  "mcpServers": {
    "geoip2": {
      "command": "python",
      "args": ["-m", "geoip2_mcp_server.server"],
      "env": {
        "MAXMIND_LICENSE_KEY": "your_license_key_here"
      }
    }
  }
}
```

After adding the configuration, restart Claude Code. The server will be available as `mcp__mcp_geoip2` with all geolocation tools.

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
Batch geolocation for multiple IPs (concurrent, configurable).

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

### 6. `geolocate_country`
Country-only geolocation for a single IP.

**Parameters:**
- `ip_address` (string, required)
- `output_format` (json|summary|csv, default: json)
- `use_cache` (boolean, default: true)

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
- Prometheus metrics available if `PROMETHEUS_PORT` is set.

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