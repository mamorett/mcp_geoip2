#!/usr/bin/env python3
"""
Enhanced GeoIP MCP Server (FastMCP) with:
- Concurrency batching for multiple IP lookups (configurable via GEOIP_CONCURRENCY)
- Metrics (internal JSON resource + optional Prometheus exporter via PROMETHEUS_PORT)
- Country-only lookup tool

Requirements:
  pip install fastmcp geoip2 pydantic prometheus-client

Env vars:
  GEOIP_CITY_DB=~/Downloads/GeoLite2-City.mmdb
  GEOIP_ASN_DB=~/Downloads/GeoLite2-ASN.mmdb
  GEOIP_COUNTRY_DB=~/Downloads/GeoLite2-Country.mmdb
  GEOIP_CACHE_TTL=3600
  GEOIP_CONCURRENCY=20
  PROMETHEUS_PORT=9000  # optional; exposes metrics at http://0.0.0.0:9000/metrics
"""

import json
import os
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ipaddress
import geoip2.database
import geoip2.errors
from pydantic import BaseModel, Field, field_validator
from typing import List, Literal
import sys, asyncio, logging

from fastmcp import FastMCP
# Metrics
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("enhanced-geoip-fastmcp")

# -------------------------
# Metrics (Prometheus)
# -------------------------

REQUESTS_TOTAL = Counter("geoip_requests_total", "Total tool requests", ["tool"])
REQUEST_ERRORS_TOTAL = Counter("geoip_request_errors_total", "Total tool request errors", ["tool"])
LOOKUP_LATENCY = Histogram("geoip_lookup_latency_seconds", "Lookup latency (seconds)", ["kind"])  # city/asn/country
BATCH_SIZE = Histogram("geoip_batch_size", "Batch size for geolocate_multiple_ips")
CACHE_HITS = Counter("geoip_cache_hits_total", "Cache hits", ["kind"])  # single/batch
CACHE_MISSES = Counter("geoip_cache_misses_total", "Cache misses", ["kind"])
SERVER_UPTIME = Gauge("geoip_server_uptime_seconds", "Server uptime seconds")

def maybe_start_prometheus():
    port = os.getenv("PROMETHEUS_PORT")
    if port:
        try:
            port_i = int(port)
            start_http_server(port_i)
            logger.info(f"Prometheus metrics exporter started on :{port_i}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus exporter on {port}: {e}")

# -------------------------
# Cache
# -------------------------

@dataclass
class CacheEntry:
    data: Dict[str, Any]
    timestamp: datetime
    ttl: int = 3600

class GeoIPCache:
    def __init__(self, default_ttl: int = 3600):
        self.cache: Dict[str, CacheEntry] = {}
        self.default_ttl = default_ttl
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        entry = self.cache.get(key)
        if entry:
            if datetime.now() - entry.timestamp < timedelta(seconds=entry.ttl):
                self.hits += 1
                return entry.data
            else:
                del self.cache[key]
        self.misses += 1
        return None

    def set(self, key: str, data: Dict[str, Any], ttl: Optional[int] = None):
        ttl = ttl or self.default_ttl
        self.cache[key] = CacheEntry(data=data, timestamp=datetime.now(), ttl=ttl)

    def clear(self):
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total else 0
        return {
            "entries": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.2f}%"
        }

# -------------------------
# Config and state
# -------------------------

class ServerState:
    def __init__(self):
        self.city_db_path = os.path.expanduser(os.getenv("GEOIP_CITY_DB", "~/Downloads/GeoLite2-City.mmdb"))
        self.asn_db_path = os.path.expanduser(os.getenv("GEOIP_ASN_DB", "~/Downloads/GeoLite2-ASN.mmdb"))
        self.country_db_path = os.path.expanduser(os.getenv("GEOIP_COUNTRY_DB", "~/Downloads/GeoLite2-Country.mmdb"))
        cache_ttl = int(os.getenv("GEOIP_CACHE_TTL", "3600"))
        self.batch_concurrency = max(1, int(os.getenv("GEOIP_CONCURRENCY", "20")))

        self.cache = GeoIPCache(default_ttl=cache_ttl)
        self.request_count = 0
        self.start_time = datetime.now()
        self.db_info: Dict[str, Any] = {}

        # Lazy readers
        self._city_reader: Optional[geoip2.database.Reader] = None
        self._asn_reader: Optional[geoip2.database.Reader] = None
        self._country_reader: Optional[geoip2.database.Reader] = None

        self.validate_db_paths()

    def validate_db_paths(self):
        self.db_info = {}
        for db_name, db_path in [
            ("city", self.city_db_path),
            ("asn", self.asn_db_path),
            ("country", self.country_db_path),
        ]:
            p = Path(db_path)
            if p.exists():
                try:
                    stat = p.stat()
                    self.db_info[db_name] = {
                        "path": str(p),
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "available": True,
                    }
                    logger.info(f"{db_name.capitalize()} database found: {p}")
                except Exception as e:
                    logger.error(f"Error accessing {db_name} database: {e}")
                    self.db_info[db_name] = {"available": False, "error": str(e)}
            else:
                logger.warning(f"{db_name.capitalize()} database not found at {p}")
                self.db_info[db_name] = {"available": False, "error": "File not found"}

    def close_readers(self):
        for r in [self._city_reader, self._asn_reader, self._country_reader]:
            try:
                if r:
                    r.close()
            except Exception:
                pass
        self._city_reader = None
        self._asn_reader = None
        self._country_reader = None

    def get_city_reader(self) -> geoip2.database.Reader:
        if not Path(self.city_db_path).exists():
            raise FileNotFoundError(f"City database not found at {self.city_db_path}")
        if self._city_reader is None:
            self._city_reader = geoip2.database.Reader(self.city_db_path)
        return self._city_reader

    def get_asn_reader(self) -> geoip2.database.Reader:
        if not Path(self.asn_db_path).exists():
            raise FileNotFoundError(f"ASN database not found at {self.asn_db_path}")
        if self._asn_reader is None:
            self._asn_reader = geoip2.database.Reader(self.asn_db_path)
        return self._asn_reader

    def get_country_reader(self) -> geoip2.database.Reader:
        if not Path(self.country_db_path).exists():
            raise FileNotFoundError(f"Country database not found at {self.country_db_path}")
        if self._country_reader is None:
            self._country_reader = geoip2.database.Reader(self.country_db_path)
        return self._country_reader

STATE = ServerState()

# -------------------------
# Utilities
# -------------------------

def validate_ip(ip_str: str) -> bool:
    try:
        ipaddress.ip_address(ip_str)
        return True
    except ValueError:
        return False

def is_private_ip(ip_str: str) -> bool:
    try:
        return ipaddress.ip_address(ip_str).is_private
    except ValueError:
        return False

def get_ip_type(ip_str: str) -> str:
    try:
        ip = ipaddress.ip_address(ip_str)
        if ip.is_private: return "private"
        if ip.is_loopback: return "loopback"
        if ip.is_multicast: return "multicast"
        if ip.is_reserved: return "reserved"
        return "public"
    except ValueError:
        return "invalid"

def haversine(lat1: float, lon1: float, lat2: float, lon2: float, unit: str = "km") -> float:
    lat1_rad = math.radians(lat1); lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2); lon2_rad = math.radians(lon2)
    dlat = lat2_rad - lat1_rad; dlon = lon2_rad - lon1_rad
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance_km = 6371.0 * c
    if unit.lower() in ("mi", "miles"):
        return distance_km * 0.621371
    return distance_km

def to_summary(item: Dict[str, Any]) -> str:
    ip = item.get("ip_address", "Unknown")
    summary = [f"IP Address: {ip}"]
    loc = item.get("location", {})
    if loc and not loc.get("error"):
        country = loc.get("country", {}).get("name", "Unknown")
        city = loc.get("city", {}).get("name", "Unknown")
        if city != "Unknown" and country != "Unknown":
            summary.append(f"Location: {city}, {country}")
        elif country != "Unknown":
            summary.append(f"Country: {country}")
        coords = loc.get("location", {})
        if coords.get("latitude") is not None and coords.get("longitude") is not None:
            summary.append(f"Coordinates: {coords['latitude']}, {coords['longitude']}")
        traits = loc.get("traits", {})
        if traits.get("ip_type"):
            summary.append(f"IP Type: {traits['ip_type']}")
    asn = item.get("asn", {})
    if asn and not asn.get("error"):
        num = asn.get("autonomous_system_number")
        org = asn.get("autonomous_system_organization")
        if num and org:
            summary.append(f"ASN: AS{num} ({org})")
    return "\n".join(summary)

def to_csv(data: Any) -> str:
    items = data if isinstance(data, list) else [data]
    if not items:
        return ""
    headers = ["ip_address"]
    first = items[0]
    if "location" in first and not first["location"].get("error"):
        headers.extend(["country", "city", "latitude", "longitude"])
    if "asn" in first and not first["asn"].get("error"):
        headers.extend(["asn_number", "asn_organization"])
    lines = [",".join(headers)]
    for item in items:
        row = [item.get("ip_address", "")]
        if "location" in item and not item["location"].get("error"):
            loc = item["location"]
            row.extend([
                loc.get("country", {}).get("name", ""),
                loc.get("city", {}).get("name", ""),
                str(loc.get("location", {}).get("latitude", "")),
                str(loc.get("location", {}).get("longitude", "")),
            ])
        if "asn" in item and not item["asn"].get("error"):
            asn = item["asn"]
            row.extend([
                str(asn.get("autonomous_system_number", "")),
                asn.get("autonomous_system_organization", ""),
            ])
        lines.append(",".join(f'"{field}"' for field in row))
    return "\n".join(lines)

def format_output(data: Any, format_type: str) -> str:
    if format_type == "json":
        return json.dumps(data, indent=2)
    if format_type == "summary":
        if isinstance(data, list):
            return "\n\n".join(to_summary(item) for item in data)
        return to_summary(data)
    if format_type == "csv":
        return to_csv(data)
    return json.dumps(data, indent=2)

# -------------------------
# Database accessors
# -------------------------

def get_city_info(ip_addr: str) -> Dict[str, Any]:
    reader = STATE.get_city_reader()
    with LOOKUP_LATENCY.labels(kind="city").time():
        try:
            response = reader.city(ip_addr)
            return {
                "country": {
                    "iso_code": response.country.iso_code,
                    "name": response.country.name,
                    "names": dict(response.country.names) if response.country.names else {},
                },
                "subdivisions": {
                    "most_specific": {
                        "name": response.subdivisions.most_specific.name,
                        "iso_code": response.subdivisions.most_specific.iso_code,
                    }
                },
                "city": {"name": response.city.name},
                "postal": {"code": response.postal.code},
                "location": {
                    "latitude": float(response.location.latitude) if response.location.latitude else None,
                    "longitude": float(response.location.longitude) if response.location.longitude else None,
                    "accuracy_radius": response.location.accuracy_radius,
                    "time_zone": response.location.time_zone,
                },
                "traits": {
                    "network": str(response.traits.network) if response.traits.network else None,
                    "ip_type": get_ip_type(ip_addr),
                    "is_private": is_private_ip(ip_addr),
                },
            }
        except geoip2.errors.AddressNotFoundError:
            return {"error": f"No city information found for IP {ip_addr}"}

def get_asn_info(ip_addr: str) -> Dict[str, Any]:
    reader = STATE.get_asn_reader()
    with LOOKUP_LATENCY.labels(kind="asn").time():
        try:
            response = reader.asn(ip_addr)
            return {
                "autonomous_system_number": response.autonomous_system_number,
                "autonomous_system_organization": response.autonomous_system_organization,
                "ip_address": str(response.ip_address),
                "network": str(response.network) if response.network else None,
            }
        except geoip2.errors.AddressNotFoundError:
            return {"error": f"No ASN information found for IP {ip_addr}"}

def get_country_info(ip_addr: str) -> Dict[str, Any]:
    reader = STATE.get_country_reader()
    with LOOKUP_LATENCY.labels(kind="country").time():
        try:
            response = reader.country(ip_addr)
            return {
                "country": {
                    "iso_code": response.country.iso_code,
                    "name": response.country.name,
                    "names": dict(response.country.names) if response.country.names else {},
                },
                "traits": {
                    "ip_type": get_ip_type(ip_addr),
                    "is_private": is_private_ip(ip_addr),
                }
            }
        except geoip2.errors.AddressNotFoundError:
            return {"error": f"No country information found for IP {ip_addr}"}

# -------------------------
# FastMCP app
# -------------------------

app = FastMCP(
    name="enhanced-geoip-mcp-server",
    version="1.2.0",
)

# Resources
@app.resource("geoip://server/status", name="Server Status", description="Current server status and statistics", mime_type="application/json")
def server_status() -> str:
    uptime = (datetime.now() - STATE.start_time).total_seconds()
    SERVER_UPTIME.set(uptime)
    data = {
        "status": "running",
        "uptime_seconds": int(uptime),
        "requests_processed": STATE.request_count,
        "databases_loaded": sum(1 for db in STATE.db_info.values() if db.get("available")),
        "cache_enabled": True,
        "version": app.version,
        "concurrency": STATE.batch_concurrency,
    }
    return json.dumps(data, indent=2)

@app.resource("geoip://databases/info", name="Database Information", description="Information about loaded GeoIP databases", mime_type="application/json")
def database_info() -> str:
    return json.dumps(STATE.db_info, indent=2)

@app.resource("geoip://cache/stats", name="Cache Statistics", description="Cache performance statistics", mime_type="application/json")
def cache_stats() -> str:
    return json.dumps(STATE.cache.stats(), indent=2)

@app.resource("geoip://metrics", name="Server Metrics", description="Internal server metrics snapshot (counters/histograms may be partial)", mime_type="application/json")
def metrics_snapshot() -> str:
    uptime = (datetime.now() - STATE.start_time).total_seconds()
    snapshot = {
        "uptime_seconds": int(uptime),
        "requests_processed": STATE.request_count,
        "cache": STATE.cache.stats(),
        "concurrency": STATE.batch_concurrency,
        "prometheus_enabled": bool(os.getenv("PROMETHEUS_PORT")),
    }
    return json.dumps(snapshot, indent=2)

# -------------------------
# Tool input models
# -------------------------

# Single IP geolocation input
class GeolocateIPInput(BaseModel):
    ip_address: str = Field(..., description="IPv4 or IPv6 address")
    include_asn: bool = True
    output_format: Literal["json", "summary", "csv"] = "json"
    use_cache: bool = True

    @field_validator("ip_address")
    @classmethod
    def _valid_ip(cls, v: str) -> str:
        if not validate_ip(v):
            raise ValueError(f"Invalid IP address format: {v}")
        return v

# Multiple IPs geolocation input (batch)
class GeolocateMultipleIPsInput(BaseModel):
    ip_addresses: List[str] = Field(..., min_length=1, max_length=100)
    include_asn: bool = True
    output_format: Literal["json", "summary", "csv"] = "json"
    use_cache: bool = True

# ASN lookup
class GetASNInput(BaseModel):
    ip_address: str

    @field_validator("ip_address")
    @classmethod
    def _valid_ip(cls, v: str) -> str:
        if not validate_ip(v):
            raise ValueError(f"Invalid IP address format: {v}")
        return v

# Distance calc
class DistanceInput(BaseModel):
    lat1: float
    lon1: float
    lat2: float
    lon2: float
    unit: Literal["km", "mi"] = "km"

# Server management
class ServerManagementInput(BaseModel):
    action: Literal["clear_cache", "get_stats", "reload_databases"]

# Country-only lookup
class GeolocateCountryInput(BaseModel):
    ip_address: str
    output_format: Literal["json", "summary", "csv"] = "json"
    use_cache: bool = True

    @field_validator("ip_address")
    @classmethod
    def _valid_ip(cls, v: str) -> str:
        if not validate_ip(v):
            raise ValueError(f"Invalid IP address format: {v}")
        return v


# -------------------------
# Tools
# -------------------------

@app.tool("geolocate_ip", description="Get comprehensive geolocation information for a single IP address")
def geolocate_ip(payload: GeolocateIPInput) -> str:
    REQUESTS_TOTAL.labels(tool="geolocate_ip").inc()
    STATE.request_count += 1

    cache_key = f"{payload.ip_address}:{payload.include_asn}"
    cached = STATE.cache.get(cache_key) if payload.use_cache else None
    if cached:
        CACHE_HITS.labels(kind="single").inc()
        return format_output(cached, payload.output_format)
    else:
        if payload.use_cache:
            CACHE_MISSES.labels(kind="single").inc()

    result: Dict[str, Any] = {"ip_address": payload.ip_address}
    try:
        result["location"] = get_city_info(payload.ip_address)
    except Exception as e:
        result["location"] = {"error": str(e)}
        REQUEST_ERRORS_TOTAL.labels(tool="geolocate_ip").inc()

    if payload.include_asn:
        try:
            result["asn"] = get_asn_info(payload.ip_address)
        except Exception as e:
            result["asn"] = {"error": str(e)}
            REQUEST_ERRORS_TOTAL.labels(tool="geolocate_ip").inc()

    if payload.use_cache:
        STATE.cache.set(cache_key, result)

    return format_output(result, payload.output_format)

@app.tool("geolocate_multiple_ips", description="Get geolocation information for multiple IP addresses with batch concurrency")
def geolocate_multiple_ips(payload: GeolocateMultipleIPsInput) -> str:
    REQUESTS_TOTAL.labels(tool="geolocate_multiple_ips").inc()
    STATE.request_count += 1
    BATCH_SIZE.observe(len(payload.ip_addresses))

    # Concurrency limiter
    sem = asyncio.Semaphore(STATE.batch_concurrency)

    async def process_ip(ip: str) -> Dict[str, Any]:
        if not validate_ip(ip):
            return {"ip_address": ip, "error": "Invalid IP address format"}

        cache_key = f"{ip}:{payload.include_asn}"
        cached = STATE.cache.get(cache_key) if payload.use_cache else None
        if cached:
            CACHE_HITS.labels(kind="batch").inc()
            return cached
        else:
            if payload.use_cache:
                CACHE_MISSES.labels(kind="batch").inc()

        async with sem:
            # Run potentially blocking DB calls in a thread
            item: Dict[str, Any] = {"ip_address": ip}
            try:
                item["location"] = await asyncio.to_thread(get_city_info, ip)
            except Exception as e:
                item["location"] = {"error": str(e)}
                REQUEST_ERRORS_TOTAL.labels(tool="geolocate_multiple_ips").inc()
            if payload.include_asn:
                try:
                    item["asn"] = await asyncio.to_thread(get_asn_info, ip)
                except Exception as e:
                    item["asn"] = {"error": str(e)}
                    REQUEST_ERRORS_TOTAL.labels(tool="geolocate_multiple_ips").inc()
            if payload.use_cache:
                STATE.cache.set(cache_key, item)
            return item

    # Run all in parallel with gather
    loop = asyncio.get_event_loop()
    results: List[Dict[str, Any]] = loop.run_until_complete(
        asyncio.gather(*(process_ip(ip) for ip in payload.ip_addresses))
    )

    return format_output(results, payload.output_format)

@app.tool("get_asn_info", description="Get ASN (Autonomous System Number) information for an IP address")
def get_asn_info_tool(payload: GetASNInput) -> str:
    REQUESTS_TOTAL.labels(tool="get_asn_info").inc()
    STATE.request_count += 1
    try:
        data = get_asn_info(payload.ip_address)
        return json.dumps({"ip_address": payload.ip_address, "asn": data}, indent=2)
    except Exception as e:
        REQUEST_ERRORS_TOTAL.labels(tool="get_asn_info").inc()
        return json.dumps({"error": str(e)}, indent=2)

@app.tool("calculate_distance", description="Calculate distance between two geographic coordinates")
def calculate_distance(payload: DistanceInput) -> str:
    REQUESTS_TOTAL.labels(tool="calculate_distance").inc()
    STATE.request_count += 1
    try:
        distance = haversine(payload.lat1, payload.lon1, payload.lat2, payload.lon2, payload.unit)
        result = {
            "point1": {"latitude": payload.lat1, "longitude": payload.lon1},
            "point2": {"latitude": payload.lat2, "longitude": payload.lon2},
            "distance": round(distance, 2),
            "unit": payload.unit,
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        REQUEST_ERRORS_TOTAL.labels(tool="calculate_distance").inc()
        return json.dumps({"error": str(e)}, indent=2)

@app.tool("server_management", description="Server management operations (clear cache, get stats, reload databases)")
def server_management(payload: ServerManagementInput) -> str:
    REQUESTS_TOTAL.labels(tool="server_management").inc()
    STATE.request_count += 1
    if payload.action == "clear_cache":
        STATE.cache.clear()
        return "Cache cleared successfully"
    if payload.action == "get_stats":
        uptime = datetime.now() - STATE.start_time
        stats = {
            "server": {
                "uptime_seconds": int(uptime.total_seconds()),
                "requests_processed": STATE.request_count,
                "start_time": STATE.start_time.isoformat(),
                "concurrency": STATE.batch_concurrency,
            },
            "cache": STATE.cache.stats(),
            "databases": STATE.db_info,
        }
        return json.dumps(stats, indent=2)
    if payload.action == "reload_databases":
        STATE.close_readers()
        STATE.validate_db_paths()
        return "Database information reloaded"
    return json.dumps({"error": f"Unknown action: {payload.action}"}, indent=2)

@app.tool("geolocate_country", description="Get country-only geolocation info for a single IP address")
def geolocate_country(payload: GeolocateCountryInput) -> str:
    REQUESTS_TOTAL.labels(tool="geolocate_country").inc()
    STATE.request_count += 1

    cache_key = f"{payload.ip_address}:country"
    cached = STATE.cache.get(cache_key) if payload.use_cache else None
    if cached:
        CACHE_HITS.labels(kind="single").inc()
        return format_output(cached, payload.output_format)
    else:
        if payload.use_cache:
            CACHE_MISSES.labels(kind="single").inc()

    result: Dict[str, Any] = {"ip_address": payload.ip_address}
    try:
        result["location"] = get_country_info(payload.ip_address)
    except Exception as e:
        result["location"] = {"error": str(e)}
        REQUEST_ERRORS_TOTAL.labels(tool="geolocate_country").inc()

    if payload.use_cache:
        STATE.cache.set(cache_key, result)

    return format_output(result, payload.output_format)

# -------------------------
# Entrypoint
# -------------------------

# Replace your current main() and __main__ section with:

def main():
    """Synchronous main function for MCP compatibility"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Start Prometheus exporter if PROMETHEUS_PORT is set
    try:
        maybe_start_prometheus()
    except NameError:
        pass
    
    # Run synchronously
    app.run()  # Use sync run instead of run_async()

if __name__ == "__main__":
    main()  # Remove asyncio.run()
