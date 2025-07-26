#!/usr/bin/env python3
"""
Enhanced GeoIP MCP Server with distance calculation and all features
"""

import asyncio
import json
import logging
import os
import sys
import time
import math
from typing import Any, Dict, List, Optional, Tuple
import ipaddress
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path

import geoip2.database
import geoip2.errors
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("geoip-mcp-server")

@dataclass
class CacheEntry:
    """Cache entry for storing lookup results"""
    data: Dict[str, Any]
    timestamp: datetime
    ttl: int = 3600  # 1 hour default TTL

class GeoIPCache:
    """Simple in-memory cache for GeoIP lookups"""
    
    def __init__(self, default_ttl: int = 3600):
        self.cache: Dict[str, CacheEntry] = {}
        self.default_ttl = default_ttl
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached entry if not expired"""
        if key in self.cache:
            entry = self.cache[key]
            if datetime.now() - entry.timestamp < timedelta(seconds=entry.ttl):
                self.hits += 1
                return entry.data
            else:
                # Remove expired entry
                del self.cache[key]
        
        self.misses += 1
        return None
    
    def set(self, key: str, data: Dict[str, Any], ttl: Optional[int] = None):
        """Set cache entry"""
        ttl = ttl or self.default_ttl
        self.cache[key] = CacheEntry(data=data, timestamp=datetime.now(), ttl=ttl)
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "entries": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.2f}%"
        }

class EnhancedGeoIPServer:
    def __init__(self):
        self.server = Server("enhanced-geoip-mcp-server")
        
        # Fix path expansion issue
        self.city_db_path = os.path.expanduser(os.getenv("GEOIP_CITY_DB", "~/Downloads/GeoLite2-City.mmdb"))
        self.asn_db_path = os.path.expanduser(os.getenv("GEOIP_ASN_DB", "~/Downloads/GeoLite2-ASN.mmdb"))
        self.country_db_path = os.path.expanduser(os.getenv("GEOIP_COUNTRY_DB", "~/Downloads/GeoLite2-Country.mmdb"))
        
        # Initialize cache
        cache_ttl = int(os.getenv("GEOIP_CACHE_TTL", "3600"))
        self.cache = GeoIPCache(default_ttl=cache_ttl)
        
        # Performance tracking
        self.request_count = 0
        self.start_time = datetime.now()
        
        # Validate database paths
        self._validate_db_paths()
        
        # Setup handlers
        self._setup_handlers()

    def _validate_db_paths(self):
        """Validate that database files exist and get their info"""
        self.db_info = {}
        
        for db_name, db_path in [
            ("city", self.city_db_path),
            ("asn", self.asn_db_path),
            ("country", self.country_db_path)
        ]:
            if os.path.exists(db_path):
                try:
                    stat = os.stat(db_path)
                    self.db_info[db_name] = {
                        "path": db_path,
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "available": True
                    }
                    logger.info(f"{db_name.capitalize()} database found: {db_path}")
                except Exception as e:
                    logger.error(f"Error accessing {db_name} database: {e}")
                    self.db_info[db_name] = {"available": False, "error": str(e)}
            else:
                logger.warning(f"{db_name.capitalize()} database not found at {db_path}")
                self.db_info[db_name] = {"available": False, "error": "File not found"}

    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float, unit: str = "km") -> float:
        """
        Calculate the great circle distance between two points on Earth
        using the Haversine formula.
        
        Args:
            lat1, lon1: Latitude and longitude of first point in decimal degrees
            lat2, lon2: Latitude and longitude of second point in decimal degrees
            unit: Unit for distance ('km' for kilometers, 'mi' for miles)
            
        Returns:
            Distance between the two points in the specified unit
        """
        # Convert decimal degrees to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (math.sin(dlat / 2) ** 2 + 
            math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        # Earth's radius in kilometers
        earth_radius_km = 6371.0
        
        # Calculate distance in kilometers
        distance_km = earth_radius_km * c
        
        # Convert to requested unit
        if unit.lower() == "mi" or unit.lower() == "miles":
            # Convert km to miles: 1 km = 0.621371 miles
            return distance_km * 0.621371
        else:
            return distance_km




    def _is_private_ip(self, ip_str: str) -> bool:
        """Check if an IP address is private/internal"""
        try:
            ip = ipaddress.ip_address(ip_str)
            return ip.is_private
        except ValueError:
            return False

    def _get_ip_type(self, ip_str: str) -> str:
        """Get the type of IP address"""
        try:
            ip = ipaddress.ip_address(ip_str)
            if ip.is_private:
                return "private"
            elif ip.is_loopback:
                return "loopback"
            elif ip.is_multicast:
                return "multicast"
            elif ip.is_reserved:
                return "reserved"
            else:
                return "public"
        except ValueError:
            return "invalid"

    def _setup_handlers(self):
        """Setup MCP server handlers"""
        
        @self.server.list_resources()
        async def handle_list_resources() -> List[Resource]:
            """List available resources"""
            return [
                Resource(
                    uri="geoip://server/status",
                    name="Server Status",
                    description="Current server status and statistics",
                    mimeType="application/json"
                ),
                Resource(
                    uri="geoip://databases/info",
                    name="Database Information",
                    description="Information about loaded GeoIP databases",
                    mimeType="application/json"
                ),
                Resource(
                    uri="geoip://cache/stats",
                    name="Cache Statistics",
                    description="Cache performance statistics",
                    mimeType="application/json"
                )
            ]

        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            """Handle resource reading"""
            if uri == "geoip://server/status":
                uptime = datetime.now() - self.start_time
                return json.dumps({
                    "status": "running",
                    "uptime_seconds": int(uptime.total_seconds()),
                    "requests_processed": self.request_count,
                    "databases_loaded": sum(1 for db in self.db_info.values() if db.get("available")),
                    "cache_enabled": True,
                    "version": "1.1.0"
                }, indent=2)
            
            elif uri == "geoip://databases/info":
                return json.dumps(self.db_info, indent=2)
            
            elif uri == "geoip://cache/stats":
                return json.dumps(self.cache.stats(), indent=2)
            
            else:
                raise ValueError(f"Unknown resource: {uri}")

        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available tools"""
            return [
                Tool(
                    name="geolocate_ip",
                    description="Get comprehensive geolocation information for a single IP address",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "ip_address": {
                                "type": "string",
                                "description": "IP address to geolocate (IPv4 or IPv6)"
                            },
                            "include_asn": {
                                "type": "boolean",
                                "description": "Include ASN information",
                                "default": True
                            },
                            "output_format": {
                                "type": "string",
                                "enum": ["json", "summary", "csv"],
                                "description": "Output format",
                                "default": "json"
                            },
                            "use_cache": {
                                "type": "boolean",
                                "description": "Use cached results if available",
                                "default": True
                            }
                        },
                        "required": ["ip_address"]
                    }
                ),
                Tool(
                    name="geolocate_multiple_ips",
                    description="Get geolocation information for multiple IP addresses with batch processing",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "ip_addresses": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of IP addresses to geolocate",
                                "maxItems": 100
                            },
                            "include_asn": {
                                "type": "boolean",
                                "description": "Include ASN information",
                                "default": True
                            },
                            "output_format": {
                                "type": "string",
                                "enum": ["json", "csv", "summary"],
                                "description": "Output format",
                                "default": "json"
                            },
                            "use_cache": {
                                "type": "boolean",
                                "description": "Use cached results if available",
                                "default": True
                            }
                        },
                        "required": ["ip_addresses"]
                    }
                ),
                Tool(
                    name="get_asn_info",
                    description="Get ASN (Autonomous System Number) information for an IP address",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "ip_address": {
                                "type": "string",
                                "description": "IP address to get ASN information for"
                            }
                        },
                        "required": ["ip_address"]
                    }
                ),
                Tool(
                    name="calculate_distance",
                    description="Calculate distance between two geographic coordinates",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "lat1": {
                                "type": "number",
                                "description": "Latitude of first point"
                            },
                            "lon1": {
                                "type": "number", 
                                "description": "Longitude of first point"
                            },
                            "lat2": {
                                "type": "number",
                                "description": "Latitude of second point"
                            },
                            "lon2": {
                                "type": "number",
                                "description": "Longitude of second point"
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["km", "mi"],
                                "description": "Unit for distance (km or mi)",
                                "default": "km"
                            }
                        },
                        "required": ["lat1", "lon1", "lat2", "lon2"]
                    }
                ),
                Tool(
                    name="server_management",
                    description="Server management operations (clear cache, get stats, etc.)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["clear_cache", "get_stats", "reload_databases"],
                                "description": "Management action to perform"
                            }
                        },
                        "required": ["action"]
                    }
                )
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls"""
            self.request_count += 1
            
            try:
                if name == "geolocate_ip":
                    return await self._geolocate_single_ip(arguments)
                elif name == "geolocate_multiple_ips":
                    return await self._geolocate_multiple_ips(arguments)
                elif name == "get_asn_info":
                    return await self._get_asn_info_tool(arguments)
                elif name == "calculate_distance":
                    return await self._calculate_distance_tool(arguments)
                elif name == "server_management":
                    return await self._server_management(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
            except Exception as e:
                logger.error(f"Error in tool {name}: {str(e)}")
                return [TextContent(type="text", text=f"Error: {str(e)}")]

    def _validate_ip(self, ip_str: str) -> bool:
        """Validate IP address format"""
        try:
            ipaddress.ip_address(ip_str)
            return True
        except ValueError:
            return False

    def _get_city_info(self, ip_address: str) -> Dict[str, Any]:
        """Get city/location information for an IP address"""
        if not os.path.exists(self.city_db_path):
            raise FileNotFoundError(f"City database not found at {self.city_db_path}")
            
        with geoip2.database.Reader(self.city_db_path) as reader:
            try:
                response = reader.city(ip_address)
                return {
                    "country": {
                        "iso_code": response.country.iso_code,
                        "name": response.country.name,
                        "names": dict(response.country.names) if response.country.names else {}
                    },
                    "subdivisions": {
                        "most_specific": {
                            "name": response.subdivisions.most_specific.name,
                            "iso_code": response.subdivisions.most_specific.iso_code
                        }
                    },
                    "city": {
                        "name": response.city.name
                    },
                    "postal": {
                        "code": response.postal.code
                    },
                    "location": {
                        "latitude": float(response.location.latitude) if response.location.latitude else None,
                        "longitude": float(response.location.longitude) if response.location.longitude else None,
                        "accuracy_radius": response.location.accuracy_radius,
                        "time_zone": response.location.time_zone
                    },
                    "traits": {
                        "network": str(response.traits.network) if response.traits.network else None,
                        "ip_type": self._get_ip_type(ip_address),
                        "is_private": self._is_private_ip(ip_address)
                    }
                }
            except geoip2.errors.AddressNotFoundError:
                return {"error": f"No city information found for IP {ip_address}"}

    def _get_asn_info(self, ip_address: str) -> Dict[str, Any]:
        """Get ASN information for an IP address"""
        if not os.path.exists(self.asn_db_path):
            raise FileNotFoundError(f"ASN database not found at {self.asn_db_path}")
            
        with geoip2.database.Reader(self.asn_db_path) as reader:
            try:
                response = reader.asn(ip_address)
                return {
                    "autonomous_system_number": response.autonomous_system_number,
                    "autonomous_system_organization": response.autonomous_system_organization,
                    "ip_address": str(response.ip_address),
                    "network": str(response.network) if response.network else None
                }
            except geoip2.errors.AddressNotFoundError:
                return {"error": f"No ASN information found for IP {ip_address}"}

    def _format_output(self, data: Any, format_type: str) -> str:
        """Format output based on requested format"""
        if format_type == "json":
            return json.dumps(data, indent=2)
        elif format_type == "summary":
            return self._to_summary(data)
        elif format_type == "csv":
            return self._to_csv(data)
        else:
            return json.dumps(data, indent=2)

    def _to_csv(self, data: Any) -> str:
        """Convert data to CSV format"""
        if isinstance(data, list):
            if not data:
                return ""
            
            # Get headers from first item
            headers = ["ip_address"]
            first_item = data[0]
            
            if "location" in first_item and not first_item["location"].get("error"):
                headers.extend(["country", "city", "latitude", "longitude"])
            if "asn" in first_item and not first_item["asn"].get("error"):
                headers.extend(["asn_number", "asn_organization"])
            
            lines = [",".join(headers)]
            
            for item in data:
                row = [item.get("ip_address", "")]
                
                if "location" in item and not item["location"].get("error"):
                    loc = item["location"]
                    row.extend([
                        loc.get("country", {}).get("name", ""),
                        loc.get("city", {}).get("name", ""),
                        str(loc.get("location", {}).get("latitude", "")),
                        str(loc.get("location", {}).get("longitude", ""))
                    ])
                
                if "asn" in item and not item["asn"].get("error"):
                    asn = item["asn"]
                    row.extend([
                        str(asn.get("autonomous_system_number", "")),
                        asn.get("autonomous_system_organization", "")
                    ])
                
                lines.append(",".join(f'"{field}"' for field in row))
            
            return "\n".join(lines)
        else:
            # Single item
            return self._to_csv([data])

    def _to_summary(self, data: Any) -> str:
        """Convert data to human-readable summary"""
        if isinstance(data, list):
            summaries = []
            for item in data:
                summaries.append(self._single_item_summary(item))
            return "\n\n".join(summaries)
        else:
            return self._single_item_summary(data)

    def _single_item_summary(self, item: Dict) -> str:
        """Create summary for single item"""
        ip = item.get("ip_address", "Unknown")
        summary = [f"IP Address: {ip}"]
        
        if "location" in item and not item["location"].get("error"):
            loc = item["location"]
            country = loc.get("country", {}).get("name", "Unknown")
            city = loc.get("city", {}).get("name", "Unknown")
            
            if city != "Unknown" and country != "Unknown":
                summary.append(f"Location: {city}, {country}")
            elif country != "Unknown":
                summary.append(f"Country: {country}")
            
            coords = loc.get("location", {})
            if coords.get("latitude") and coords.get("longitude"):
                summary.append(f"Coordinates: {coords['latitude']}, {coords['longitude']}")
            
            traits = loc.get("traits", {})
            if traits.get("ip_type"):
                summary.append(f"IP Type: {traits['ip_type']}")
        
        if "asn" in item and not item["asn"].get("error"):
            asn = item["asn"]
            asn_num = asn.get("autonomous_system_number")
            asn_org = asn.get("autonomous_system_organization")
            if asn_num and asn_org:
                summary.append(f"ASN: AS{asn_num} ({asn_org})")
        
        return "\n".join(summary)

    async def _geolocate_single_ip(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle single IP geolocation"""
        ip_address = arguments.get("ip_address")
        include_asn = arguments.get("include_asn", True)
        output_format = arguments.get("output_format", "json")
        use_cache = arguments.get("use_cache", True)
        
        if not ip_address:
            return [TextContent(type="text", text="Error: IP address is required")]
        
        if not self._validate_ip(ip_address):
            return [TextContent(type="text", text=f"Error: Invalid IP address format: {ip_address}")]
        
        # Check cache first
        cache_key = f"{ip_address}:{include_asn}"
        if use_cache:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                formatted_output = self._format_output(cached_result, output_format)
                return [TextContent(type="text", text=formatted_output)]
        
        result = {"ip_address": ip_address}
        
        # Get city/location information
        try:
            city_info = self._get_city_info(ip_address)
            result["location"] = city_info
        except Exception as e:
            result["location"] = {"error": str(e)}
        
        # Get ASN information if requested
        if include_asn:
            try:
                asn_info = self._get_asn_info(ip_address)
                result["asn"] = asn_info
            except Exception as e:
                result["asn"] = {"error": str(e)}
        
        # Cache the result
        if use_cache:
            self.cache.set(cache_key, result)
        
        formatted_output = self._format_output(result, output_format)
        return [TextContent(type="text", text=formatted_output)]

    async def _geolocate_multiple_ips(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle multiple IP geolocation"""
        ip_addresses = arguments.get("ip_addresses", [])
        include_asn = arguments.get("include_asn", True)
        output_format = arguments.get("output_format", "json")
        use_cache = arguments.get("use_cache", True)
        
        if not ip_addresses:
            return [TextContent(type="text", text="Error: IP addresses list is required")]
        
        results = []
        
        for ip_address in ip_addresses:
            if not self._validate_ip(ip_address):
                results.append({
                    "ip_address": ip_address,
                    "error": "Invalid IP address format"
                })
                continue
            
            # Check cache first
            cache_key = f"{ip_address}:{include_asn}"
            if use_cache:
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    results.append(cached_result)
                    continue
            
            result = {"ip_address": ip_address}
            
            # Get city/location information
            try:
                city_info = self._get_city_info(ip_address)
                result["location"] = city_info
            except Exception as e:
                result["location"] = {"error": str(e)}
            
            # Get ASN information if requested
            if include_asn:
                try:
                    asn_info = self._get_asn_info(ip_address)
                    result["asn"] = asn_info
                except Exception as e:
                    result["asn"] = {"error": str(e)}
            
            # Cache the result
            if use_cache:
                self.cache.set(cache_key, result)
            
            results.append(result)
        
        formatted_output = self._format_output(results, output_format)
        return [TextContent(type="text", text=formatted_output)]

    async def _get_asn_info_tool(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle ASN information request"""
        ip_address = arguments.get("ip_address")
        
        if not ip_address:
            return [TextContent(type="text", text="Error: IP address is required")]
        
        if not self._validate_ip(ip_address):
            return [TextContent(type="text", text=f"Error: Invalid IP address format: {ip_address}")]
        
        try:
            asn_info = self._get_asn_info(ip_address)
            result = {
                "ip_address": ip_address,
                "asn": asn_info
            }
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    async def _calculate_distance_tool(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle distance calculation request"""
        try:
            lat1 = float(arguments.get("lat1"))
            lon1 = float(arguments.get("lon1"))
            lat2 = float(arguments.get("lat2"))
            lon2 = float(arguments.get("lon2"))
            unit = arguments.get("unit", "km")
            
            distance = self._calculate_distance(lat1, lon1, lat2, lon2, unit)
            
            result = {
                "point1": {"latitude": lat1, "longitude": lon1},
                "point2": {"latitude": lat2, "longitude": lon2},
                "distance": round(distance, 2),
                "unit": unit
            }
            
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
        except (ValueError, TypeError) as e:
            return [TextContent(type="text", text=f"Error: Invalid coordinates - {str(e)}")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    async def _server_management(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle server management operations"""
        action = arguments.get("action")
        
        if action == "clear_cache":
            self.cache.clear()
            return [TextContent(type="text", text="Cache cleared successfully")]
        
        elif action == "get_stats":
            uptime = datetime.now() - self.start_time
            stats = {
                "server": {
                    "uptime_seconds": int(uptime.total_seconds()),
                    "requests_processed": self.request_count,
                    "start_time": self.start_time.isoformat()
                },
                "cache": self.cache.stats(),
                "databases": self.db_info
            }
            return [TextContent(type="text", text=json.dumps(stats, indent=2))]
        
        elif action == "reload_databases":
            self._validate_db_paths()
            return [TextContent(type="text", text="Database information reloaded")]
        
        else:
            return [TextContent(type="text", text=f"Error: Unknown action: {action}")]

    async def run(self):
        """Run the enhanced server"""
        logger.info("Starting Enhanced GeoIP MCP Server...")
        async with stdio_server() as (read_stream, write_stream):
            # Create proper initialization options
            init_options = InitializationOptions(
                server_name="enhanced-geoip-mcp-server",
                server_version="1.1.0",
                capabilities={}  # Use empty dict instead of calling get_capabilities
            )
            
            await self.server.run(
                read_stream,
                write_stream,
                init_options
            )

async def main():
    """Main entry point"""
    server = EnhancedGeoIPServer()
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())
