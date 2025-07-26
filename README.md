# GeoIP MCP Server

An MCP (Model Context Protocol) server that provides IP geolocation services using MaxMind's GeoIP2 databases.

## Features

- Single IP geolocation
- Bulk IP geolocation
- ASN (Autonomous System Number) information
- Support for both IPv4 and IPv6 addresses
- City, country, subdivision, and postal code information
- Latitude/longitude coordinates
- Network information

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download MaxMind GeoIP2 databases:

Download GeoLite2-City.mmdb and GeoLite2-ASN.mmdb from MaxMind
You can get free databases by signing up at https://www.maxmind.com/
Set environment variables:

```bash
export GEOIP_CITY_DB="~/Downloads/GeoLite2-City.mmdb"
export GEOIP_ASN_DB="~/Downloads/GeoLite2-ASN.mmdb"
```

## Usage
Run the server:

```bash
python server.py
```

## Available Tools
- geolocate_ip
Get geolocation information for a single IP address.

Parameters:

- ip_address (required): IP address to geolocate
- include_asn (optional): Include ASN information (default: true)
- geolocate_multiple_ips

Get geolocation information for multiple IP addresses.

Parameters:

ip_addresses (required): Array of IP addresses to geolocate
include_asn (optional): Include ASN information (default: true)
get_asn_info
Get ASN information for an IP address.

Parameters:

ip_address (required): IP address to get ASN information for
Example Output
json


{
  "ip_address": "203.0.113.0",
  "location": {
    "country": {
      "iso_code": "US",
      "name": "United States",
      "names": {
        "en": "United States",
        "zh-CN": "美国"
      }
    },
    "subdivisions": {
      "most_specific": {
        "name": "Minnesota",
        "iso_code": "MN"
      }
    },
    "city": {
      "name": "Minneapolis"
    },
    "postal": {
      "code": "55455"
    },
    "location": {
      "latitude": 44.9733,
      "longitude": -93.2323,
      "accuracy_radius": 50,
      "time_zone": "America/Chicago"
    },
    "traits": {
      "network": "203.0.113.0/24"
    }
  },
  "asn": {
    "autonomous_system_number": 1221,
    "autonomous_system_organization": "Telstra Pty Ltd",
    "ip_address": "203.0.113.0",
    "network": "203.0.113.0/24"
  }
}
Error Handling
The server handles various error conditions:

Invalid IP address formats
Missing database files
IP addresses not found in databases
Network connectivity issues
License
This project is licensed under the MIT License.

code



## Installation and Setup Instructions

1. **Create the project directory:**
```bash
mkdir geoip-mcp-server
cd geoip-mcp-server
Create the files with the content provided above.

Install dependencies:

bash


pip install -r requirements.txt
Download GeoIP2 databases:

Sign up for a free MaxMind account at https://www.maxmind.com/
Download GeoLite2-City.mmdb and GeoLite2-ASN.mmdb
Place them in a known location
Set environment variables:

bash


export GEOIP_CITY_DB="~/Downloads/your/GeoLite2-City.mmdb"
export GEOIP_ASN_DB="~/Downloads/your/GeoLite2-ASN.mmdb"
Run the server:
bash


python server.py
Key Features
Comprehensive geolocation: Provides country, city, subdivision, postal code, and coordinates
ASN information: Includes autonomous system details
Bulk processing: Can handle multiple IPs in a single request
Error handling: Graceful handling of invalid IPs and missing data
Flexible output: JSON-formatted responses with detailed information
Environment configuration: Easy database path configuration via environment variables
The server follows MCP protocol standards and can be integrated with any MCP-compatible client to provide IP geolocation services.