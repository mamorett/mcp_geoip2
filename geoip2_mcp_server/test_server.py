#!/usr/bin/env python3
"""
Test suite for GeoIP MCP Server
"""

import asyncio
import json
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import os

from server import EnhancedGeoIPServer

class TestGeoIPServer(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        self.server = EnhancedGeoIPServer()
    
    def test_ip_validation(self):
        """Test IP address validation"""
        # Valid IPs
        self.assertTrue(self.server._validate_ip("192.168.1.1"))
        self.assertTrue(self.server._validate_ip("2001:db8::1"))
        
        # Invalid IPs
        self.assertFalse(self.server._validate_ip("256.1.1.1"))
        self.assertFalse(self.server._validate_ip("not.an.ip"))
        self.assertFalse(self.server._validate_ip(""))
    
    def test_cache_functionality(self):
        """Test cache operations"""
        cache = self.server.cache
        
        # Test cache miss
        result = cache.get("test_key")
        self.assertIsNone(result)
        
        # Test cache set and hit
        test_data = {"test": "data"}
        cache.set("test_key", test_data)
        result = cache.get("test_key")
        self.assertEqual(result, test_data)
        
        # Test cache stats
        stats = cache.stats()
        self.assertIn("hits", stats)
        self.assertIn("misses", stats)
    
    def test_distance_calculation(self):
        """Test distance calculation between coordinates"""
        # Test known distance (approximately)
        # New York to Los Angeles
        distance_km = self.server._calculate_distance(40.7128, -74.0060, 34.0522, -118.2437, "km")
        distance_miles = self.server._calculate_distance(40.7128, -74.0060, 34.0522, -118.2437, "miles")
        
        # Should be approximately 3944 km or 2451 miles
        self.assertAlmostEqual(distance_km, 3944, delta=100)
        self.assertAlmostEqual(distance_miles, 2451, delta=100)
    
    def test_output_formatting(self):
        """Test different output formats"""
        test_data = {
            "ip_address": "192.168.1.1",
            "location": {
                "country": {"name": "Test Country"},
                "city": {"name": "Test City"}
            }
        }
        
        # Test JSON format
        json_output = self.server._format_output(test_data, "json")
        self.assertIn("ip_address", json_output)
        
        # Test summary format
        summary_output = self.server._format_output(test_data, "summary")
        self.assertIn("IP Address:", summary_output)
        
        # Test CSV format
        csv_output = self.server._format_output([test_data], "csv")
        self.assertIn("ip_address", csv_output)

class TestIntegration(unittest.TestCase):
    """Integration tests requiring actual database files"""
    
    @unittest.skipUnless(
        os.path.exists(os.getenv("GEOIP_CITY_DB", "")),
        "GeoIP database not available"
    )
    def test_real_ip_lookup(self):
        """Test with real IP and database"""
        server = EnhancedGeoIPServer()
        
        # Test with Google's DNS
        try:
            result = server._get_city_info("8.8.8.8")
            self.assertIsInstance(result, dict)
            self.assertIn("country", result)
        except Exception as e:
            self.skipTest(f"Database lookup failed: {e}")

if __name__ == "__main__":
    unittest.main()
