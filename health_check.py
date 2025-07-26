#!/usr/bin/env python3
"""
Health check script for GeoIP MCP Server
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import subprocess
import requests
from typing import Dict, Any, List

class HealthChecker:
    """Comprehensive health checking for GeoIP MCP Server"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config = self._load_config(config_file)
        self.checks = []
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration file"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    
    def check_database_files(self) -> Dict[str, Any]:
        """Check if database files exist and are readable"""
        result = {
            "name": "Database Files",
            "status": "healthy",
            "details": {},
            "timestamp": datetime.now().isoformat()
        }
        
        db_paths = {
            "city": os.getenv("GEOIP_CITY_DB"),
            "asn": os.getenv("GEOIP_ASN_DB"),
            "country": os.getenv("GEOIP_COUNTRY_DB")
        }
        
        for db_name, db_path in db_paths.items():
            if not db_path:
                result["details"][db_name] = {"status": "not_configured"}
                continue
            
            path = Path(db_path)
            if not path.exists():
                result["details"][db_name] = {"status": "missing", "path": db_path}
                result["status"] = "unhealthy"
            elif not path.is_file():
                result["details"][db_name] = {"status": "not_file", "path": db_path}
                result["status"] = "unhealthy"
            elif not os.access(path, os.R_OK):
                result["details"][db_name] = {"status": "not_readable", "path": db_path}
                result["status"] = "unhealthy"
            else:
                stat = path.stat()
                result["details"][db_name] = {
                    "status": "healthy",
                    "path": db_path,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                }
        
        return result
    
    def check_database_functionality(self) -> Dict[str, Any]:
        """Test database functionality with sample queries"""
        result = {
            "name": "Database Functionality",
            "status": "healthy",
            "details": {},
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            import geoip2.database
            
            # Test city database
            city_db = os.getenv("GEOIP_CITY_DB")
            if city_db and os.path.exists(city_db):
                try:
                    with geoip2.database.Reader(city_db) as reader:
                        response = reader.city("8.8.8.8")
                        result["details"]["city"] = {
                            "status": "healthy",
                            "test_ip": "8.8.8.8",
                            "country": response.country.name
                        }
                except Exception as e:
                    result["details"]["city"] = {"status": "error", "error": str(e)}
                    result["status"] = "unhealthy"
            
            # Test ASN database
            asn_db = os.getenv("GEOIP_ASN_DB")
            if asn_db and os.path.exists(asn_db):
                try:
                    with geoip2.database.Reader(asn_db) as reader:
                        response = reader.asn("8.8.8.8")
                        result["details"]["asn"] = {
                            "status": "healthy",
                            "test_ip": "8.8.8.8",
                            "asn": response.autonomous_system_number
                        }
                except Exception as e:
                    result["details"]["asn"] = {"status": "error", "error": str(e)}
                    result["status"] = "unhealthy"
                    
        except ImportError:
            result["status"] = "unhealthy"
            result["details"]["error"] = "geoip2 library not available"
        
        return result
    
    def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        result = {
            "name": "System Resources",
            "status": "healthy",
            "details": {},
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            result["details"]["cpu"] = {
                "usage_percent": cpu_percent,
                "status": "healthy" if cpu_percent < 80 else "warning" if cpu_percent < 95 else "critical"
            }
            
            # Memory usage
            memory = psutil.virtual_memory()
            result["details"]["memory"] = {
                "usage_percent": memory.percent,
                "available_gb": round(memory.available / (1024**3), 2),
                "status": "healthy" if memory.percent < 80 else "warning" if memory.percent < 95 else "critical"
            }
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            result["details"]["disk"] = {
                "usage_percent": round(disk_percent, 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "status": "healthy" if disk_percent < 80 else "warning" if disk_percent < 95 else "critical"
            }
            
            # Check if any resource is in critical state
            if any(detail.get("status") == "critical" for detail in result["details"].values()):
                result["status"] = "unhealthy"
            elif any(detail.get("status") == "warning" for detail in result["details"].values()):
                result["status"] = "warning"
                
        except ImportError:
            result["status"] = "warning"
            result["details"]["error"] = "psutil library not available"
        
        return result
    
    def check_process_status(self) -> Dict[str, Any]:
        """Check if the server process is running"""
        result = {
            "name": "Process Status",
            "status": "healthy",
            "details": {},
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Check if process is running (simple approach)
            output = subprocess.check_output(["pgrep", "-f", "server.py"], text=True)
            pids = output.strip().split('\n')
            
            result["details"]["running"] = True
            result["details"]["process_count"] = len(pids)
            result["details"]["pids"] = pids
            
        except subprocess.CalledProcessError:
            result["status"] = "unhealthy"
            result["details"]["running"] = False
            result["details"]["error"] = "Server process not found"
        
        return result
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        checks = [
            self.check_database_files(),
            self.check_database_functionality(),
            self.check_system_resources(),
            self.check_process_status()
        ]
        
        # Determine overall status
        overall_status = "healthy"
        if any(check["status"] == "unhealthy" for check in checks):
            overall_status = "unhealthy"
        elif any(check["status"] == "warning" for check in checks):
            overall_status = "warning"
        
        return {
            "overall_status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "checks": checks
        }
    
    def run_check_and_exit(self):
        """Run health check and exit with appropriate code"""
        results = self.run_all_checks()
        
        print(json.dumps(results, indent=2))
        
        if results["overall_status"] == "healthy":
            sys.exit(0)
        elif results["overall_status"] == "warning":
            sys.exit(1)
        else:
            sys.exit(2)

def main():
    """Main entry point"""
    checker = HealthChecker()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--json":
        checker.run_check_and_exit()
    else:
        results = checker.run_all_checks()
        
        print(f"🏥 GeoIP MCP Server Health Check")
        print(f"Overall Status: {results['overall_status'].upper()}")
        print(f"Timestamp: {results['timestamp']}")
        print()
        
        for check in results['checks']:
            status_emoji = {
                'healthy': '✅',
                'warning': '⚠️',
                'unhealthy': '❌'
            }.get(check['status'], '❓')
            
            print(f"{status_emoji} {check['name']}: {check['status'].upper()}")
            
            if check['details']:
                for key, value in check['details'].items():
                    if isinstance(value, dict):
                        print(f"   {key}: {value}")
                    else:
                        print(f"   {key}: {value}")
            print()

if __name__ == "__main__":
    main()
