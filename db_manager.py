#!/usr/bin/env python3
"""
Database management utilities for GeoIP MCP Server
"""

import os
import requests
import gzip
import tarfile
import hashlib
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class GeoIPDatabaseManager:
    """Manage GeoIP database downloads and updates"""
    
    def __init__(self, license_key: str, data_dir: str = "./data"):
        self.license_key = license_key
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.databases = {
            "GeoLite2-City": "https://download.maxmind.com/app/geoip_download?edition_id=GeoLite2-City&license_key={}&suffix=tar.gz",
            "GeoLite2-Country": "https://download.maxmind.com/app/geoip_download?edition_id=GeoLite2-Country&license_key={}&suffix=tar.gz",
            "GeoLite2-ASN": "https://download.maxmind.com/app/geoip_download?edition_id=GeoLite2-ASN&license_key={}&suffix=tar.gz"
        }
    
    def download_database(self, db_name: str) -> bool:
        """Download and extract a specific database"""
        if db_name not in self.databases:
            logger.error(f"Unknown database: {db_name}")
            return False
        
        url = self.databases[db_name].format(self.license_key)
        
        try:
            logger.info(f"Downloading {db_name}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Save compressed file
            compressed_file = self.data_dir / f"{db_name}.tar.gz"
            with open(compressed_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract the .mmdb file
            with tarfile.open(compressed_file, 'r:gz') as tar:
                for member in tar.getmembers():
                    if member.name.endswith('.mmdb'):
                        # Extract to data directory with simplified name
                        member.name = f"{db_name}.mmdb"
                        tar.extract(member, self.data_dir)
                        break
            
            # Clean up compressed file
            compressed_file.unlink()
            
            logger.info(f"Successfully downloaded and extracted {db_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {db_name}: {e}")
            return False
    
    def download_all_databases(self) -> Dict[str, bool]:
        """Download all databases"""
        results = {}
        for db_name in self.databases:
            results[db_name] = self.download_database(db_name)
        return results
    
    def check_database_age(self, db_name: str) -> int:
        """Check age of database in days"""
        db_file = self.data_dir / f"{db_name}.mmdb"
        if not db_file.exists():
            return -1
        
        mtime = datetime.fromtimestamp(db_file.stat().st_mtime)
        age = (datetime.now() - mtime).days
        return age
    
    def update_if_old(self, max_age_days: int = 7) -> Dict[str, Any]:
        """Update databases if they are older than specified days"""
        results = {}
        
        for db_name in self.databases:
            age = self.check_database_age(db_name)
            
            if age == -1:
                logger.info(f"{db_name} not found, downloading...")
                results[db_name] = {
                    "action": "download",
                    "success": self.download_database(db_name)
                }
            elif age > max_age_days:
                logger.info(f"{db_name} is {age} days old, updating...")
                results[db_name] = {
                    "action": "update",
                    "age_days": age,
                    "success": self.download_database(db_name)
                }
            else:
                logger.info(f"{db_name} is {age} days old, no update needed")
                results[db_name] = {
                    "action": "skip",
                    "age_days": age,
                    "success": True
                }
        
        return results
    
    def verify_database_integrity(self, db_name: str) -> bool:
        """Verify database file integrity"""
        db_file = self.data_dir / f"{db_name}.mmdb"
        
        if not db_file.exists():
            return False
        
        try:
            import geoip2.database
            with geoip2.database.Reader(str(db_file)) as reader:
                # Try a simple lookup to verify the database works
                if "City" in db_name:
                    reader.city("8.8.8.8")
                elif "Country" in db_name:
                    reader.country("8.8.8.8")
                elif "ASN" in db_name:
                    reader.asn("8.8.8.8")
            return True
        except Exception as e:
            logger.error(f"Database integrity check failed for {db_name}: {e}")
            return False
