#!/usr/bin/env python3
"""
Performance monitoring and metrics collection for GeoIP MCP Server
"""

import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
import json

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    requests_per_second: float
    cache_hit_rate: float
    active_connections: int
    response_time_avg: float
    database_query_time: float

class PerformanceMonitor:
    """Monitor server performance and collect metrics"""
    
    def __init__(self, collection_interval: int = 60):
        self.collection_interval = collection_interval
        self.metrics_history: List[PerformanceMetrics] = []
        self.request_times: List[float] = []
        self.db_query_times: List[float] = []
        self.last_request_count = 0
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start performance monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._collect_metrics)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def record_request_time(self, response_time: float):
        """Record a request response time"""
        self.request_times.append(response_time)
        # Keep only last 1000 requests
        if len(self.request_times) > 1000:
            self.request_times = self.request_times[-1000:]
    
    def record_db_query_time(self, query_time: float):
        """Record a database query time"""
        self.db_query_times.append(query_time)
        # Keep only last 1000 queries
        if len(self.db_query_times) > 1000:
            self.db_query_times = self.db_query_times[-1000:]
    
    def _collect_metrics(self):
        """Collect performance metrics periodically"""
        while self.monitoring:
            try:
                # System metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                memory_used_mb = memory.used / (1024 * 1024)
                
                # Application metrics
                avg_response_time = sum(self.request_times) / len(self.request_times) if self.request_times else 0
                avg_db_query_time = sum(self.db_query_times) / len(self.db_query_times) if self.db_query_times else 0
                
                # Create metrics entry
                metrics = PerformanceMetrics(
                    timestamp=datetime.now().isoformat(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory_percent,
                    memory_used_mb=memory_used_mb,
                    requests_per_second=0,  # Will be calculated by server
                    cache_hit_rate=0,       # Will be provided by server
                    active_connections=0,   # Will be provided by server
                    response_time_avg=avg_response_time,
                    database_query_time=avg_db_query_time
                )
                
                self.metrics_history.append(metrics)
                
                # Keep only last 24 hours of metrics (assuming 1-minute intervals)
                if len(self.metrics_history) > 1440:
                    self.metrics_history = self.metrics_history[-1440:]
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                print(f"Error collecting metrics: {e}")
                time.sleep(self.collection_interval)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        if not self.metrics_history:
            return {}
        
        latest = self.metrics_history[-1]
        return asdict(latest)
    
    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance metrics summary for the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_metrics = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m.timestamp) >= cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        # Calculate averages and extremes
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        response_times = [m.response_time_avg for m in recent_metrics if m.response_time_avg > 0]
        
        return {
            "period_hours": hours,
            "data_points": len(recent_metrics),
            "cpu": {
                "avg": sum(cpu_values) / len(cpu_values),
                "min": min(cpu_values),
                "max": max(cpu_values)
            },
            "memory": {
                "avg": sum(memory_values) / len(memory_values),
                "min": min(memory_values),
                "max": max(memory_values)
            },
            "response_time": {
                "avg": sum(response_times) / len(response_times) if response_times else 0,
                "min": min(response_times) if response_times else 0,
                "max": max(response_times) if response_times else 0
            }
        }
