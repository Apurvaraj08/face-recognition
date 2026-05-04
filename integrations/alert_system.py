#!/usr/bin/env python3
"""
Real-time Alert System for Face Recognition
Provides notifications and alerts for recognition events
"""

import os
import json
import time
import threading
from datetime import datetime
from collections import defaultdict
import winsound  # For Windows sound alerts
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
CONFIG_DIR = os.path.join(ROOT, "config")
ALERTS_DIR = os.path.join(ROOT, "alerts")

class AlertSystem:
    def __init__(self):
        self.alerts_enabled = True
        self.alert_history = []
        self.alert_rules = self.load_alert_rules()
        self.notification_queue = []
        self.alert_thread = None
        self.running = False
        
        # Ensure alerts directory exists
        os.makedirs(ALERTS_DIR, exist_ok=True)
        
    def load_alert_rules(self):
        """Load alert rules from configuration"""
        rules_file = os.path.join(CONFIG_DIR, "alert_rules.json")
        default_rules = {
            "student_detected": {
                "enabled": True,
                "sound": True,
                "email": False,
                "popup": True,
                "log": True
            },
            "unknown_face": {
                "enabled": True,
                "sound": True,
                "email": False,
                "popup": True,
                "log": True
            },
            "high_detection_count": {
                "enabled": True,
                "threshold": 100,
                "sound": True,
                "email": True,
                "popup": True,
                "log": True
            },
            "no_faces_detected": {
                "enabled": True,
                "timeout": 30,  # seconds
                "sound": True,
                "email": False,
                "popup": True,
                "log": True
            }
        }
        
        if os.path.exists(rules_file):
            try:
                with open(rules_file, 'r') as f:
                    loaded_rules = json.load(f)
                # Merge with defaults
                for rule_type, rule_config in loaded_rules.items():
                    if rule_type in default_rules:
                        default_rules[rule_type].update(rule_config)
                    else:
                        default_rules[rule_type] = rule_config
            except:
                pass
        
        return default_rules
    
    def start_alert_system(self):
        """Start the alert system background thread"""
        if not self.running:
            self.running = True
            self.alert_thread = threading.Thread(target=self._alert_worker, daemon=True)
            self.alert_thread.start()
            print("🔔 Alert system started")
    
    def stop_alert_system(self):
        """Stop the alert system"""
        self.running = False
        if self.alert_thread:
            self.alert_thread.join()
        print("🔕 Alert system stopped")
    
    def _alert_worker(self):
        """Background worker for processing alerts"""
        while self.running:
            if self.notification_queue:
                alert = self.notification_queue.pop(0)
                self._process_alert(alert)
            time.sleep(0.1)
    
    def trigger_alert(self, alert_type, message, data=None):
        """Trigger an alert"""
        if not self.alerts_enabled:
            return
        
        alert = {
            "type": alert_type,
            "message": message,
            "data": data or {},
            "timestamp": datetime.now().isoformat(),
            "id": len(self.alert_history) + 1
        }
        
        self.alert_history.append(alert)
        self.notification_queue.append(alert)
        
        # Log alert
        self._log_alert(alert)
    
    def _process_alert(self, alert):
        """Process an alert based on rules"""
        alert_type = alert["type"]
        rules = self.alert_rules.get(alert_type, {})
        
        if not rules.get("enabled", False):
            return
        
        # Sound alert
        if rules.get("sound", False):
            self._play_sound(alert_type)
        
        # Popup alert
        if rules.get("popup", False):
            self._show_popup(alert)
        
        # Email alert
        if rules.get("email", False):
            self._send_email(alert)
    
    def _play_sound(self, alert_type):
        """Play sound alert"""
        try:
            if alert_type == "student_detected":
                winsound.Beep(1000, 200)  # High pitch, short
            elif alert_type == "unknown_face":
                winsound.Beep(500, 300)   # Low pitch, longer
            elif alert_type == "high_detection_count":
                winsound.Beep(1500, 500)  # Very high pitch, long
            elif alert_type == "no_faces_detected":
                winsound.Beep(300, 400)   # Very low pitch, long
        except:
            pass  # Fail silently if sound doesn't work
    
    def _show_popup(self, alert):
        """Show popup alert"""
        try:
            import tkinter as tk
            from tkinter import messagebox
            
            root = tk.Tk()
            root.withdraw()  # Hide main window
            
            title = f"Face Recognition Alert: {alert['type']}"
            messagebox.showinfo(title, alert['message'])
            root.destroy()
        except:
            print(f"🔔 POPUP: {alert['message']}")
    
    def _send_email(self, alert):
        """Send email alert (placeholder - requires email configuration)"""
        print(f"📧 EMAIL: {alert['message']} (Email configuration needed)")
    
    def _log_alert(self, alert):
        """Log alert to file"""
        log_file = os.path.join(ALERTS_DIR, f"alerts_{datetime.now().strftime('%Y-%m-%d')}.json")
        
        # Load existing alerts
        alerts = []
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    alerts = json.load(f)
            except:
                alerts = []
        
        alerts.append(alert)
        
        # Save alerts
        with open(log_file, 'w') as f:
            json.dump(alerts, f, indent=2)
    
    def check_detection_alerts(self, recognition_stats, face_count):
        """Check for detection-based alerts"""
        # High detection count alert
        total_detections = sum(recognition_stats.values())
        high_count_rules = self.alert_rules.get("high_detection_count", {})
        
        if (high_count_rules.get("enabled", False) and 
            total_detections > high_count_rules.get("threshold", 100)):
            self.trigger_alert(
                "high_detection_count",
                f"High detection count: {total_detections}",
                {"total_detections": total_detections, "stats": dict(recognition_stats)}
            )
        
        # No faces detected alert
        no_faces_rules = self.alert_rules.get("no_faces_detected", {})
        if (no_faces_rules.get("enabled", False) and face_count == 0):
            # Check if this is the first time no faces detected
            if not hasattr(self, '_no_faces_start_time'):
                self._no_faces_start_time = time.time()
            elif time.time() - self._no_faces_start_time > no_faces_rules.get("timeout", 30):
                self.trigger_alert(
                    "no_faces_detected",
                    "No faces detected for extended period",
                    {"timeout": no_faces_rules.get("timeout", 30)}
                )
                self._no_faces_start_time = time.time()  # Reset
        else:
            if hasattr(self, '_no_faces_start_time'):
                delattr(self, '_no_faces_start_time')
    
    def trigger_student_alert(self, student_name, similarity, detection_count):
        """Trigger student detection alert"""
        rules = self.alert_rules.get("student_detected", {})
        
        if rules.get("enabled", False):
            self.trigger_alert(
                "student_detected",
                f"Student {student_name} detected (confidence: {similarity:.2f})",
                {
                    "student": student_name,
                    "similarity": similarity,
                    "detection_count": detection_count
                }
            )
    
    def trigger_unknown_alert(self, similarity):
        """Trigger unknown face alert"""
        rules = self.alert_rules.get("unknown_face", {})
        
        if rules.get("enabled", False):
            self.trigger_alert(
                "unknown_face",
                f"Unknown face detected (confidence: {similarity:.2f})",
                {"similarity": similarity}
            )
    
    def get_alert_summary(self):
        """Get summary of recent alerts"""
        recent_alerts = [alert for alert in self.alert_history 
                        if (datetime.now() - datetime.fromisoformat(alert['timestamp'])).seconds < 3600]
        
        alert_counts = defaultdict(int)
        for alert in recent_alerts:
            alert_counts[alert['type']] += 1
        
        return {
            "total_alerts": len(recent_alerts),
            "alert_types": dict(alert_counts),
            "recent_alerts": recent_alerts[-10:]  # Last 10 alerts
        }
    
    def save_alert_rules(self):
        """Save alert rules to configuration"""
        rules_file = os.path.join(CONFIG_DIR, "alert_rules.json")
        with open(rules_file, 'w') as f:
            json.dump(self.alert_rules, f, indent=2)
    
    def update_alert_rule(self, alert_type, rule_config):
        """Update alert rule configuration"""
        if alert_type in self.alert_rules:
            self.alert_rules[alert_type].update(rule_config)
            self.save_alert_rules()
            return True
        return False
