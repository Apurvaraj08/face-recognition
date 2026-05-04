#!/usr/bin/env python3
"""
Face Recognition Analytics Dashboard
Real-time analytics and monitoring interface
"""

import os
import json
import time
import threading
from datetime import datetime, timedelta
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
REPORTS_DIR = os.path.join(ROOT, "reports")
ANALYTICS_DIR = os.path.join(ROOT, "analytics")

class AnalyticsDashboard:
    def __init__(self):
        self.analytics_data = {
            'recognition_timeline': deque(maxlen=1000),
            'student_stats': defaultdict(lambda: defaultdict(int)),
            'hourly_stats': defaultdict(int),
            'quality_scores': deque(maxlen=500),
            'detection_rates': deque(maxlen=100),
            'alert_counts': defaultdict(int)
        }
        
        self.running = False
        self.update_thread = None
        self.root = None
        self.figures = {}
        
        # Ensure analytics directory exists
        os.makedirs(ANALYTICS_DIR, exist_ok=True)
        
    def start_dashboard(self):
        """Start the analytics dashboard"""
        self.running = True
        self.root = tk.Tk()
        self.root.title("Face Recognition Analytics Dashboard")
        self.root.geometry("1400x800")
        
        # Create main interface
        self._create_interface()
        
        # Start data update thread
        self.update_thread = threading.Thread(target=self._update_data_worker, daemon=True)
        self.update_thread.start()
        
        # Start matplotlib animations
        self._start_animations()
        
        print("📊 Analytics dashboard started")
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.root.mainloop()
    
    def _create_interface(self):
        """Create the dashboard interface"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Real-time Statistics
        self._create_realtime_tab(notebook)
        
        # Tab 2: Recognition Analytics
        self._create_analytics_tab(notebook)
        
        # Tab 3: Performance Metrics
        self._create_performance_tab(notebook)
        
        # Tab 4: Alert Monitoring
        self._create_alerts_tab(notebook)
    
    def _create_realtime_tab(self, notebook):
        """Create real-time statistics tab"""
        realtime_frame = ttk.Frame(notebook)
        notebook.add(realtime_frame, text="Real-time Statistics")
        
        # Current statistics
        stats_frame = ttk.LabelFrame(realtime_frame, text="Current Statistics")
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.current_stats = {
            'total_recognitions': tk.StringVar(value="0"),
            'active_students': tk.StringVar(value="0"),
            'avg_quality': tk.StringVar(value="0.00"),
            'detection_rate': tk.StringVar(value="0.0/sec")
        }
        
        for i, (key, var) in enumerate(self.current_stats.items()):
            ttk.Label(stats_frame, text=f"{key.replace('_', ' ').title()}:").grid(row=i//2, column=(i%2)*2, sticky=tk.W, padx=5, pady=2)
            ttk.Label(stats_frame, textvariable=var).grid(row=i//2, column=(i%2)*2+1, sticky=tk.W, padx=5, pady=2)
        
        # Real-time chart
        chart_frame = ttk.LabelFrame(realtime_frame, text="Recognition Timeline")
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.fig_timeline, self.ax_timeline = plt.subplots(figsize=(12, 4))
        self.canvas_timeline = FigureCanvasTkAgg(self.fig_timeline, chart_frame)
        self.canvas_timeline.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.figures['timeline'] = (self.fig_timeline, self.ax_timeline, self.canvas_timeline)
    
    def _create_analytics_tab(self, notebook):
        """Create recognition analytics tab"""
        analytics_frame = ttk.Frame(notebook)
        notebook.add(analytics_frame, text="Recognition Analytics")
        
        # Student performance chart
        student_frame = ttk.LabelFrame(analytics_frame, text="Student Recognition Performance")
        student_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.fig_student, self.ax_student = plt.subplots(figsize=(12, 4))
        self.canvas_student = FigureCanvasTkAgg(self.fig_student, student_frame)
        self.canvas_student.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.figures['student'] = (self.fig_student, self.ax_student, self.canvas_student)
        
        # Quality score distribution
        quality_frame = ttk.LabelFrame(analytics_frame, text="Quality Score Distribution")
        quality_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.fig_quality, self.ax_quality = plt.subplots(figsize=(12, 4))
        self.canvas_quality = FigureCanvasTkAgg(self.fig_quality, quality_frame)
        self.canvas_quality.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.figures['quality'] = (self.fig_quality, self.ax_quality, self.canvas_quality)
    
    def _create_performance_tab(self, notebook):
        """Create performance metrics tab"""
        performance_frame = ttk.Frame(notebook)
        notebook.add(performance_frame, text="Performance Metrics")
        
        # Detection rate chart
        rate_frame = ttk.LabelFrame(performance_frame, text="Detection Rate Over Time")
        rate_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.fig_rate, self.ax_rate = plt.subplots(figsize=(12, 4))
        self.canvas_rate = FigureCanvasTkAgg(self.fig_rate, rate_frame)
        self.canvas_rate.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.figures['rate'] = (self.fig_rate, self.ax_rate, self.canvas_rate)
        
        # Hourly statistics
        hourly_frame = ttk.LabelFrame(performance_frame, text="Hourly Recognition Statistics")
        hourly_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.fig_hourly, self.ax_hourly = plt.subplots(figsize=(12, 4))
        self.canvas_hourly = FigureCanvasTkAgg(self.fig_hourly, hourly_frame)
        self.canvas_hourly.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.figures['hourly'] = (self.fig_hourly, self.ax_hourly, self.canvas_hourly)
    
    def _create_alerts_tab(self, notebook):
        """Create alert monitoring tab"""
        alerts_frame = ttk.Frame(notebook)
        notebook.add(alerts_frame, text="Alert Monitoring")
        
        # Alert summary
        summary_frame = ttk.LabelFrame(alerts_frame, text="Alert Summary")
        summary_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.alert_summary = {
            'total_alerts': tk.StringVar(value="0"),
            'student_alerts': tk.StringVar(value="0"),
            'unknown_alerts': tk.StringVar(value="0"),
            'system_alerts': tk.StringVar(value="0")
        }
        
        for i, (key, var) in enumerate(self.alert_summary.items()):
            ttk.Label(summary_frame, text=f"{key.replace('_', ' ').title()}:").grid(row=i//2, column=(i%2)*2, sticky=tk.W, padx=5, pady=2)
            ttk.Label(summary_frame, textvariable=var).grid(row=i//2, column=(i%2)*2+1, sticky=tk.W, padx=5, pady=2)
        
        # Alert timeline
        alert_timeline_frame = ttk.LabelFrame(alerts_frame, text="Alert Timeline")
        alert_timeline_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.fig_alerts, self.ax_alerts = plt.subplots(figsize=(12, 4))
        self.canvas_alerts = FigureCanvasTkAgg(self.fig_alerts, alert_timeline_frame)
        self.canvas_alerts.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.figures['alerts'] = (self.fig_alerts, self.ax_alerts, self.canvas_alerts)
    
    def _start_animations(self):
        """Start matplotlib animations"""
        # Timeline animation
        self.ani_timeline = animation.FuncAnimation(
            self.fig_timeline, self._update_timeline_chart, 
            interval=1000, blit=False
        )
        
        # Student chart animation
        self.ani_student = animation.FuncAnimation(
            self.fig_student, self._update_student_chart, 
            interval=2000, blit=False
        )
        
        # Quality chart animation
        self.ani_quality = animation.FuncAnimation(
            self.fig_quality, self._update_quality_chart, 
            interval=2000, blit=False
        )
        
        # Rate chart animation
        self.ani_rate = animation.FuncAnimation(
            self.fig_rate, self._update_rate_chart, 
            interval=1500, blit=False
        )
        
        # Hourly chart animation
        self.ani_hourly = animation.FuncAnimation(
            self.fig_hourly, self._update_hourly_chart, 
            interval=5000, blit=False
        )
        
        # Alerts chart animation
        self.ani_alerts = animation.FuncAnimation(
            self.fig_alerts, self._update_alerts_chart, 
            interval=2000, blit=False
        )
    
    def _update_timeline_chart(self, frame):
        """Update recognition timeline chart"""
        self.ax_timeline.clear()
        
        if self.analytics_data['recognition_timeline']:
            data = list(self.analytics_data['recognition_timeline'])
            times = [datetime.fromisoformat(d['timestamp']) for d in data]
            students = [d['student'] for d in data]
            
            # Create color map for students
            unique_students = list(set(students))
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_students)))
            student_colors = dict(zip(unique_students, colors))
            
            # Plot timeline
            for student in unique_students:
                student_times = [times[i] for i, s in enumerate(students) if s == student]
                student_y = [unique_students.index(student)] * len(student_times)
                self.ax_timeline.scatter(student_times, student_y, 
                                      c=[student_colors[student]], label=student, s=50)
            
            self.ax_timeline.set_yticks(range(len(unique_students)))
            self.ax_timeline.set_yticklabels(unique_students)
            self.ax_timeline.set_xlabel('Time')
            self.ax_timeline.set_title('Recognition Timeline')
            self.ax_timeline.legend()
            self.ax_timeline.grid(True, alpha=0.3)
        
        self.canvas_timeline.draw()
    
    def _update_student_chart(self, frame):
        """Update student recognition performance chart"""
        self.ax_student.clear()
        
        student_stats = dict(self.analytics_data['student_stats'])
        if student_stats:
            students = list(student_stats.keys())
            counts = [student_stats[s]['count'] for s in students]
            avg_quality = [student_stats[s]['avg_quality'] for s in students]
            
            x = np.arange(len(students))
            width = 0.35
            
            self.ax_student.bar(x - width/2, counts, width, label='Recognition Count', alpha=0.7)
            self.ax_student.bar(x + width/2, avg_quality, width, label='Avg Quality', alpha=0.7)
            
            self.ax_student.set_xlabel('Students')
            self.ax_student.set_ylabel('Count / Quality')
            self.ax_student.set_title('Student Recognition Performance')
            self.ax_student.set_xticks(x)
            self.ax_student.set_xticklabels(students)
            self.ax_student.legend()
            self.ax_student.grid(True, alpha=0.3)
        
        self.canvas_student.draw()
    
    def _update_quality_chart(self, frame):
        """Update quality score distribution chart"""
        self.ax_quality.clear()
        
        if self.analytics_data['quality_scores']:
            scores = list(self.analytics_data['quality_scores'])
            self.ax_quality.hist(scores, bins=20, alpha=0.7, edgecolor='black')
            self.ax_quality.set_xlabel('Quality Score')
            self.ax_quality.set_ylabel('Frequency')
            self.ax_quality.set_title('Quality Score Distribution')
            self.ax_quality.grid(True, alpha=0.3)
        
        self.canvas_quality.draw()
    
    def _update_rate_chart(self, frame):
        """Update detection rate chart"""
        self.ax_rate.clear()
        
        if self.analytics_data['detection_rates']:
            rates = list(self.analytics_data['detection_rates'])
            times = list(range(len(rates)))
            
            self.ax_rate.plot(times, rates, 'b-', linewidth=2)
            self.ax_rate.fill_between(times, rates, alpha=0.3)
            self.ax_rate.set_xlabel('Time (seconds)')
            self.ax_rate.set_ylabel('Detection Rate (faces/sec)')
            self.ax_rate.set_title('Detection Rate Over Time')
            self.ax_rate.grid(True, alpha=0.3)
        
        self.canvas_rate.draw()
    
    def _update_hourly_chart(self, frame):
        """Update hourly statistics chart"""
        self.ax_hourly.clear()
        
        hourly_stats = dict(self.analytics_data['hourly_stats'])
        if hourly_stats:
            hours = list(range(24))
            counts = [hourly_stats.get(h, 0) for h in hours]
            
            self.ax_hourly.bar(hours, counts, alpha=0.7, edgecolor='black')
            self.ax_hourly.set_xlabel('Hour of Day')
            self.ax_hourly.set_ylabel('Recognition Count')
            self.ax_hourly.set_title('Hourly Recognition Statistics')
            self.ax_hourly.set_xticks(hours[::2])  # Show every 2 hours
            self.ax_hourly.grid(True, alpha=0.3)
        
        self.canvas_hourly.draw()
    
    def _update_alerts_chart(self, frame):
        """Update alerts chart"""
        self.ax_alerts.clear()
        
        alert_counts = dict(self.analytics_data['alert_counts'])
        if alert_counts:
            alert_types = list(alert_counts.keys())
            counts = list(alert_counts.values())
            
            colors = ['red', 'orange', 'yellow', 'blue', 'green']
            self.ax_alerts.pie(counts, labels=alert_types, autopct='%1.1f%%', 
                              colors=colors[:len(alert_types)])
            self.ax_alerts.set_title('Alert Distribution')
        
        self.canvas_alerts.draw()
    
    def _update_data_worker(self):
        """Background worker to update analytics data"""
        while self.running:
            self._update_analytics_data()
            self._update_ui_stats()
            time.sleep(1)
    
    def _update_analytics_data(self):
        """Update analytics data from reports"""
        try:
            # Load latest recognition stats
            stats_file = os.path.join(REPORTS_DIR, "stable_recognition_stats.json")
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                
                # Update current statistics
                total_recognitions = stats.get('total_recognitions', 0)
                unique_students = len(stats.get('recognition_stats', {}))
                
                self.current_stats['total_recognitions'].set(str(total_recognitions))
                self.current_stats['active_students'].set(str(unique_students))
                
                # Update student stats
                for student, count in stats.get('recognition_stats', {}).items():
                    self.analytics_data['student_stats'][student]['count'] = count
                    self.analytics_data['student_stats'][student]['last_seen'] = datetime.now()
                
                # Add to timeline
                current_hour = datetime.now().hour
                self.analytics_data['hourly_stats'][current_hour] += 1
                
                # Simulate quality scores (in real implementation, this would come from actual data)
                if np.random.random() > 0.3:  # 70% chance of new quality score
                    quality = np.random.beta(2, 2)  # Beta distribution for quality scores
                    self.analytics_data['quality_scores'].append(quality)
                
                # Simulate detection rates
                if len(self.analytics_data['detection_rates']) == 0:
                    self.analytics_data['detection_rates'].append(np.random.uniform(0.5, 3.0))
                else:
                    last_rate = self.analytics_data['detection_rates'][-1]
                    new_rate = last_rate + np.random.normal(0, 0.2)
                    new_rate = max(0.1, min(5.0, new_rate))  # Clamp between 0.1 and 5.0
                    self.analytics_data['detection_rates'].append(new_rate)
            
            # Load alert data
            alerts_dir = os.path.join(ROOT, "alerts")
            if os.path.exists(alerts_dir):
                alert_files = [f for f in os.listdir(alerts_dir) if f.startswith('alerts_')]
                for alert_file in alert_files:
                    alert_path = os.path.join(alerts_dir, alert_file)
                    try:
                        with open(alert_path, 'r') as f:
                            alerts = json.load(f)
                        for alert in alerts:
                            alert_type = alert.get('type', 'unknown')
                            self.analytics_data['alert_counts'][alert_type] += 1
                    except:
                        pass
            
            # Update average quality
            if self.analytics_data['quality_scores']:
                avg_quality = np.mean(list(self.analytics_data['quality_scores']))
                self.current_stats['avg_quality'].set(f"{avg_quality:.2f}")
            
            # Update detection rate
            if self.analytics_data['detection_rates']:
                current_rate = self.analytics_data['detection_rates'][-1]
                self.current_stats['detection_rate'].set(f"{current_rate:.1f}/sec")
            
            # Update alert summary
            total_alerts = sum(self.analytics_data['alert_counts'].values())
            self.alert_summary['total_alerts'].set(str(total_alerts))
            self.alert_summary['student_alerts'].set(str(self.analytics_data['alert_counts'].get('student_detected', 0)))
            self.alert_summary['unknown_alerts'].set(str(self.analytics_data['alert_counts'].get('unknown_face', 0)))
            self.alert_summary['system_alerts'].set(str(total_alerts - self.analytics_data['alert_counts'].get('student_detected', 0) - self.analytics_data['alert_counts'].get('unknown_face', 0)))
            
        except Exception as e:
            print(f"Error updating analytics data: {e}")
    
    def _update_ui_stats(self):
        """Update UI statistics"""
        if self.root and self.running:
            try:
                self.root.update_idletasks()
            except:
                pass
    
    def _on_closing(self):
        """Handle window closing"""
        self.running = False
        if self.update_thread:
            self.update_thread.join()
        plt.close('all')
        self.root.destroy()
    
    def add_recognition_event(self, student, similarity, quality):
        """Add a recognition event to analytics"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'student': student,
            'similarity': similarity,
            'quality': quality
        }
        
        self.analytics_data['recognition_timeline'].append(event)
        self.analytics_data['student_stats'][student]['count'] += 1
        self.analytics_data['student_stats'][student]['avg_quality'] = quality
        self.analytics_data['quality_scores'].append(quality)
    
    def export_analytics_report(self, filename=None):
        """Export analytics report to file"""
        if filename is None:
            filename = f"analytics_report_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'analytics_data': {
                'total_recognitions': len(self.analytics_data['recognition_timeline']),
                'student_stats': dict(self.analytics_data['student_stats']),
                'hourly_stats': dict(self.analytics_data['hourly_stats']),
                'alert_counts': dict(self.analytics_data['alert_counts']),
                'avg_quality': np.mean(list(self.analytics_data['quality_scores'])) if self.analytics_data['quality_scores'] else 0
            }
        }
        
        report_path = os.path.join(ANALYTICS_DIR, filename)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Analytics report exported to: {report_path}")
        return report_path

# Standalone dashboard runner
def run_dashboard():
    """Run the analytics dashboard standalone"""
    dashboard = AnalyticsDashboard()
    dashboard.start_dashboard()

if __name__ == "__main__":
    run_dashboard()
