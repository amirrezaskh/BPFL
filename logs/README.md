# System Logs Directory

This directory contains runtime logs from all BPFL system components, providing comprehensive monitoring and debugging capabilities for the federated learning and blockchain infrastructure.

## Directory Structure

```
logs/
├── aggregator.txt          # Central aggregation server logs
├── app.txt                 # Express API gateway logs
├── node_0.txt             # Client node 0 training logs
├── node_1.txt             # Client node 1 training logs  
├── node_2.txt             # Client node 2 training logs
└── node_3.txt             # Client node 3 training logs
```

## Log File Overview

### 🔄 **Aggregator Logs** (`aggregator.txt`)

**Content:**
- Model aggregation process details
- Contribution calculation results
- Reward distribution outcomes
- Global model update confirmations
- Blockchain interaction status

**Sample Log Entries:**
```
2024-01-15 10:30:15 - Starting aggregation for round 5
2024-01-15 10:30:16 - Received 4 model submissions
2024-01-15 10:30:17 - Computing contributions: [0.245, 0.187, 0.332, 0.201]
2024-01-15 10:30:18 - Updated global model with weighted average
2024-01-15 10:30:19 - Applied personalization with gamma=0.7
2024-01-15 10:30:20 - Distributed rewards: [73.5, 56.1, 99.6, 60.3] tokens
2024-01-15 10:30:21 - Stored results on blockchain
2024-01-15 10:30:22 - Round 5 aggregation complete
```

**Key Monitoring Points:**
- **Submission Validation**: Ensures all expected models received
- **Contribution Fairness**: Monitors contribution score distribution
- **Performance Tracking**: Global model loss improvements
- **Economic Balance**: Token distribution and pricing updates

### 🌐 **API Gateway Logs** (`app.txt`)

**Content:**
- HTTP request/response handling
- Blockchain transaction confirmations
- Round orchestration events
- Smart contract invocation results
- System health monitoring

**Sample Log Entries:**
```
2024-01-15 10:25:00 - Server starting on port 3000
2024-01-15 10:25:01 - Connected to Hyperledger Fabric network
2024-01-15 10:25:02 - Token contract initialized: tokenCC
2024-01-15 10:25:03 - Model contract initialized: modelCC
2024-01-15 10:28:45 - POST /api/model/ - Model submission from wallet_0
2024-01-15 10:28:46 - Blockchain transaction confirmed: model_0 created
2024-01-15 10:29:15 - All 4 models submitted - triggering aggregation
2024-01-15 10:30:25 - POST /api/aggregator/ - Processing reward distribution
2024-01-15 10:30:26 - Round 5 complete - starting round 6
```

**Monitoring Capabilities:**
- **Request Tracking**: All API endpoint usage
- **Blockchain Status**: Smart contract transaction confirmations
- **Round Coordination**: Training round lifecycle management
- **Error Detection**: Failed transactions and system issues

### 🤖 **Client Node Logs** (`node_[0-3].txt`)

**Content:**
- Local training progress and metrics
- Data loading and preprocessing status
- Model submission confirmations
- HTTP communication with API gateway
- Training hyperparameter details

**Sample Log Entries:**
```
2024-01-15 10:28:30 - Node 0 starting training round 5
2024-01-15 10:28:31 - Loaded personalized model from /nodes/models/g_model_0.pt
2024-01-15 10:28:32 - Training dataset: 12500 samples (uniform distribution)
2024-01-15 10:28:33 - Epoch 1/5: Loss = 1.2456
2024-01-15 10:28:45 - Epoch 2/5: Loss = 1.1892
2024-01-15 10:29:05 - Epoch 3/5: Loss = 1.1345
2024-01-15 10:29:25 - Epoch 4/5: Loss = 1.0987
2024-01-15 10:29:45 - Epoch 5/5: Loss = 1.0654
2024-01-15 10:29:46 - Training complete - submitting model to blockchain
2024-01-15 10:29:47 - Model submission confirmed: model_0
2024-01-15 10:29:48 - Waiting for next round...
```

**Training Insights:**
- **Convergence Monitoring**: Local model loss trajectories
- **Data Distribution**: Training set characteristics
- **Performance Analysis**: Epoch-by-epoch improvement
- **Network Communication**: Blockchain submission status

## Log Analysis Tools

### 📊 **Real-time Monitoring**

#### **Live Log Viewing:**
```bash
# Monitor aggregation process
tail -f logs/aggregator.txt

# Track API gateway activity  
tail -f logs/app.txt

# Follow specific client training
tail -f logs/node_0.txt

# Monitor all components simultaneously
tail -f logs/*.txt
```

#### **Multi-window Monitoring:**
```bash
# Terminal multiplexer for comprehensive monitoring
tmux new-session -d -s bpfl-monitoring

# Split into multiple panes
tmux split-window -h
tmux split-window -v
tmux select-pane -t 0
tmux split-window -v

# Monitor different components in each pane
tmux send-keys -t 0 'tail -f logs/aggregator.txt' Enter
tmux send-keys -t 1 'tail -f logs/app.txt' Enter  
tmux send-keys -t 2 'tail -f logs/node_0.txt' Enter
tmux send-keys -t 3 'tail -f logs/node_1.txt' Enter

# Attach to monitoring session
tmux attach-session -t bpfl-monitoring
```

### 🔍 **Log Analysis Scripts**

#### **Training Progress Extractor:**
```python
#!/usr/bin/env python3
"""Extract training metrics from node logs"""

import re
import json
from datetime import datetime

def extract_training_metrics(log_file):
    """Extract loss values and timing from node logs"""
    
    metrics = {
        'rounds': [],
        'epochs': [],
        'losses': [],
        'timestamps': []
    }
    
    with open(log_file, 'r') as f:
        for line in f:
            # Extract epoch losses
            epoch_match = re.search(r'Epoch (\d+)/\d+: Loss = ([\d.]+)', line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                loss = float(epoch_match.group(2))
                
                # Extract timestamp
                time_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                if time_match:
                    timestamp = datetime.strptime(time_match.group(1), '%Y-%m-%d %H:%M:%S')
                    
                    metrics['epochs'].append(epoch)
                    metrics['losses'].append(loss)
                    metrics['timestamps'].append(timestamp.isoformat())
            
            # Extract round information
            round_match = re.search(r'starting training round (\d+)', line)
            if round_match:
                round_num = int(round_match.group(1))
                metrics['rounds'].append(round_num)
    
    return metrics

# Example usage
node_0_metrics = extract_training_metrics('logs/node_0.txt')
print(f"Node 0 completed {len(node_0_metrics['rounds'])} rounds")
print(f"Final training loss: {node_0_metrics['losses'][-1]:.4f}")
```

#### **System Health Checker:**
```python
#!/usr/bin/env python3
"""Monitor system health from logs"""

import re
from collections import defaultdict

def check_system_health(log_directory):
    """Analyze logs for errors and performance issues"""
    
    health_report = {
        'errors': defaultdict(list),
        'warnings': defaultdict(list),
        'performance': {},
        'status': 'healthy'
    }
    
    log_files = ['aggregator.txt', 'app.txt', 'node_0.txt', 'node_1.txt', 'node_2.txt', 'node_3.txt']
    
    for log_file in log_files:
        try:
            with open(f'{log_directory}/{log_file}', 'r') as f:
                for line_num, line in enumerate(f, 1):
                    # Check for errors
                    if re.search(r'ERROR|Exception|Failed|Error', line, re.IGNORECASE):
                        health_report['errors'][log_file].append({
                            'line': line_num,
                            'message': line.strip(),
                            'severity': 'high'
                        })
                        health_report['status'] = 'degraded'
                    
                    # Check for warnings
                    elif re.search(r'WARNING|WARN|timeout', line, re.IGNORECASE):
                        health_report['warnings'][log_file].append({
                            'line': line_num,
                            'message': line.strip(),
                            'severity': 'medium'
                        })
        
        except FileNotFoundError:
            health_report['errors']['system'].append({
                'message': f'Log file {log_file} not found',
                'severity': 'high'
            })
            health_report['status'] = 'critical'
    
    return health_report

# Example usage
health = check_system_health('logs')
print(f"System Status: {health['status']}")
if health['errors']:
    print(f"Found {sum(len(errors) for errors in health['errors'].values())} errors")
```

#### **Performance Analyzer:**
```python
#!/usr/bin/env python3
"""Analyze system performance from logs"""

import re
from datetime import datetime
import statistics

def analyze_performance(log_directory):
    """Extract performance metrics from logs"""
    
    performance = {
        'round_duration': [],
        'aggregation_time': [],
        'training_time': [],
        'blockchain_latency': []
    }
    
    # Analyze aggregator performance
    with open(f'{log_directory}/aggregator.txt', 'r') as f:
        content = f.read()
        
        # Extract round durations
        round_starts = re.findall(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Starting aggregation', content)
        round_ends = re.findall(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*aggregation complete', content)
        
        for start, end in zip(round_starts, round_ends):
            start_time = datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
            end_time = datetime.strptime(end, '%Y-%m-%d %H:%M:%S')
            duration = (end_time - start_time).total_seconds()
            performance['aggregation_time'].append(duration)
    
    # Analyze node training performance
    for node_id in range(4):
        try:
            with open(f'{log_directory}/node_{node_id}.txt', 'r') as f:
                content = f.read()
                
                # Extract training durations
                training_starts = re.findall(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*starting training', content)
                training_ends = re.findall(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Training complete', content)
                
                for start, end in zip(training_starts, training_ends):
                    start_time = datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
                    end_time = datetime.strptime(end, '%Y-%m-%d %H:%M:%S')
                    duration = (end_time - start_time).total_seconds()
                    performance['training_time'].append(duration)
        
        except FileNotFoundError:
            continue
    
    # Calculate statistics
    stats = {}
    for metric, values in performance.items():
        if values:
            stats[metric] = {
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'std': statistics.stdev(values) if len(values) > 1 else 0,
                'min': min(values),
                'max': max(values)
            }
    
    return stats

# Example usage
perf_stats = analyze_performance('logs')
print("Performance Analysis:")
for metric, stats in perf_stats.items():
    print(f"{metric}: {stats['mean']:.2f}±{stats['std']:.2f}s")
```

### 📈 **Log Visualization**

#### **Training Progress Visualization:**
```python
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

def visualize_training_progress(log_directory):
    """Create visual timeline of training progress"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot training losses for each node
    for node_id in range(4):
        try:
            metrics = extract_training_metrics(f'{log_directory}/node_{node_id}.txt')
            
            if metrics['losses']:
                ax = axes[node_id // 2, node_id % 2]
                
                # Convert timestamps
                timestamps = [datetime.fromisoformat(ts) for ts in metrics['timestamps']]
                
                ax.plot(timestamps, metrics['losses'], 'o-', linewidth=2, markersize=4)
                ax.set_title(f'Node {node_id} Training Progress')
                ax.set_xlabel('Time')
                ax.set_ylabel('Training Loss')
                ax.grid(True, alpha=0.3)
                
                # Format x-axis
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        except Exception as e:
            print(f"Error processing node {node_id}: {e}")
    
    plt.tight_layout()
    plt.savefig('logs/training_progress_timeline.png', dpi=300, bbox_inches='tight')
    plt.show()

# Example usage
visualize_training_progress('logs')
```

### 🚨 **Automated Monitoring**

#### **Log Alert System:**
```bash
#!/bin/bash
# Real-time log monitoring with alerts

LOG_DIR="logs"
ALERT_KEYWORDS="ERROR|CRITICAL|FAILED|Exception"

# Function to send alerts (customize as needed)
send_alert() {
    local message="$1"
    echo "ALERT: $message" >> alerts.log
    # Add email/Slack notification here
    # curl -X POST -H 'Content-type: application/json' \
    #   --data "{\"text\":\"BPFL Alert: $message\"}" \
    #   $SLACK_WEBHOOK_URL
}

# Monitor all log files for errors
tail -F $LOG_DIR/*.txt | while read line; do
    if echo "$line" | grep -E "$ALERT_KEYWORDS" > /dev/null; then
        timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        send_alert "[$timestamp] $line"
    fi
done
```

#### **Performance Threshold Monitoring:**
```python
#!/usr/bin/env python3
"""Monitor performance thresholds"""

import time
import json
from datetime import datetime, timedelta

class PerformanceMonitor:
    def __init__(self, config_file='monitoring_config.json'):
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        self.thresholds = self.config['thresholds']
        self.alerts = []
    
    def check_training_time_threshold(self):
        """Alert if training takes too long"""
        current_time = datetime.now()
        
        for node_id in range(4):
            log_file = f'logs/node_{node_id}.txt'
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    
                    # Find last training start
                    for line in reversed(lines):
                        if 'starting training' in line:
                            start_time = self.extract_timestamp(line)
                            if start_time:
                                duration = (current_time - start_time).total_seconds()
                                if duration > self.thresholds['max_training_time']:
                                    self.alert(f'Node {node_id} training timeout: {duration:.1f}s')
                            break
            except FileNotFoundError:
                self.alert(f'Node {node_id} log file missing')
    
    def extract_timestamp(self, line):
        """Extract timestamp from log line"""
        import re
        match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
        if match:
            return datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
        return None
    
    def alert(self, message):
        """Handle alert generation"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'severity': 'warning'
        }
        self.alerts.append(alert)
        print(f"ALERT: {message}")
    
    def run_monitoring_loop(self):
        """Continuous monitoring loop"""
        while True:
            self.check_training_time_threshold()
            time.sleep(30)  # Check every 30 seconds

# Example monitoring configuration
config = {
    "thresholds": {
        "max_training_time": 300,      # 5 minutes
        "max_aggregation_time": 60,    # 1 minute
        "min_contribution": 0.001,     # Minimum expected contribution
        "max_loss_increase": 0.1       # Maximum acceptable loss increase
    }
}

# Save configuration
with open('monitoring_config.json', 'w') as f:
    json.dump(config, f, indent=2)

# Start monitoring
if __name__ == '__main__':
    monitor = PerformanceMonitor()
    monitor.run_monitoring_loop()
```

## Log Retention and Management

### 🗂️ **Log Rotation**

```bash
#!/bin/bash
# Log rotation script to manage disk space

LOG_DIR="logs"
BACKUP_DIR="logs/archive"
MAX_SIZE="100M"  # Maximum log file size

mkdir -p $BACKUP_DIR

for log_file in $LOG_DIR/*.txt; do
    if [ -f "$log_file" ]; then
        # Check file size
        size=$(du -h "$log_file" | cut -f1)
        
        if [[ $(du -b "$log_file" | cut -f1) -gt 104857600 ]]; then  # 100MB
            # Rotate log file
            timestamp=$(date +%Y%m%d_%H%M%S)
            backup_name="$BACKUP_DIR/$(basename $log_file .txt)_$timestamp.txt"
            
            mv "$log_file" "$backup_name"
            touch "$log_file"  # Create new empty log file
            
            echo "Rotated $log_file to $backup_name"
        fi
    fi
done

# Compress old backups
find $BACKUP_DIR -name "*.txt" -mtime +7 -exec gzip {} \;

# Remove very old compressed logs
find $BACKUP_DIR -name "*.txt.gz" -mtime +30 -delete
```

### 📊 **Log Analytics Dashboard**

```python
#!/usr/bin/env python3
"""Generate log analytics dashboard"""

import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def generate_dashboard():
    """Create comprehensive log analytics dashboard"""
    
    # Analyze system activity over time
    activity_stats = analyze_system_activity()
    
    # Generate dashboard plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # System uptime
    axes[0,0].plot(activity_stats['timestamps'], activity_stats['uptime'])
    axes[0,0].set_title('System Uptime')
    axes[0,0].set_ylabel('Hours')
    
    # Request rate
    axes[0,1].plot(activity_stats['timestamps'], activity_stats['request_rate'])
    axes[0,1].set_title('API Request Rate')
    axes[0,1].set_ylabel('Requests/min')
    
    # Error rate
    axes[0,2].plot(activity_stats['timestamps'], activity_stats['error_rate'])
    axes[0,2].set_title('Error Rate')
    axes[0,2].set_ylabel('Errors/hour')
    
    # Training performance
    axes[1,0].plot(activity_stats['timestamps'], activity_stats['avg_training_time'])
    axes[1,0].set_title('Average Training Time')
    axes[1,0].set_ylabel('Seconds')
    
    # Blockchain transactions
    axes[1,1].plot(activity_stats['timestamps'], activity_stats['tx_count'])
    axes[1,1].set_title('Blockchain Transactions')
    axes[1,1].set_ylabel('Transactions/hour')
    
    # Resource utilization
    axes[1,2].plot(activity_stats['timestamps'], activity_stats['cpu_usage'])
    axes[1,2].set_title('CPU Utilization')
    axes[1,2].set_ylabel('Percentage')
    
    plt.tight_layout()
    plt.savefig('logs/system_dashboard.png', dpi=300, bbox_inches='tight')
    
    return activity_stats

# Example usage
dashboard_data = generate_dashboard()
print("Dashboard generated: logs/system_dashboard.png")
```

---

The comprehensive logging system provides essential monitoring, debugging, and performance analysis capabilities for the BPFL framework, enabling operators to maintain system health and optimize performance across all components.
