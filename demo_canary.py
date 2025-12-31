"""
Canary Deployment Demonstration Script

This simulates a canary deployment rollout with traffic shifting:
1% -> 5% -> 25% -> 50% -> 100%

Run: python demo_canary.py
"""

import time
import random
from datetime import datetime
from typing import Dict, List
import sys

class Colors:
    """Terminal colors for better visualization"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

class CanaryDeploymentDemo:
    """Demonstrates canary deployment with gradual traffic shifting"""
    
    def __init__(self):
        self.stages = [1, 5, 25, 50, 100]  # Traffic percentages
        self.stage_duration = 10  # seconds per stage (shortened for demo)
        self.current_stage = 0
        
        # Simulate metrics
        self.old_model_metrics = {
            'accuracy': 0.85,
            'latency_ms': 45,
            'error_rate': 0.02
        }
        
        self.new_model_metrics = {
            'accuracy': 0.88,  # Better accuracy
            'latency_ms': 42,   # Faster
            'error_rate': 0.015 # Lower errors
        }
        
        self.threshold = {
            'accuracy': 0.80,
            'latency_ms': 100,
            'error_rate': 0.05
        }
    
    def print_header(self):
        """Print deployment header"""
        print(f"\n{Colors.HEADER}{'=' * 70}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}   CANARY DEPLOYMENT DEMONSTRATION{Colors.ENDC}")
        print(f"{Colors.HEADER}{'=' * 70}{Colors.ENDC}\n")
        
        print(f"{Colors.CYAN}Strategy:{Colors.ENDC} Progressive Traffic Shifting")
        print(f"{Colors.CYAN}Stages:{Colors.ENDC} {' → '.join([f'{s}%' for s in self.stages])}")
        print(f"{Colors.CYAN}Duration:{Colors.ENDC} {self.stage_duration}s per stage")
        print()
    
    def print_models(self):
        """Display model information"""
        print(f"{Colors.BOLD}Models:{Colors.ENDC}")
        print(f"  {Colors.BLUE}● Champion (Old):{Colors.ENDC} titanic_classifier v1.0")
        print(f"  {Colors.GREEN}● Challenger (New):{Colors.ENDC} titanic_classifier v2.0")
        print()
    
    def simulate_traffic(self, canary_percentage: int, duration: int):
        """Simulate traffic distribution"""
        print(f"\n{Colors.YELLOW}[Stage {self.current_stage + 1}/{len(self.stages)}]{Colors.ENDC} "
              f"Routing {canary_percentage}% traffic to new model...")
        print(f"Monitoring for {duration} seconds...\n")
        
        # Simulate monitoring over time
        for i in range(duration):
            time.sleep(1)
            
            # Simulate requests
            total_requests = random.randint(80, 120)
            canary_requests = int(total_requests * canary_percentage / 100)
            old_requests = total_requests - canary_requests
            
            # Add some variance to metrics
            new_acc = self.new_model_metrics['accuracy'] + random.uniform(-0.02, 0.02)
            new_latency = self.new_model_metrics['latency_ms'] + random.uniform(-5, 5)
            
            # Progress bar
            progress = (i + 1) / duration
            bar_length = 30
            filled = int(bar_length * progress)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            print(f"\r  [{bar}] {int(progress * 100)}% | "
                  f"Old: {old_requests:3d} req | "
                  f"New: {canary_requests:3d} req | "
                  f"Acc: {new_acc:.3f} | "
                  f"Latency: {new_latency:.1f}ms", end='', flush=True)
        
        print()  # New line after progress
    
    def check_metrics(self, canary_percentage: int) -> bool:
        """Check if metrics meet thresholds"""
        print(f"\n{Colors.CYAN}Analyzing metrics...{Colors.ENDC}")
        
        # Simulate metric collection with some variance
        actual_accuracy = self.new_model_metrics['accuracy'] + random.uniform(-0.01, 0.01)
        actual_latency = self.new_model_metrics['latency_ms'] + random.uniform(-3, 3)
        actual_error = self.new_model_metrics['error_rate'] + random.uniform(-0.002, 0.002)
        
        metrics = {
            'Accuracy': {
                'value': actual_accuracy,
                'threshold': self.threshold['accuracy'],
                'higher_is_better': True
            },
            'Latency': {
                'value': actual_latency,
                'threshold': self.threshold['latency_ms'],
                'higher_is_better': False
            },
            'Error Rate': {
                'value': actual_error,
                'threshold': self.threshold['error_rate'],
                'higher_is_better': False
            }
        }
        
        all_passed = True
        
        for metric_name, metric_data in metrics.items():
            value = metric_data['value']
            threshold = metric_data['threshold']
            higher_is_better = metric_data['higher_is_better']
            
            if higher_is_better:
                passed = value >= threshold
            else:
                passed = value <= threshold
            
            symbol = f"{Colors.GREEN}✓{Colors.ENDC}" if passed else f"{Colors.RED}✗{Colors.ENDC}"
            
            if metric_name == 'Error Rate':
                print(f"  {symbol} {metric_name}: {value:.3f} (threshold: <{threshold})")
            elif metric_name == 'Latency':
                print(f"  {symbol} {metric_name}: {value:.1f}ms (threshold: <{threshold}ms)")
            else:
                print(f"  {symbol} {metric_name}: {value:.3f} (threshold: >{threshold})")
            
            all_passed = all_passed and passed
        
        return all_passed
    
    def rollback(self):
        """Simulate rollback"""
        print(f"\n{Colors.RED}{Colors.BOLD}⚠ ROLLBACK TRIGGERED{Colors.ENDC}")
        print(f"{Colors.RED}Metrics degraded. Rolling back to old model...{Colors.ENDC}")
        time.sleep(2)
        print(f"{Colors.GREEN}✓ Rollback complete. All traffic restored to old model.{Colors.ENDC}")
    
    def deploy_stage(self, canary_percentage: int):
        """Deploy a single canary stage"""
        self.simulate_traffic(canary_percentage, self.stage_duration)
        
        if not self.check_metrics(canary_percentage):
            self.rollback()
            return False
        
        print(f"\n{Colors.GREEN}✓ Stage {self.current_stage + 1} completed successfully!{Colors.ENDC}")
        self.current_stage += 1
        
        if canary_percentage < 100:
            print(f"{Colors.CYAN}→ Proceeding to next stage...{Colors.ENDC}")
            time.sleep(2)
        
        return True
    
    def run(self):
        """Run the full canary deployment"""
        self.print_header()
        self.print_models()
        
        input(f"{Colors.YELLOW}Press ENTER to start canary deployment...{Colors.ENDC}")
        
        for stage_percentage in self.stages:
            if not self.deploy_stage(stage_percentage):
                sys.exit(1)
        
        # Deployment complete
        print(f"\n{Colors.GREEN}{'=' * 70}{Colors.ENDC}")
        print(f"{Colors.GREEN}{Colors.BOLD}   🎉 CANARY DEPLOYMENT COMPLETED SUCCESSFULLY!{Colors.ENDC}")
        print(f"{Colors.GREEN}{'=' * 70}{Colors.ENDC}\n")
        
        print(f"{Colors.BOLD}Summary:{Colors.ENDC}")
        print(f"  • Total stages: {len(self.stages)}")
        print(f"  • All metrics passed thresholds")
        print(f"  • New model (v2.0) now serving 100% of traffic")
        print(f"  • Old model (v1.0) decommissioned")
        print()


if __name__ == "__main__":
    print(f"{Colors.CYAN}Initializing canary deployment system...{Colors.ENDC}")
    time.sleep(1)
    
    demo = CanaryDeploymentDemo()
    demo.run()
    
    print(f"{Colors.CYAN}Deployment demonstration complete!{Colors.ENDC}")
    print(f"For your school project, this demonstrates:")
    print(f"  1. Progressive traffic shifting (1% → 5% → 25% → 50% → 100%)")
    print(f"  2. Real-time monitoring at each stage")
    print(f"  3. Automatic metric validation")
    print(f"  4. Rollback capability on failure")
    print()
