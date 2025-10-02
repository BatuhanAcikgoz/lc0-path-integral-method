#!/usr/bin/env python3
"""
LC0 Path Integral Performans Analiz Scripti

Bu script, gömülü LC0 Path Integral sisteminin performansını detaylı olarak analiz eder
ve çeşitli parametreler altında davranışını inceler.

Kullanım:
    python path_integral_performance_analyzer.py --lc0-path ./lc0
"""

import argparse
# Matplotlib backend: GUI gerektirmeyen Agg
import matplotlib
matplotlib.use('Agg')
import chess
import chess.engine
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import subprocess
import sys
import os
from pathlib import Path
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

class PathIntegralPerformanceAnalyzer:
    """LC0 Path Integral performans analiz sınıfı"""
    
    def __init__(self, lc0_path: str):
        self.lc0_path = lc0_path
        self.results = {
            'lambda_analysis': {},
            'sample_analysis': {},
            'mode_analysis': {},
            'position_complexity': {},
            'resource_usage': {},
            'scalability': {}
        }
        
        # Test pozisyonları (karmaşıklık seviyelerine göre)
        self.test_positions = [
            {
                'name': 'simple_opening',
                'fen': 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1',
                'complexity': 'low',
                'description': 'Basit açılış pozisyonu'
            },
            {
                'name': 'complex_opening',
                'fen': 'r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 4',
                'complexity': 'medium',
                'description': 'Karmaşık açılış pozisyonu'
            },
            {
                'name': 'tactical_middlegame',
                'fen': 'r2qkb1r/ppp2ppp/2n1bn2/3pp3/3PP3/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 0 6',
                'complexity': 'high',
                'description': 'Taktiksel orta oyun'
            },
            {
                'name': 'complex_middlegame',
                'fen': 'r1bq1rk1/ppp1nppp/3p1n2/4p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQ - 0 7',
                'complexity': 'high',
                'description': 'Karmaşık orta oyun'
            },
            {
                'name': 'simple_endgame',
                'fen': '8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1',
                'complexity': 'low',
                'description': 'Basit son oyun'
            },
            {
                'name': 'complex_endgame',
                'fen': '2r3k1/1p3ppp/p2p4/4n3/P1P1P3/2N2P2/1P4PP/3R2K1 w - - 0 1',
                'complexity': 'medium',
                'description': 'Karmaşık son oyun'
            }
        ]
        
        # Lambda değerleri analizi
        self.lambda_values = [0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
        
        # Sample sayıları analizi
        self.sample_counts = [10, 25, 50, 75, 100, 150, 200, 300, 500, 1000]
        
        # Modlar ve ödül mekanizmaları
        self.modes = ['competitive', 'quantum_limit']
        self.reward_modes = ['policy', 'cp_score', 'hybrid']
    
    def monitor_resources(self, duration: float) -> Dict:
        """Sistem kaynaklarını izle"""
        start_time = time.time()
        cpu_samples = []
        memory_samples = []
        
        def collect_samples():
            while time.time() - start_time < duration:
                cpu_samples.append(psutil.cpu_percent(interval=0.1))
                memory_samples.append(psutil.virtual_memory().percent)
                time.sleep(0.1)
        
        monitor_thread = threading.Thread(target=collect_samples)
        monitor_thread.start()
        monitor_thread.join()
        
        return {
            'cpu_mean': np.mean(cpu_samples) if cpu_samples else 0,
            'cpu_max': np.max(cpu_samples) if cpu_samples else 0,
            'memory_mean': np.mean(memory_samples) if memory_samples else 0,
            'memory_max': np.max(memory_samples) if memory_samples else 0,
            'duration': duration
        }
    
    def run_single_analysis(self, position: Dict, config: Dict) -> Dict:
        """Tek bir analiz testi çalıştır"""
        try:
            engine = chess.engine.SimpleEngine.popen_uci(self.lc0_path)
            
            # Konfigürasyon uygula
            # engine.py ile tutarlı LC0 ayarları ekle
            enhanced_config = config.copy()
            enhanced_config.update({
                'PolicyTemperature': 0.7,  # Doğru UCI seçeneği
                'CPuct': 1.0,
                # Diğer seçenekler mevcut değil, atla
            })
            engine.configure(enhanced_config)
            
            board = chess.Board(position['fen'])
            
            # Kaynak izlemeyi başlat
            start_time = time.perf_counter()
            
            # Analiz çalıştır
            info = engine.analyse(board, chess.engine.Limit(time=15.0))
            
            end_time = time.perf_counter()
            analysis_time = end_time - start_time
            
            result = {
                'position': position['name'],
                'complexity': position['complexity'],
                'config': config.copy(),
                'time': analysis_time,
                'nodes': info.get('nodes', 0),
                'nps': info.get('nps', 0),
                'best_move': str(info['pv'][0]) if info.get('pv') else None,
                'evaluation': info['score'].relative.score() if info.get('score') else 0,
                'pv_length': len(info.get('pv', [])),
                'success': True,
                'error': None
            }
            
            engine.quit()
            return result
            
        except Exception as e:
            return {
                'position': position['name'],
                'complexity': position['complexity'],
                'config': config.copy(),
                'success': False,
                'error': str(e),
                'time': 0,
                'nodes': 0,
                'nps': 0
            }
    
    def analyze_lambda_sensitivity(self):
        """Lambda değerlerinin performansa etkisini analiz et"""
        print("Lambda Sensitivity Analizi Başlıyor...")
        print("-" * 50)
        
        results = []
        
        # Sabit parametreler
        base_config = {
            'PathIntegralSamples': 50,
            'PathIntegralMode': 'competitive'
        }
        
        total_tests = len(self.lambda_values) * len(self.test_positions)
        current_test = 0
        
        for lambda_val in self.lambda_values:
            for position in self.test_positions:
                current_test += 1
                print(f"[{current_test}/{total_tests}] Lambda: {lambda_val}, Pozisyon: {position['name']}")
                
                config = base_config.copy()
                config['PathIntegralLambda'] = lambda_val
                
                result = self.run_single_analysis(position, config)
                results.append(result)
                
                if result['success']:
                    print(f"  Süre: {result['time']:.2f}s, NPS: {result['nps']:,}")
                else:
                    print(f"  Hata: {result['error']}")
        
        self.results['lambda_analysis'] = results
        print("Lambda analizi tamamlandı.\n")
    
    def analyze_sample_scaling(self):
        """Sample sayısının performansa etkisini analiz et"""
        print("Sample Scaling Analizi Başlıyor...")
        print("-" * 50)
        
        results = []
        
        # Sabit parametreler
        base_config = {
            'PathIntegralLambda': 0.1,
            'PathIntegralMode': 'competitive'
        }
        
        total_tests = len(self.sample_counts) * len(self.test_positions)
        current_test = 0
        
        for sample_count in self.sample_counts:
            for position in self.test_positions:
                current_test += 1
                print(f"[{current_test}/{total_tests}] Samples: {sample_count}, Pozisyon: {position['name']}")
                
                config = base_config.copy()
                config['PathIntegralSamples'] = sample_count
                
                result = self.run_single_analysis(position, config)
                results.append(result)
                
                if result['success']:
                    print(f"  Süre: {result['time']:.2f}s, NPS: {result['nps']:,}")
                else:
                    print(f"  Hata: {result['error']}")
        
        self.results['sample_analysis'] = results
        print("Sample scaling analizi tamamlandı.\n")
    
    def analyze_mode_performance(self):
        """Farklı modların performansını karşılaştır"""
        print("Mode Performance Analizi Başlıyor...")
        print("-" * 50)
        
        results = []
        
        # Test konfigürasyonları
        test_configs = []
        
        # Competitive mode
        test_configs.append({
            'PathIntegralLambda': 0.1,
            'PathIntegralSamples': 50,
            'PathIntegralMode': 'competitive'
        })
        
        # Quantum limit modes
        for reward_mode in self.reward_modes:
            test_configs.append({
                'PathIntegralLambda': 0.1,
                'PathIntegralSamples': 50,
                'PathIntegralMode': 'quantum_limit',
                'PathIntegralRewardMode': reward_mode
            })
        
        total_tests = len(test_configs) * len(self.test_positions)
        current_test = 0
        
        for config in test_configs:
            for position in self.test_positions:
                current_test += 1
                mode_str = config['PathIntegralMode']
                if 'PathIntegralRewardMode' in config:
                    mode_str += f"_{config['PathIntegralRewardMode']}"
                
                print(f"[{current_test}/{total_tests}] Mode: {mode_str}, Pozisyon: {position['name']}")
                
                result = self.run_single_analysis(position, config)
                results.append(result)
                
                if result['success']:
                    print(f"  Süre: {result['time']:.2f}s, NPS: {result['nps']:,}")
                else:
                    print(f"  Hata: {result['error']}")
        
        self.results['mode_analysis'] = results
        print("Mode performance analizi tamamlandı.\n")
    
    def analyze_position_complexity(self):
        """Pozisyon karmaşıklığının performansa etkisini analiz et"""
        print("Position Complexity Analizi Başlıyor...")
        print("-" * 50)
        
        results = []
        
        # Sabit konfigürasyon
        config = {
            'PathIntegralLambda': 0.1,
            'PathIntegralSamples': 100,
            'PathIntegralMode': 'quantum_limit',
            'PathIntegralRewardMode': 'hybrid'
        }
        
        # Her pozisyonu birden fazla kez test et
        for run in range(3):  # 3 tekrar
            for position in self.test_positions:
                print(f"Run {run+1}/3, Pozisyon: {position['name']} ({position['complexity']})")
                
                result = self.run_single_analysis(position, config)
                result['run'] = run + 1
                results.append(result)
                
                if result['success']:
                    print(f"  Süre: {result['time']:.2f}s, NPS: {result['nps']:,}")
                else:
                    print(f"  Hata: {result['error']}")
        
        self.results['position_complexity'] = results
        print("Position complexity analizi tamamlandı.\n")
    
    def create_performance_visualizations(self):
        """Performans görselleştirmeleri oluştur"""
        print("Performans görselleştirmeleri oluşturuluyor...")
        
        # Grafik stili
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Lambda Sensitivity Analizi
        if self.results['lambda_analysis']:
            df_lambda = pd.DataFrame([r for r in self.results['lambda_analysis'] if r['success']])
            
            if not df_lambda.empty:
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle('Lambda Değerinin Performansa Etkisi', fontsize=16)
                
                # Süre vs Lambda
                lambda_time = df_lambda.groupby('config')['time'].mean().reset_index()
                lambda_vals = [c['PathIntegralLambda'] for c in lambda_time['config']]
                axes[0, 0].semilogx(lambda_vals, lambda_time['time'], 'o-')
                axes[0, 0].set_xlabel('Lambda Değeri')
                axes[0, 0].set_ylabel('Ortalama Süre (s)')
                axes[0, 0].set_title('Lambda vs Analiz Süresi')
                axes[0, 0].grid(True, alpha=0.3)
                
                # NPS vs Lambda
                lambda_nps = df_lambda.groupby('config')['nps'].mean().reset_index()
                axes[0, 1].semilogx(lambda_vals, lambda_nps['nps'], 'o-', color='orange')
                axes[0, 1].set_xlabel('Lambda Değeri')
                axes[0, 1].set_ylabel('Ortalama NPS')
                axes[0, 1].set_title('Lambda vs Nodes Per Second')
                axes[0, 1].grid(True, alpha=0.3)
                
                # Karmaşıklığa göre lambda etkisi
                for complexity in ['low', 'medium', 'high']:
                    subset = df_lambda[df_lambda['complexity'] == complexity]
                    if not subset.empty:
                        comp_time = subset.groupby('config')['time'].mean().reset_index()
                        comp_lambda_vals = [c['PathIntegralLambda'] for c in comp_time['config']]
                        axes[1, 0].semilogx(comp_lambda_vals, comp_time['time'], 'o-', label=complexity)
                
                axes[1, 0].set_xlabel('Lambda Değeri')
                axes[1, 0].set_ylabel('Ortalama Süre (s)')
                axes[1, 0].set_title('Pozisyon Karmaşıklığına Göre Lambda Etkisi')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
                
                # Lambda dağılımı heatmap
                pivot_data = df_lambda.pivot_table(values='time', index='position', 
                                                  columns=[c['PathIntegralLambda'] for c in df_lambda['config']], 
                                                  aggfunc='mean')
                if not pivot_data.empty:
                    sns.heatmap(pivot_data, ax=axes[1, 1], cmap='viridis', annot=True, fmt='.2f')
                    axes[1, 1].set_title('Pozisyon-Lambda Süre Heatmap')
                
                plt.tight_layout()
                plt.savefig('path_integral_lambda_analysis.png', dpi=300, bbox_inches='tight')
                # plt.show() kaldırıldı: sadece dosyaya kaydediyoruz

        # 2. Sample Scaling Analizi
        if self.results['sample_analysis']:
            df_samples = pd.DataFrame([r for r in self.results['sample_analysis'] if r['success']])
            
            if not df_samples.empty:
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle('Sample Sayısının Performansa Etkisi', fontsize=16)
                
                # Süre vs Samples
                sample_time = df_samples.groupby('config')['time'].mean().reset_index()
                sample_counts = [c['PathIntegralSamples'] for c in sample_time['config']]
                axes[0, 0].plot(sample_counts, sample_time['time'], 'o-')
                axes[0, 0].set_xlabel('Sample Sayısı')
                axes[0, 0].set_ylabel('Ortalama Süre (s)')
                axes[0, 0].set_title('Sample Sayısı vs Analiz Süresi')
                axes[0, 0].grid(True, alpha=0.3)
                
                # Efficiency (NPS/Sample)
                sample_nps = df_samples.groupby('config')['nps'].mean().reset_index()
                efficiency = [nps/samples for nps, samples in zip(sample_nps['nps'], sample_counts)]
                axes[0, 1].plot(sample_counts, efficiency, 'o-', color='green')
                axes[0, 1].set_xlabel('Sample Sayısı')
                axes[0, 1].set_ylabel('Efficiency (NPS/Sample)')
                axes[0, 1].set_title('Sample Efficiency')
                axes[0, 1].grid(True, alpha=0.3)
                
                # Scaling factor
                if len(sample_time) > 1:
                    base_time = sample_time['time'].iloc[0]
                    scaling_factors = sample_time['time'] / base_time
                    axes[1, 0].plot(sample_counts, scaling_factors, 'o-', color='red')
                    axes[1, 0].set_xlabel('Sample Sayısı')
                    axes[1, 0].set_ylabel('Scaling Factor (vs minimum)')
                    axes[1, 0].set_title('Scaling Behavior')
                    axes[1, 0].grid(True, alpha=0.3)
                
                # Sample distribution by complexity
                for complexity in ['low', 'medium', 'high']:
                    subset = df_samples[df_samples['complexity'] == complexity]
                    if not subset.empty:
                        comp_time = subset.groupby('config')['time'].mean().reset_index()
                        comp_samples = [c['PathIntegralSamples'] for c in comp_time['config']]
                        axes[1, 1].plot(comp_samples, comp_time['time'], 'o-', label=complexity)
                
                axes[1, 1].set_xlabel('Sample Sayısı')
                axes[1, 1].set_ylabel('Ortalama Süre (s)')
                axes[1, 1].set_title('Karmaşıklığa Göre Sample Scaling')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig('path_integral_sample_analysis.png', dpi=300, bbox_inches='tight')
                # plt.show() kaldırıldı: sadece dosyaya kaydediyoruz

        # 3. Mode Performance Karşılaştırması
        if self.results['mode_analysis']:
            df_modes = pd.DataFrame([r for r in self.results['mode_analysis'] if r['success']])
            
            if not df_modes.empty:
                # Mode string oluştur
                df_modes['mode_str'] = df_modes['config'].apply(
                    lambda c: c['PathIntegralMode'] + 
                    (f"_{c['PathIntegralRewardMode']}" if 'PathIntegralRewardMode' in c else "")
                )
                
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle('Mode Performance Karşılaştırması', fontsize=16)
                
                # Ortalama süre karşılaştırması
                mode_time = df_modes.groupby('mode_str')['time'].mean().reset_index()
                axes[0, 0].bar(mode_time['mode_str'], mode_time['time'])
                axes[0, 0].set_xlabel('Mode')
                axes[0, 0].set_ylabel('Ortalama Süre (s)')
                axes[0, 0].set_title('Mode vs Analiz Süresi')
                axes[0, 0].tick_params(axis='x', rotation=45)
                axes[0, 0].grid(True, alpha=0.3)
                
                # NPS karşılaştırması
                mode_nps = df_modes.groupby('mode_str')['nps'].mean().reset_index()
                axes[0, 1].bar(mode_nps['mode_str'], mode_nps['nps'], color='orange')
                axes[0, 1].set_xlabel('Mode')
                axes[0, 1].set_ylabel('Ortalama NPS')
                axes[0, 1].set_title('Mode vs Nodes Per Second')
                axes[0, 1].tick_params(axis='x', rotation=45)
                axes[0, 1].grid(True, alpha=0.3)
                
                # Karmaşıklığa göre mode performansı
                pivot_mode_complexity = df_modes.pivot_table(
                    values='time', index='mode_str', columns='complexity', aggfunc='mean'
                )
                if not pivot_mode_complexity.empty:
                    sns.heatmap(pivot_mode_complexity, ax=axes[1, 0], annot=True, fmt='.2f', cmap='viridis')
                    axes[1, 0].set_title('Mode-Karmaşıklık Süre Heatmap')
                
                # Box plot - mode performance distribution
                if len(df_modes) > 10:
                    sns.boxplot(data=df_modes, x='mode_str', y='time', ax=axes[1, 1])
                    axes[1, 1].set_xlabel('Mode')
                    axes[1, 1].set_ylabel('Süre (s)')
                    axes[1, 1].set_title('Mode Performance Dağılımı')
                    axes[1, 1].tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                plt.savefig('path_integral_mode_analysis.png', dpi=300, bbox_inches='tight')
                # plt.show() kaldırıldı: sadece dosyaya kaydediyoruz

        print("Görselleştirmeler tamamlandı:")
        print("  - path_integral_lambda_analysis.png")
        print("  - path_integral_sample_analysis.png")
        print("  - path_integral_mode_analysis.png")
    
    def generate_performance_report(self):
        """Detaylı performans raporu oluştur"""
        report_filename = 'path_integral_performance_report.md'
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("# LC0 Path Integral Performans Analiz Raporu\n\n")
            f.write(f"**Rapor Tarihi**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Lambda Analizi
            if self.results['lambda_analysis']:
                successful_lambda = [r for r in self.results['lambda_analysis'] if r['success']]
                if successful_lambda:
                    f.write("## Lambda Sensitivity Analizi\n\n")
                    
                    # Lambda değerlerine göre ortalama performans
                    lambda_perf = defaultdict(list)
                    for result in successful_lambda:
                        lambda_val = result['config']['PathIntegralLambda']
                        lambda_perf[lambda_val].append(result['time'])
                    
                    f.write("| Lambda | Ortalama Süre (s) | Std Dev | Test Sayısı |\n")
                    f.write("|--------|------------------|---------|-------------|\n")
                    
                    for lambda_val in sorted(lambda_perf.keys()):
                        times = lambda_perf[lambda_val]
                        mean_time = np.mean(times)
                        std_time = np.std(times)
                        count = len(times)
                        f.write(f"| {lambda_val} | {mean_time:.3f} | {std_time:.3f} | {count} |\n")
                    
                    # En iyi lambda değeri
                    best_lambda = min(lambda_perf.keys(), key=lambda x: np.mean(lambda_perf[x]))
                    f.write(f"\n**En İyi Lambda Değeri**: {best_lambda} ({np.mean(lambda_perf[best_lambda]):.3f}s)\n\n")
            
            # Sample Analizi
            if self.results['sample_analysis']:
                successful_samples = [r for r in self.results['sample_analysis'] if r['success']]
                if successful_samples:
                    f.write("## Sample Scaling Analizi\n\n")
                    
                    sample_perf = defaultdict(list)
                    for result in successful_samples:
                        sample_count = result['config']['PathIntegralSamples']
                        sample_perf[sample_count].append(result['time'])
                    
                    f.write("| Sample Sayısı | Ortalama Süre (s) | Efficiency (s/sample) | Scaling Factor |\n")
                    f.write("|---------------|------------------|----------------------|----------------|\n")
                    
                    base_time = None
                    for sample_count in sorted(sample_perf.keys()):
                        times = sample_perf[sample_count]
                        mean_time = np.mean(times)
                        efficiency = mean_time / sample_count
                        
                        if base_time is None:
                            base_time = mean_time
                            scaling_factor = 1.0
                        else:
                            scaling_factor = mean_time / base_time
                        
                        f.write(f"| {sample_count} | {mean_time:.3f} | {efficiency:.6f} | {scaling_factor:.2f} |\n")
                    
                    f.write("\n")
            
            # Mode Analizi
            if self.results['mode_analysis']:
                successful_modes = [r for r in self.results['mode_analysis'] if r['success']]
                if successful_modes:
                    f.write("## Mode Performance Analizi\n\n")
                    
                    mode_perf = defaultdict(list)
                    for result in successful_modes:
                        mode = result['config']['PathIntegralMode']
                        if 'PathIntegralRewardMode' in result['config']:
                            mode += f"_{result['config']['PathIntegralRewardMode']}"
                        mode_perf[mode].append(result['time'])
                    
                    f.write("| Mode | Ortalama Süre (s) | Std Dev | Test Sayısı |\n")
                    f.write("|------|------------------|---------|-------------|\n")
                    
                    for mode in sorted(mode_perf.keys()):
                        times = mode_perf[mode]
                        mean_time = np.mean(times)
                        std_time = np.std(times)
                        count = len(times)
                        f.write(f"| {mode} | {mean_time:.3f} | {std_time:.3f} | {count} |\n")
                    
                    # En hızlı mod
                    fastest_mode = min(mode_perf.keys(), key=lambda x: np.mean(mode_perf[x]))
                    f.write(f"\n**En Hızlı Mode**: {fastest_mode} ({np.mean(mode_perf[fastest_mode]):.3f}s)\n\n")
            
            # Pozisyon Karmaşıklığı
            if self.results['position_complexity']:
                successful_complexity = [r for r in self.results['position_complexity'] if r['success']]
                if successful_complexity:
                    f.write("## Pozisyon Karmaşıklığı Analizi\n\n")
                    
                    complexity_perf = defaultdict(list)
                    for result in successful_complexity:
                        complexity = result['complexity']
                        complexity_perf[complexity].append(result['time'])
                    
                    f.write("| Karmaşıklık | Ortalama Süre (s) | Std Dev | Test Sayısı |\n")
                    f.write("|-------------|------------------|---------|-------------|\n")
                    
                    for complexity in ['low', 'medium', 'high']:
                        if complexity in complexity_perf:
                            times = complexity_perf[complexity]
                            mean_time = np.mean(times)
                            std_time = np.std(times)
                            count = len(times)
                            f.write(f"| {complexity} | {mean_time:.3f} | {std_time:.3f} | {count} |\n")
                    
                    f.write("\n")
            
            # Öneriler
            f.write("## Performans Önerileri\n\n")
            
            # Lambda önerisi
            if self.results['lambda_analysis']:
                successful_lambda = [r for r in self.results['lambda_analysis'] if r['success']]
                if successful_lambda:
                    lambda_perf = defaultdict(list)
                    for result in successful_lambda:
                        lambda_val = result['config']['PathIntegralLambda']
                        lambda_perf[lambda_val].append(result['time'])
                    
                    best_lambda = min(lambda_perf.keys(), key=lambda x: np.mean(lambda_perf[x]))
                    f.write(f"1. **Lambda Değeri**: Optimal performans için λ={best_lambda} kullanın.\n")
            
            # Sample önerisi
            if self.results['sample_analysis']:
                f.write("2. **Sample Sayısı**: Performans-kalite dengesini göz önünde bulundurarak 50-100 arası sample kullanın.\n")
            
            # Mode önerisi
            if self.results['mode_analysis']:
                successful_modes = [r for r in self.results['mode_analysis'] if r['success']]
                if successful_modes:
                    mode_perf = defaultdict(list)
                    for result in successful_modes:
                        mode = result['config']['PathIntegralMode']
                        if 'PathIntegralRewardMode' in result['config']:
                            mode += f"_{result['config']['PathIntegralRewardMode']}"
                        mode_perf[mode].append(result['time'])
                    
                    fastest_mode = min(mode_perf.keys(), key=lambda x: np.mean(mode_perf[x]))
                    f.write(f"3. **Mode Seçimi**: En hızlı performans için {fastest_mode} modunu kullanın.\n")
            
            f.write("4. **Genel**: GPU kullanımını etkinleştirin ve yeterli sistem belleği sağlayın.\n\n")
            
            # Sonuç
            f.write("## Sonuç\n\n")
            f.write("Bu analiz, LC0 Path Integral sisteminin çeşitli parametreler altındaki performansını ")
            f.write("detaylı olarak incelemiştir. Sonuçlar, optimal ayarların pozisyon karmaşıklığına ve ")
            f.write("kullanım amacına göre değişebileceğini göstermektedir.\n")
        
        print(f"Performans raporu {report_filename} dosyasına kaydedildi.")
    
    def save_results(self, filename: str = 'path_integral_performance_results.json'):
        """Sonuçları JSON dosyasına kaydet"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        print(f"Sonuçlar {filename} dosyasına kaydedildi.")
    
    def run_full_analysis(self):
        """Tam performans analizini çalıştır"""
        print("LC0 Path Integral Tam Performans Analizi Başlıyor...")
        print("=" * 60)
        
        try:
            # Lambda sensitivity analizi
            self.analyze_lambda_sensitivity()
            
            # Sample scaling analizi
            self.analyze_sample_scaling()
            
            # Mode performance analizi
            self.analyze_mode_performance()
            
            # Position complexity analizi
            self.analyze_position_complexity()
            
            # Görselleştirmeler
            self.create_performance_visualizations()
            
            # Sonuçları kaydet
            self.save_results()
            
            # Rapor oluştur
            self.generate_performance_report()
            
            print("\n" + "=" * 60)
            print("PERFORMANS ANALİZİ TAMAMLANDI!")
            print("=" * 60)
            print("Oluşturulan dosyalar:")
            print("  - path_integral_performance_results.json")
            print("  - path_integral_performance_report.md")
            print("  - path_integral_lambda_analysis.png")
            print("  - path_integral_sample_analysis.png")
            print("  - path_integral_mode_analysis.png")
            
        except KeyboardInterrupt:
            print("\nKullanıcı tarafından iptal edildi.")
        except Exception as e:
            print(f"Beklenmeyen hata: {e}")
            import traceback
            traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='LC0 Path Integral performans analizi')
    parser.add_argument('--lc0-path', required=True, help='LC0 executable path')
    parser.add_argument('--quick', action='store_true', help='Hızlı analiz (daha az test)')
    parser.add_argument('--no-plots', action='store_true', help='Grafik oluşturmayı atla')
    
    args = parser.parse_args()
    
    # Dosya kontrolü
    if not os.path.exists(args.lc0_path):
        print(f"Hata: LC0 executable bulunamadı: {args.lc0_path}")
        sys.exit(1)
    
    # Analyzer oluştur
    analyzer = PathIntegralPerformanceAnalyzer(args.lc0_path)
    
    # Hızlı analiz için parametreleri azalt
    if args.quick:
        analyzer.lambda_values = [0.01, 0.1, 0.5]
        analyzer.sample_counts = [25, 50, 100]
        analyzer.test_positions = analyzer.test_positions[:3]  # İlk 3 pozisyon
    
    # Analizi çalıştır
    analyzer.run_full_analysis()

if __name__ == '__main__':
    main()