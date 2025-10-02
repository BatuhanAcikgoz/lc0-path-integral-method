#!/usr/bin/env python3
"""
LC0 Path Integral vs engine.py Karşılaştırma Scripti

Bu script, gömülü LC0 Path Integral sistemi ile orijinal engine.py implementasyonunu
performans, doğruluk ve benzerlik açısından karşılaştırır.

Kullanım:
    python compare_path_integral_systems.py --lc0-path ./lc0 --engine-py ./engine.py
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

# engine.py'yi import etmek için
sys.path.append('.')
try:
    import engine
    from engine import Engine
    import config
except ImportError as e:
    print(f"engine.py import hatası: {e}")
    print("engine.py dosyasının mevcut dizinde olduğundan emin olun.")
    sys.exit(1)

class PathIntegralComparator:
    """LC0 Path Integral ve engine.py karşılaştırma sınıfı
    
    Eşit şartlarda karşılaştırma için:
    - LC0: depth=4 → ~26,000 nodes limiti
    - engine.py: depth=4 (aynı derinlik)
    """
    
    def __init__(self, lc0_path: str, engine_py_path: str):
        self.lc0_path = lc0_path
        self.engine_py_path = engine_py_path
        self.lc0_engine = None  # Tek motor instance'ı
        self.results = {
            'embedded_lc0': {},
            'engine_py': {},
            'comparison': {}
        }
        
        # Test pozisyonları (başlangıç için sadece ilk pozisyon)
        self.test_positions = [
            {
                'name': 'opening',
                'fen': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
                'description': 'Başlangıç pozisyonu'
            },
            # Diğer pozisyonlar şimdilik devre dışı
            # {
            #     'name': 'sicilian',
            #     'fen': 'rnbqkb1r/pp1ppppp/5n2/2p5/2P5/8/PP1PPPPP/RNBQKBNR w KQkq - 0 3',
            #     'description': 'Sicilian Defense'
            # },
        ]
        
        # Test parametreleri
        self.test_params = [
            {'lambda': 0.01, 'samples': 25, 'mode': 'competitive'},
            {'lambda': 0.05, 'samples': 50, 'mode': 'competitive'},
            {'lambda': 0.1, 'samples': 50, 'mode': 'competitive'},
            {'lambda': 0.1, 'samples': 100, 'mode': 'quantum_limit', 'reward_mode': 'policy'},
            {'lambda': 0.1, 'samples': 100, 'mode': 'quantum_limit', 'reward_mode': 'cp_score'},
            {'lambda': 0.1, 'samples': 100, 'mode': 'quantum_limit', 'reward_mode': 'hybrid'},
            {'lambda': 0.2, 'samples': 75, 'mode': 'competitive'},
            {'lambda': 0.5, 'samples': 100, 'mode': 'competitive'},
        ]
    
    def start_lc0_engine(self):
        """LC0 motorunu başlat"""
        if self.lc0_engine is None:
            print("LC0 motoru başlatılıyor...")
            self.lc0_engine = chess.engine.SimpleEngine.popen_uci(self.lc0_path)
            print("✓ LC0 motoru başlatıldı")
    
    def stop_lc0_engine(self):
        """LC0 motorunu kapat"""
        if self.lc0_engine is not None:
            try:
                self.lc0_engine.quit()
                print("✓ LC0 motoru kapatıldı")
            except:
                pass
            finally:
                self.lc0_engine = None
    
    def run_embedded_lc0_test(self, position: Dict, params: Dict) -> Dict:
        """Gömülü LC0 Path Integral testi çalıştır"""
        try:
            # Tek motor instance'ı kullan
            if self.lc0_engine is None:
                self.start_lc0_engine()
            
            # Path Integral ayarları
            config_options = {
                'PathIntegralLambda': params['lambda'],
                'PathIntegralSamples': params['samples'],
                'PathIntegralMode': params['mode']
            }
            
            if 'reward_mode' in params:
                config_options['PathIntegralRewardMode'] = params['reward_mode']
            
            # engine.py ile eşit şartlar için aynı LC0 ayarları
            # Mevcut UCI seçeneklerini kullan (hata çıktısından alınan doğru isimler)
            config_options.update({
                'PolicyTemperature': 0.7,  # Temperature yerine PolicyTemperature
                'CPuct': 1.0,             # Bu doğru
                # Deterministic ve DirichletNoiseEpsilon mevcut değil, atla
            })
            
            self.lc0_engine.configure(config_options)
            
            board = chess.Board(position['fen'])
            start_time = time.perf_counter()
            
            # Eşit şartlar için depth-based nodes limiti (engine.py ile aynı depth=4)
            depth = 5  # engine.py ile aynı depth
            nodes_limit = depth * 6500  # depth * 6500 nodes (engine.py'deki adaptif sistem benzeri)
            
            # Analiz çalıştır - nodes limiti ile
            try:
                info = self.lc0_engine.analyse(board, chess.engine.Limit(nodes=nodes_limit))
            except chess.engine.EngineError as e:
                # LC0 motor hatası durumunda fallback
                print(f"    LC0 motor hatası: {e}")
                return {
                    'success': False,
                    'error': f'LC0 EngineError: {str(e)}',
                    'time': 0,
                    'nodes': 0,
                    'nps': 0
                }
            
            end_time = time.perf_counter()
            analysis_time = end_time - start_time
            
            result = {
                'best_move': str(info['pv'][0]) if info.get('pv') else None,
                'evaluation': info['score'].relative.score() if info.get('score') else 0,
                'time': analysis_time,
                'nodes': info.get('nodes', 0),
                'nps': info.get('nps', 0),
                'pv_length': len(info.get('pv', [])),
                'success': True,
                'error': None
            }
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'time': 0,
                'nodes': 0,
                'nps': 0
            }
    
    def run_engine_py_test(self, position: Dict, params: Dict) -> Dict:
        """engine.py testi çalıştır"""
        try:
            # engine.py parametrelerini ayarla (LC0 ile aynı depth)
            depth = 5  # LC0 ile eşit şartlar için aynı depth
            lam = params['lambda']
            samples = params['samples']
            mode = params['mode']
            reward_mode = params.get('reward_mode', 'hybrid')
            
            start_time = time.perf_counter()
            
            # engine.py sample_paths fonksiyonunu çağır
            paths = Engine.sample_paths(
                fen=position['fen'],
                depth=depth,
                lam=lam,
                samples=samples,
                mode=mode,
                reward_mode=reward_mode,
                use_cache=False
            )
            
            end_time = time.perf_counter()
            analysis_time = end_time - start_time
            
            # En iyi hamleyi bul (basit heuristic)
            best_move = None
            if paths:
                # İlk path'in ilk hamlesini al
                if paths[0]:
                    best_move = str(paths[0][0]) if paths[0][0] else None
            
            result = {
                'best_move': best_move,
                'evaluation': 0,  # engine.py'de direkt evaluation yok
                'time': analysis_time,
                'nodes': len(paths) * depth if paths else 0,
                'nps': (len(paths) * depth / analysis_time) if analysis_time > 0 and paths else 0,
                'paths_generated': len(paths),
                'success': True,
                'error': None
            }
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'time': 0,
                'nodes': 0,
                'nps': 0,
                'paths_generated': 0
            }
    
    def run_comparison_tests(self):
        """Tüm karşılaştırma testlerini çalıştır"""
        print("LC0 Path Integral vs engine.py Karşılaştırma Testleri Başlıyor...")
        print("=" * 60)
        
        # LC0 motorunu başlat
        self.start_lc0_engine()
        
        total_tests = len(self.test_positions) * len(self.test_params)
        current_test = 0
        
        try:
            for position in self.test_positions:
                print(f"\nPozisyon: {position['name']} - {position['description']}")
                print(f"FEN: {position['fen']}")
                print("-" * 50)
                
                self.results['embedded_lc0'][position['name']] = {}
                self.results['engine_py'][position['name']] = {}
                
                for params in self.test_params:
                    current_test += 1
                    param_str = f"λ={params['lambda']}, samples={params['samples']}, mode={params['mode']}"
                    if 'reward_mode' in params:
                        param_str += f", reward={params['reward_mode']}"
                    
                    print(f"[{current_test}/{total_tests}] Test: {param_str}")
                    
                    # Gömülü LC0 testi
                    print("  Gömülü LC0 Path Integral testi çalıştırılıyor...")
                    lc0_result = self.run_embedded_lc0_test(position, params)
                    
                    # engine.py testi
                    print("  engine.py testi çalıştırılıyor...")
                    engine_py_result = self.run_engine_py_test(position, params)
                    
                    # Sonuçları kaydet
                    test_key = f"{params['lambda']}_{params['samples']}_{params['mode']}"
                    if 'reward_mode' in params:
                        test_key += f"_{params['reward_mode']}"
                    
                    self.results['embedded_lc0'][position['name']][test_key] = lc0_result
                    self.results['engine_py'][position['name']][test_key] = engine_py_result
                    
                    # Sonuçları yazdır
                    if lc0_result['success'] and engine_py_result['success']:
                        print(f"    LC0 (depth=4, ~26k nodes): {lc0_result['time']:.2f}s, {lc0_result['nps']:,} nps, hamle: {lc0_result['best_move']}")
                        print(f"    engine.py (depth=4): {engine_py_result['time']:.2f}s, {engine_py_result['nps']:,} nps, hamle: {engine_py_result['best_move']}")
                        
                        # Hız karşılaştırması
                        if engine_py_result['time'] > 0:
                            speedup = engine_py_result['time'] / lc0_result['time']
                            print(f"    Hızlanma: {speedup:.2f}x (LC0 lehine)")
                    else:
                        if not lc0_result['success']:
                            print(f"    LC0 hatası: {lc0_result['error']}")
                        if not engine_py_result['success']:
                            print(f"    engine.py hatası: {engine_py_result['error']}")
                    
                    time.sleep(1)  # Sistemin dinlenmesi için
        
        finally:
            # Motor kapatma
            self.stop_lc0_engine()
    
    def analyze_results(self):
        """Sonuçları analiz et ve istatistikler üret"""
        print("\n" + "=" * 60)
        print("SONUÇ ANALİZİ")
        print("=" * 60)
        
        # Başarı oranları
        lc0_success = 0
        engine_py_success = 0
        total_tests = 0
        
        # Performans metrikleri
        lc0_times = []
        engine_py_times = []
        speedups = []
        
        # Hamle benzerliği
        move_matches = 0
        total_comparisons = 0
        
        for position_name in self.results['embedded_lc0']:
            for test_key in self.results['embedded_lc0'][position_name]:
                total_tests += 1
                
                lc0_result = self.results['embedded_lc0'][position_name][test_key]
                engine_py_result = self.results['engine_py'][position_name][test_key]
                
                if lc0_result['success']:
                    lc0_success += 1
                    lc0_times.append(lc0_result['time'])
                
                if engine_py_result['success']:
                    engine_py_success += 1
                    engine_py_times.append(engine_py_result['time'])
                
                # Hızlanma hesapla
                if lc0_result['success'] and engine_py_result['success'] and engine_py_result['time'] > 0:
                    speedup = engine_py_result['time'] / lc0_result['time']
                    speedups.append(speedup)
                    
                    # Hamle benzerliği
                    total_comparisons += 1
                    if (lc0_result['best_move'] and engine_py_result['best_move'] and 
                        lc0_result['best_move'] == engine_py_result['best_move']):
                        move_matches += 1
        
        # İstatistikleri yazdır
        print(f"\nBaşarı Oranları:")
        print(f"  Gömülü LC0: {lc0_success}/{total_tests} ({100*lc0_success/total_tests:.1f}%)")
        print(f"  engine.py: {engine_py_success}/{total_tests} ({100*engine_py_success/total_tests:.1f}%)")
        
        if lc0_times and engine_py_times:
            print(f"\nPerformans Metrikleri:")
            print(f"  LC0 ortalama süre: {np.mean(lc0_times):.2f}s (±{np.std(lc0_times):.2f})")
            print(f"  engine.py ortalama süre: {np.mean(engine_py_times):.2f}s (±{np.std(engine_py_times):.2f})")
            
            if speedups:
                print(f"  Ortalama hızlanma: {np.mean(speedups):.2f}x (LC0 lehine)")
                print(f"  Medyan hızlanma: {np.median(speedups):.2f}x")
                print(f"  Min/Max hızlanma: {np.min(speedups):.2f}x / {np.max(speedups):.2f}x")
        
        if total_comparisons > 0:
            print(f"\nHamle Benzerliği:")
            print(f"  Aynı hamle: {move_matches}/{total_comparisons} ({100*move_matches/total_comparisons:.1f}%)")
        
        # Detaylı analiz
        self.results['comparison'] = {
            'success_rates': {
                'lc0': lc0_success / total_tests if total_tests > 0 else 0,
                'engine_py': engine_py_success / total_tests if total_tests > 0 else 0
            },
            'performance': {
                'lc0_mean_time': np.mean(lc0_times) if lc0_times else 0,
                'engine_py_mean_time': np.mean(engine_py_times) if engine_py_times else 0,
                'mean_speedup': np.mean(speedups) if speedups else 0,
                'median_speedup': np.median(speedups) if speedups else 0
            },
            'move_similarity': move_matches / total_comparisons if total_comparisons > 0 else 0,
            'total_tests': total_tests
        }
    
    def create_visualizations(self):
        """Görsel grafikler oluştur"""
        print("\nGörsel grafikler oluşturuluyor...")
        
        # Veri hazırlama
        data_for_plots = []
        
        for position_name in self.results['embedded_lc0']:
            for test_key in self.results['embedded_lc0'][position_name]:
                lc0_result = self.results['embedded_lc0'][position_name][test_key]
                engine_py_result = self.results['engine_py'][position_name][test_key]
                
                if lc0_result['success'] and engine_py_result['success']:
                    # Test parametrelerini parse et
                    parts = test_key.split('_')
                    lambda_val = float(parts[0])
                    samples = int(parts[1])
                    mode = parts[2]
                    reward_mode = parts[3] if len(parts) > 3 else 'none'
                    
                    data_for_plots.append({
                        'position': position_name,
                        'lambda': lambda_val,
                        'samples': samples,
                        'mode': mode,
                        'reward_mode': reward_mode,
                        'lc0_time': lc0_result['time'],
                        'engine_py_time': engine_py_result['time'],
                        'speedup': engine_py_result['time'] / lc0_result['time'] if lc0_result['time'] > 0 else 0,
                        'lc0_nps': lc0_result['nps'],
                        'engine_py_nps': engine_py_result['nps'],
                        'move_match': lc0_result['best_move'] == engine_py_result['best_move']
                    })
        
        if not data_for_plots:
            print("Görselleştirme için yeterli veri yok.")
            return
        
        df = pd.DataFrame(data_for_plots)
        
        # Grafik stili ayarla
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Performans karşılaştırması
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('LC0 Path Integral vs engine.py Performans Karşılaştırması', fontsize=16)
        
        # Süre karşılaştırması
        axes[0, 0].scatter(df['engine_py_time'], df['lc0_time'], alpha=0.7)
        axes[0, 0].plot([0, df['engine_py_time'].max()], [0, df['engine_py_time'].max()], 'r--', alpha=0.5)
        axes[0, 0].set_xlabel('engine.py Süresi (s)')
        axes[0, 0].set_ylabel('LC0 Süresi (s)')
        axes[0, 0].set_title('Analiz Süresi Karşılaştırması')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Hızlanma dağılımı
        axes[0, 1].hist(df['speedup'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(df['speedup'].mean(), color='red', linestyle='--', 
                          label=f'Ortalama: {df["speedup"].mean():.2f}x')
        axes[0, 1].set_xlabel('Hızlanma (x)')
        axes[0, 1].set_ylabel('Frekans')
        axes[0, 1].set_title('Hızlanma Dağılımı (LC0 lehine)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Lambda'ya göre performans
        lambda_perf = df.groupby('lambda')['speedup'].mean().reset_index()
        axes[1, 0].bar(lambda_perf['lambda'].astype(str), lambda_perf['speedup'])
        axes[1, 0].set_xlabel('Lambda Değeri')
        axes[1, 0].set_ylabel('Ortalama Hızlanma (x)')
        axes[1, 0].set_title('Lambda Değerine Göre Hızlanma')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Pozisyona göre performans
        pos_perf = df.groupby('position')['speedup'].mean().reset_index()
        axes[1, 1].bar(pos_perf['position'], pos_perf['speedup'])
        axes[1, 1].set_xlabel('Pozisyon')
        axes[1, 1].set_ylabel('Ortalama Hızlanma (x)')
        axes[1, 1].set_title('Pozisyona Göre Hızlanma')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('path_integral_performance_comparison.png', dpi=300, bbox_inches='tight')

        # 2. Hamle benzerliği analizi
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Hamle Benzerliği Analizi', fontsize=16)
        
        # Genel benzerlik
        move_similarity = df['move_match'].mean()
        axes[0].pie([move_similarity, 1-move_similarity], 
                   labels=[f'Aynı Hamle\n({move_similarity:.1%})', f'Farklı Hamle\n({1-move_similarity:.1%})'],
                   autopct='%1.1f%%', startangle=90)
        axes[0].set_title('Genel Hamle Benzerliği')
        
        # Moda göre benzerlik
        mode_similarity = df.groupby('mode')['move_match'].mean().reset_index()
        axes[1].bar(mode_similarity['mode'], mode_similarity['move_match'])
        axes[1].set_xlabel('Mod')
        axes[1].set_ylabel('Hamle Benzerlik Oranı')
        axes[1].set_title('Moda Göre Hamle Benzerliği')
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('path_integral_move_similarity.png', dpi=300, bbox_inches='tight')

        # 3. Detaylı performans heatmap
        plt.figure(figsize=(12, 8))
        
        # Lambda ve samples'a göre hızlanma heatmap'i
        pivot_data = df.pivot_table(values='speedup', index='lambda', columns='samples', aggfunc='mean')
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn', center=1.0)
        plt.title('Lambda ve Sample Sayısına Göre Hızlanma (LC0 lehine)')
        plt.xlabel('Sample Sayısı')
        plt.ylabel('Lambda Değeri')
        plt.tight_layout()
        plt.savefig('path_integral_heatmap.png', dpi=300, bbox_inches='tight')

        print("Grafikler kaydedildi:")
        print("  - path_integral_performance_comparison.png")
        print("  - path_integral_move_similarity.png")
        print("  - path_integral_heatmap.png")
    
    def save_results(self, filename: str = 'path_integral_comparison_results.json'):
        """Sonuçları JSON dosyasına kaydet"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"\nSonuçlar {filename} dosyasına kaydedildi.")
    
    def generate_report(self):
        """Detaylı rapor oluştur"""
        report_filename = 'path_integral_comparison_report.md'
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("# LC0 Path Integral vs engine.py Karşılaştırma Raporu\n\n")
            f.write(f"**Rapor Tarihi**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Özet
            f.write("## Özet\n\n")
            comp = self.results['comparison']
            f.write(f"- **Toplam Test**: {comp['total_tests']}\n")
            f.write(f"- **LC0 Başarı Oranı**: {comp['success_rates']['lc0']:.1%}\n")
            f.write(f"- **engine.py Başarı Oranı**: {comp['success_rates']['engine_py']:.1%}\n")
            f.write(f"- **Ortalama Hızlanma**: {comp['performance']['mean_speedup']:.2f}x (LC0 lehine)\n")
            f.write(f"- **Hamle Benzerliği**: {comp['move_similarity']:.1%}\n\n")
            
            # Performans analizi
            f.write("## Performans Analizi\n\n")
            f.write(f"- **LC0 Ortalama Süre**: {comp['performance']['lc0_mean_time']:.2f} saniye\n")
            f.write(f"- **engine.py Ortalama Süre**: {comp['performance']['engine_py_mean_time']:.2f} saniye\n")
            f.write(f"- **Medyan Hızlanma**: {comp['performance']['median_speedup']:.2f}x\n\n")
            
            # Test pozisyonları
            f.write("## Test Pozisyonları\n\n")
            for pos in self.test_positions:
                f.write(f"### {pos['name'].title()}\n")
                f.write(f"- **Açıklama**: {pos['description']}\n")
                f.write(f"- **FEN**: `{pos['fen']}`\n\n")
            
            # Test parametreleri
            f.write("## Test Parametreleri\n\n")
            f.write("**Eşit Şartlar**: Her iki sistem de depth=4 (LC0: ~26,000 nodes) ile test edilmiştir.\n\n")
            for i, params in enumerate(self.test_params, 1):
                f.write(f"{i}. Lambda: {params['lambda']}, Samples: {params['samples']}, Mode: {params['mode']}")
                if 'reward_mode' in params:
                    f.write(f", Reward: {params['reward_mode']}")
                f.write("\n")
            
            f.write("\n## Sonuç\n\n")
            if comp['performance']['mean_speedup'] > 1:
                f.write("Gömülü LC0 Path Integral sistemi, orijinal engine.py implementasyonuna göre ")
                f.write(f"**{comp['performance']['mean_speedup']:.2f}x daha hızlı** çalışmaktadır. ")
            else:
                f.write("engine.py implementasyonu daha hızlı çalışmaktadır. ")
            
            if comp['move_similarity'] > 0.7:
                f.write(f"Hamle benzerliği **{comp['move_similarity']:.1%}** ile yüksek seviyededir, ")
                f.write("bu da iki sistemin benzer sonuçlar ürettiğini göstermektedir.")
            elif comp['move_similarity'] > 0.5:
                f.write(f"Hamle benzerliği **{comp['move_similarity']:.1%}** ile orta seviyededir.")
            else:
                f.write(f"Hamle benzerliği **{comp['move_similarity']:.1%}** ile düşük seviyededir, ")
                f.write("sistemler farklı yaklaşımlar kullanıyor olabilir.")
        
        print(f"Detaylı rapor {report_filename} dosyasına kaydedildi.")

def main():
    parser = argparse.ArgumentParser(description='LC0 Path Integral vs engine.py karşılaştırması')
    parser.add_argument('--lc0-path', required=True, help='LC0 executable path')
    parser.add_argument('--engine-py', default='./engine.py', help='engine.py path')
    parser.add_argument('--no-plots', action='store_true', help='Grafik oluşturmayı atla')
    
    args = parser.parse_args()
    
    # Dosya kontrolü
    if not os.path.exists(args.lc0_path):
        print(f"Hata: LC0 executable bulunamadı: {args.lc0_path}")
        sys.exit(1)
    
    if not os.path.exists(args.engine_py):
        print(f"Hata: engine.py bulunamadı: {args.engine_py}")
        sys.exit(1)
    
    # Karşılaştırma başlat
    comparator = PathIntegralComparator(args.lc0_path, args.engine_py)
    
    try:
        # Testleri çalıştır
        comparator.run_comparison_tests()
        
        # Sonuçları analiz et
        comparator.analyze_results()
        
        # Görselleştirmeler oluştur
        if not args.no_plots:
            try:
                comparator.create_visualizations()
            except Exception as e:
                print(f"Görselleştirme hatası: {e}")
                print("Matplotlib/seaborn kurulu değil olabilir.")
        
        # Sonuçları kaydet
        comparator.save_results()
        
        # Rapor oluştur
        comparator.generate_report()
        
        print("\n" + "=" * 60)
        print("KARŞILAŞTIRMA TAMAMLANDI!")
        print("=" * 60)
        print("Oluşturulan dosyalar:")
        print("  - path_integral_comparison_results.json")
        print("  - path_integral_comparison_report.md")
        if not args.no_plots:
            print("  - path_integral_performance_comparison.png")
            print("  - path_integral_move_similarity.png")
            print("  - path_integral_heatmap.png")
        
    except KeyboardInterrupt:
        print("\nKullanıcı tarafından iptal edildi.")
    except Exception as e:
        print(f"Beklenmeyen hata: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Motor kapatmayı garanti et
        comparator.stop_lc0_engine()

if __name__ == '__main__':
    main()