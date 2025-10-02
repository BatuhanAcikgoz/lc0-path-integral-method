#!/usr/bin/env python3
"""
Normal LC0 vs engine.py Karşılaştırma Scripti
Path Integral henüz çalışmadığı için normal LC0 ile karşılaştırma
"""

import argparse
import chess
import chess.engine
import json
import time
import numpy as np
import sys
import os

# engine.py'yi import et
sys.path.append('.')
try:
    import engine
    from engine import Engine
    import config
except ImportError as e:
    print(f"engine.py import hatası: {e}")
    sys.exit(1)

class NormalLc0Comparator:
    """Normal LC0 ve engine.py karşılaştırma sınıfı"""
    
    def __init__(self, lc0_path: str):
        self.lc0_path = lc0_path
        self.lc0_engine = None
        self.results = {
            'normal_lc0': {},
            'engine_py': {},
            'comparison': {}
        }
        
        # Test pozisyonları
        self.test_positions = [
            {
                'name': 'opening',
                'fen': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
                'description': 'Başlangıç pozisyonu'
            }
        ]
        
        # Test parametreleri
        self.test_params = [
            {'lambda': 0.1, 'samples': 25, 'mode': 'competitive'},
            {'lambda': 0.1, 'samples': 50, 'mode': 'competitive'},
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
    
    def run_normal_lc0_test(self, position, params):
        """Normal LC0 testi çalıştır"""
        try:
            if self.lc0_engine is None:
                self.start_lc0_engine()
            
            # Normal LC0 ayarları (Path Integral YOK)
            config_options = {
                'PolicyTemperature': 0.7,
                'CPuct': 1.0,
            }
            
            self.lc0_engine.configure(config_options)
            
            board = chess.Board(position['fen'])
            start_time = time.perf_counter()
            
            # Eşit şartlar için aynı nodes limiti
            depth = 4
            nodes_limit = depth * 6500
            
            # Analiz çalıştır
            limit = chess.engine.Limit(nodes=nodes_limit, time=30.0)
            info = self.lc0_engine.analyse(board, limit)
            
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
    
    def run_engine_py_test(self, position, params):
        """engine.py testi çalıştır"""
        try:
            depth = 4
            lam = params['lambda']
            samples = params['samples']
            mode = params['mode']
            reward_mode = params.get('reward_mode', 'hybrid')
            
            start_time = time.perf_counter()
            
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
            
            best_move = None
            if paths and paths[0]:
                best_move = str(paths[0][0]) if paths[0][0] else None
            
            result = {
                'best_move': best_move,
                'evaluation': 0,
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
        """Karşılaştırma testlerini çalıştır"""
        print("Normal LC0 vs engine.py Karşılaştırma Testleri Başlıyor...")
        print("=" * 60)
        
        self.start_lc0_engine()
        
        total_tests = len(self.test_positions) * len(self.test_params)
        current_test = 0
        
        try:
            for position in self.test_positions:
                print(f"\nPozisyon: {position['name']} - {position['description']}")
                print(f"FEN: {position['fen']}")
                print("-" * 50)
                
                self.results['normal_lc0'][position['name']] = {}
                self.results['engine_py'][position['name']] = {}
                
                for params in self.test_params:
                    current_test += 1
                    param_str = f"λ={params['lambda']}, samples={params['samples']}, mode={params['mode']}"
                    
                    print(f"[{current_test}/{total_tests}] Test: {param_str}")
                    
                    # Normal LC0 testi
                    print("  Normal LC0 testi çalıştırılıyor...")
                    lc0_result = self.run_normal_lc0_test(position, params)
                    
                    # engine.py testi
                    print("  engine.py testi çalıştırılıyor...")
                    engine_py_result = self.run_engine_py_test(position, params)
                    
                    # Sonuçları kaydet
                    test_key = f"{params['lambda']}_{params['samples']}_{params['mode']}"
                    
                    self.results['normal_lc0'][position['name']][test_key] = lc0_result
                    self.results['engine_py'][position['name']][test_key] = engine_py_result
                    
                    # Sonuçları yazdır
                    if lc0_result['success'] and engine_py_result['success']:
                        print(f"    Normal LC0: {lc0_result['time']:.2f}s, {lc0_result['nps']:,} nps, hamle: {lc0_result['best_move']}")
                        print(f"    engine.py: {engine_py_result['time']:.2f}s, {engine_py_result['nps']:,} nps, hamle: {engine_py_result['best_move']}")
                        
                        if engine_py_result['time'] > 0:
                            speedup = engine_py_result['time'] / lc0_result['time']
                            print(f"    Hızlanma: {speedup:.2f}x (LC0 lehine)")
                    else:
                        if not lc0_result['success']:
                            print(f"    LC0 hatası: {lc0_result['error']}")
                        if not engine_py_result['success']:
                            print(f"    engine.py hatası: {engine_py_result['error']}")
                    
                    time.sleep(1)
        
        finally:
            self.stop_lc0_engine()
    
    def save_results(self, filename: str = 'normal_lc0_comparison_results.json'):
        """Sonuçları kaydet"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"Sonuçlar {filename} dosyasına kaydedildi.")

def main():
    parser = argparse.ArgumentParser(description='Normal LC0 vs engine.py karşılaştırması')
    parser.add_argument('--lc0-path', required=True, help='LC0 executable path')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.lc0_path):
        print(f"Hata: LC0 executable bulunamadı: {args.lc0_path}")
        sys.exit(1)
    
    comparator = NormalLc0Comparator(args.lc0_path)
    
    try:
        comparator.run_comparison_tests()
        comparator.save_results()
        
        print("\n" + "=" * 60)
        print("KARŞILAŞTIRMA TAMAMLANDI!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\nKullanıcı tarafından iptal edildi.")
    except Exception as e:
        print(f"Beklenmeyen hata: {e}")
        import traceback
        traceback.print_exc()
    finally:
        comparator.stop_lc0_engine()

if __name__ == '__main__':
    main()