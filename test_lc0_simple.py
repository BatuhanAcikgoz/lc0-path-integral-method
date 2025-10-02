#!/usr/bin/env python3
"""
Basit LC0 Path Integral Test Scripti
"""

import chess
import chess.engine
import time

def test_lc0_basic():
    """Temel LC0 testi"""
    lc0_path = "/home/batuhanacikgoz04/Documents/GitHub/lc0-path-integral-method/buildDir/lc0"
    
    print("LC0 motoru başlatılıyor...")
    engine = chess.engine.SimpleEngine.popen_uci(lc0_path)
    
    try:
        # Temel ayarlar
        print("Temel ayarlar uygulanıyor...")
        engine.configure({
            'PathIntegralLambda': 0.1,
            'PathIntegralSamples': 25,
            'PathIntegralMode': 'competitive',
            'PolicyTemperature': 0.7,
            'CPuct': 1.0
        })
        
        # Basit pozisyon testi
        print("Pozisyon analizi başlıyor...")
        board = chess.Board()  # Başlangıç pozisyonu
        
        start_time = time.perf_counter()
        
        # Kısa analiz
        info = engine.analyse(board, chess.engine.Limit(nodes=1000, time=5.0))
        
        end_time = time.perf_counter()
        
        print(f"✓ Analiz tamamlandı!")
        print(f"  Süre: {end_time - start_time:.2f}s")
        print(f"  En iyi hamle: {info.get('pv', [None])[0] if info.get('pv') else 'None'}")
        print(f"  Değerlendirme: {info.get('score')}")
        print(f"  Nodes: {info.get('nodes', 0)}")
        print(f"  NPS: {info.get('nps', 0)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Hata: {e}")
        return False
        
    finally:
        print("Motor kapatılıyor...")
        engine.quit()

def test_lc0_path_integral_modes():
    """Path Integral modlarını test et"""
    lc0_path = "/home/batuhanacikgoz04/Documents/GitHub/lc0-path-integral-method/buildDir/lc0"
    
    print("Path Integral modları test ediliyor...")
    engine = chess.engine.SimpleEngine.popen_uci(lc0_path)
    
    try:
        board = chess.Board()
        
        # Competitive mode test
        print("\n1. Competitive Mode Test:")
        engine.configure({
            'PathIntegralLambda': 0.1,
            'PathIntegralSamples': 10000,  # Küçük sample
            'PathIntegralMode': 'competitive'
        })
        
        info = engine.analyse(board, chess.engine.Limit(nodes=500, time=3.0))
        print(f"  ✓ Competitive: {info.get('pv', [None])[0] if info.get('pv') else 'None'}")
        
        # Quantum limit mode test
        print("\n2. Quantum Limit Mode Test:")
        engine.configure({
            'PathIntegralLambda': 0.1,
            'PathIntegralSamples': 10000,
            'PathIntegralMode': 'quantum_limit',
            'PathIntegralRewardMode': 'policy'
        })
        
        info = engine.analyse(board, chess.engine.Limit(nodes=500, time=3.0))
        print(f"  ✓ Quantum Limit: {info.get('pv', [None])[0] if info.get('pv') else 'None'}")
        
        return True
        
    except Exception as e:
        print(f"✗ Mode test hatası: {e}")
        return False
        
    finally:
        engine.quit()

if __name__ == "__main__":
    print("=" * 50)
    print("LC0 Path Integral Basit Test")
    print("=" * 50)
    
    # Temel test
    if test_lc0_basic():
        print("\n" + "=" * 50)
        # Mode testleri
        test_lc0_path_integral_modes()
    
    print("\nTest tamamlandı!")