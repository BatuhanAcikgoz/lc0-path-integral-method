#!/usr/bin/env python3
"""
Path Integral Final Test
"""

import chess
import chess.engine
import time

def test_path_integral_working():
    """Path Integral'ın çalışıp çalışmadığını test et"""
    lc0_path = "./buildDir/lc0"
    
    print("Path Integral LC0 motoru başlatılıyor...")
    
    try:
        engine = chess.engine.SimpleEngine.popen_uci(lc0_path)
        
        # Path Integral ayarları
        print("Path Integral ayarları uygulanıyor...")
        engine.configure({
            'PathIntegralLambda': 0.1,
            'PathIntegralSamples': 10000,  # Küçük sample hızlı test için
            'PathIntegralMode': 'competitive'
        })
        
        # Test
        print("Analiz başlıyor...")
        board = chess.Board()
        
        start_time = time.perf_counter()
        info = engine.analyse(board, chess.engine.Limit(nodes=100000, time=3.0))
        end_time = time.perf_counter()
        
        print(f"Analiz tamamlandı: {end_time - start_time:.2f}s")
        print(f"En iyi hamle: {info.get('pv', [None])[0] if info.get('pv') else 'None'}")
        print(f"Nodes: {info.get('nodes', 0)}")
        
        # Sonuç kontrolü
        if info.get('pv') and len(info['pv']) > 0:
            print("✅ Path Integral çalışıyor!")
            return True
        else:
            print("❌ Path Integral çalışmıyor")
            return False
            
    except Exception as e:
        print(f"Hata: {e}")
        return False
    finally:
        try:
            engine.quit()
        except:
            pass

if __name__ == "__main__":
    test_path_integral_working()