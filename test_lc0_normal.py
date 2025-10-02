#!/usr/bin/env python3
"""
LC0 Normal Mode Test - Path Integral olmadan
"""

import chess
import chess.engine
import time

def test_lc0_normal():
    """LC0'ı normal modda test et"""
    lc0_path = "/home/batuhanacikgoz04/Documents/GitHub/lc0-path-integral-method/buildDir/lc0"
    
    print("LC0 motoru başlatılıyor (Normal Mode)...")
    engine = chess.engine.SimpleEngine.popen_uci(lc0_path)
    
    try:
        # Path Integral KAPALI - sadece normal LC0 ayarları
        print("Normal LC0 ayarları uygulanıyor...")
        engine.configure({
            # Path Integral ayarlarını VERME - varsayılan olarak kapalı olmalı
            'PolicyTemperature': 0.7,
            'CPuct': 1.0
        })
        
        # Basit pozisyon testi
        print("Pozisyon analizi başlıyor...")
        board = chess.Board()  # Başlangıç pozisyonu
        
        start_time = time.perf_counter()
        
        # Normal analiz
        info = engine.analyse(board, chess.engine.Limit(nodes=1000, time=5.0))
        
        end_time = time.perf_counter()
        
        print(f"✓ Normal LC0 analizi tamamlandı!")
        print(f"  Süre: {end_time - start_time:.2f}s")
        print(f"  En iyi hamle: {info.get('pv', [None])[0] if info.get('pv') else 'None'}")
        print(f"  Değerlendirme: {info.get('score')}")
        print(f"  Nodes: {info.get('nodes', 0)}")
        print(f"  NPS: {info.get('nps', 0)}")
        
        # Hamle kontrolü
        if info.get('pv') and len(info['pv']) > 0:
            print(f"✅ LC0 normal modda düzgün çalışıyor!")
            return True
        else:
            print(f"❌ LC0 normal modda da hamle döndürmüyor!")
            return False
        
    except Exception as e:
        print(f"✗ Hata: {e}")
        return False
        
    finally:
        print("Motor kapatılıyor...")
        engine.quit()

if __name__ == "__main__":
    print("=" * 50)
    print("LC0 Normal Mode Test")
    print("=" * 50)
    
    test_lc0_normal()