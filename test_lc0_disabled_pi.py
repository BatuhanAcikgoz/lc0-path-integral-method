#!/usr/bin/env python3
"""
LC0 Test - Path Integral Tamamen Devre Dışı
"""

import chess
import chess.engine
import time

def test_lc0_disabled_pi():
    """LC0'ı Path Integral devre dışı bırakarak test et"""
    lc0_path = "/home/batuhanacikgoz04/Documents/GitHub/lc0-path-integral-method/buildDir/lc0"
    
    print("LC0 motoru başlatılıyor (Path Integral KAPALI)...")
    engine = chess.engine.SimpleEngine.popen_uci(lc0_path)
    
    try:
        # Path Integral'ı KAPALI ayarları
        print("Path Integral devre dışı bırakılıyor...")
        engine.configure({
            'PathIntegralLambda': 0.0,      # Lambda = 0 -> devre dışı
            'PathIntegralSamples': 0,       # Samples = 0 -> devre dışı
            'PathIntegralMode': 'competitive',  # Mode ayarı (ama lambda=0 olduğu için etkisiz)
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
        
        print(f"✓ LC0 analizi tamamlandı!")
        print(f"  Süre: {end_time - start_time:.2f}s")
        print(f"  En iyi hamle: {info.get('pv', [None])[0] if info.get('pv') else 'None'}")
        print(f"  Değerlendirme: {info.get('score')}")
        print(f"  Nodes: {info.get('nodes', 0)}")
        print(f"  NPS: {info.get('nps', 0)}")
        
        # Hamle kontrolü
        if info.get('pv') and len(info['pv']) > 0:
            print(f"✅ LC0 Path Integral kapalıyken düzgün çalışıyor!")
            return True
        else:
            print(f"❌ LC0 hala hamle döndürmüyor!")
            return False
        
    except Exception as e:
        print(f"✗ Hata: {e}")
        return False
        
    finally:
        print("Motor kapatılıyor...")
        engine.quit()

def test_different_pi_settings():
    """Farklı Path Integral ayarlarını test et"""
    lc0_path = "/home/batuhanacikgoz04/Documents/GitHub/lc0-path-integral-method/buildDir/lc0"
    
    settings = [
        {'name': 'Lambda=0, Samples=0', 'PathIntegralLambda': 0.0, 'PathIntegralSamples': 0},
        {'name': 'Lambda=0, Samples=1', 'PathIntegralLambda': 0.0, 'PathIntegralSamples': 1},
        {'name': 'Lambda=0.001, Samples=1', 'PathIntegralLambda': 0.001, 'PathIntegralSamples': 1},
    ]
    
    for setting in settings:
        print(f"\n--- Test: {setting['name']} ---")
        engine = chess.engine.SimpleEngine.popen_uci(lc0_path)
        
        try:
            engine.configure({
                'PathIntegralLambda': setting['PathIntegralLambda'],
                'PathIntegralSamples': setting['PathIntegralSamples'],
                'PathIntegralMode': 'competitive'
            })
            
            board = chess.Board()
            info = engine.analyse(board, chess.engine.Limit(nodes=100, time=2.0))
            
            best_move = info.get('pv', [None])[0] if info.get('pv') else None
            nodes = info.get('nodes', 0)
            
            if best_move and nodes > 0:
                print(f"✅ {setting['name']}: Çalışıyor - {best_move}, {nodes} nodes")
            else:
                print(f"❌ {setting['name']}: Çalışmıyor - {best_move}, {nodes} nodes")
                
        except Exception as e:
            print(f"❌ {setting['name']}: Hata - {e}")
        finally:
            engine.quit()

if __name__ == "__main__":
    print("=" * 60)
    print("LC0 Path Integral Devre Dışı Test")
    print("=" * 60)
    
    # Ana test
    if test_lc0_disabled_pi():
        print("\n" + "=" * 60)
        print("Farklı Path Integral Ayarları Test Ediliyor...")
        test_different_pi_settings()
    
    print("\nTest tamamlandı!")