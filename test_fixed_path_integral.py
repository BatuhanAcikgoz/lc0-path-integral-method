#!/usr/bin/env python3
"""
Düzeltilmiş Path Integral LC0 Test
"""

import chess
import chess.engine
import time

def test_fixed_path_integral():
    """Düzeltilmiş Path Integral LC0'ı test et"""
    lc0_path = "./buildDir/lc0"
    
    print("Düzeltilmiş Path Integral LC0 motoru başlatılıyor...")
    engine = chess.engine.SimpleEngine.popen_uci(lc0_path)
    
    try:
        # Path Integral ayarları
        print("Path Integral ayarları uygulanıyor...")
        engine.configure({
            'PathIntegralLambda': 0.1,
            'PathIntegralSamples': 10000,
            'PathIntegralMode': 'competitive',
            'PolicyTemperature': 0.7,
            'CPuct': 1.0
        })
        
        # Basit pozisyon testi
        print("Pozisyon analizi başlıyor...")
        board = chess.Board()  # Başlangıç pozisyonu
        
        start_time = time.perf_counter()
        
        # Analiz
        info = engine.analyse(board, chess.engine.Limit(nodes=100000, time=5.0))
        
        end_time = time.perf_counter()
        
        print(f"✓ Path Integral LC0 analizi tamamlandı!")
        print(f"  Süre: {end_time - start_time:.2f}s")
        print(f"  En iyi hamle: {info.get('pv', [None])[0] if info.get('pv') else 'None'}")
        print(f"  Değerlendirme: {info.get('score')}")
        print(f"  Nodes: {info.get('nodes', 0)}")
        print(f"  NPS: {info.get('nps', 0)}")
        
        # Hamle kontrolü
        if info.get('pv') and len(info['pv']) > 0:
            print(f"✅ Path Integral LC0 düzgün çalışıyor!")
            print(f"✅ Path Integral etkin ama LC0 search sistemi çalışıyor!")
            return True
        else:
            print(f"❌ Hala hamle döndürmüyor!")
            return False
        
    except Exception as e:
        print(f"✗ Hata: {e}")
        return False
        
    finally:
        print("Motor kapatılıyor...")
        engine.quit()

def test_different_modes():
    """Farklı Path Integral modlarını test et"""
    lc0_path = "./buildDir/lc0"
    
    modes = [
        {'name': 'Competitive', 'PathIntegralMode': 'competitive'},
        {'name': 'Quantum Limit', 'PathIntegralMode': 'quantum_limit', 'PathIntegralRewardMode': 'hybrid'}
    ]
    
    for mode_config in modes:
        print(f"\n--- {mode_config['name']} Mode Test ---")
        engine = chess.engine.SimpleEngine.popen_uci(lc0_path)
        
        try:
            config = {
                'PathIntegralLambda': 0.1,
                'PathIntegralSamples': 10000,  # Küçük sample hızlı test için
            }
            config.update(mode_config)
            del config['name']  # name key'ini kaldır
            
            engine.configure(config)
            
            board = chess.Board()
            info = engine.analyse(board, chess.engine.Limit(nodes=100000, time=3.0))
            
            best_move = info.get('pv', [None])[0] if info.get('pv') else None
            nodes = info.get('nodes', 0)
            
            if best_move and nodes > 0:
                print(f"✅ {mode_config['name']}: Çalışıyor - {best_move}, {nodes} nodes")
            else:
                print(f"❌ {mode_config['name']}: Çalışmıyor - {best_move}, {nodes} nodes")
                
        except Exception as e:
            print(f"❌ {mode_config['name']}: Hata - {e}")
        finally:
            engine.quit()

if __name__ == "__main__":
    print("=" * 60)
    print("Düzeltilmiş Path Integral LC0 Test")
    print("=" * 60)
    
    # Ana test
    if test_fixed_path_integral():
        print("\n" + "=" * 60)
        print("Farklı Path Integral Modları Test Ediliyor...")
        test_different_modes()
    
    print("\nTest tamamlandı!")