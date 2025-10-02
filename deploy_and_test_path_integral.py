#!/usr/bin/env python3
"""
LC0 Path Integral Deployment ve Test Scripti

Bu script, LC0 Path Integral sisteminin derlenmesi, kurulumu ve kapsamlı testlerini otomatikleştirir.

Kullanım:
    python deploy_and_test_path_integral.py --build --test --compare
"""

import argparse
import subprocess
import sys
import os
import json
import time
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any
import tempfile

class PathIntegralDeployer:
    """LC0 Path Integral deployment ve test sınıfı"""
    
    def __init__(self, source_dir: str = ".", build_dir: str = "builddir"):
        self.source_dir = Path(source_dir).resolve()
        self.build_dir = Path(build_dir).resolve()
        self.test_results = {}
        
        # Test konfigürasyonları
        self.test_configs = [
            {
                'name': 'basic_competitive',
                'config': {
                    'PathIntegralMode': 'competitive',
                    'PathIntegralLambda': 0.1,
                    'PathIntegralSamples': 50
                },
                'expected_success': True
            },
            {
                'name': 'quantum_policy',
                'config': {
                    'PathIntegralMode': 'quantum_limit',
                    'PathIntegralRewardMode': 'policy',
                    'PathIntegralLambda': 0.05,
                    'PathIntegralSamples': 100
                },
                'expected_success': True
            },
            {
                'name': 'quantum_hybrid',
                'config': {
                    'PathIntegralMode': 'quantum_limit',
                    'PathIntegralRewardMode': 'hybrid',
                    'PathIntegralLambda': 0.1,
                    'PathIntegralSamples': 75
                },
                'expected_success': True
            },
            {
                'name': 'high_exploration',
                'config': {
                    'PathIntegralMode': 'quantum_limit',
                    'PathIntegralRewardMode': 'cp_score',
                    'PathIntegralLambda': 0.01,
                    'PathIntegralSamples': 200
                },
                'expected_success': True
            },
            {
                'name': 'invalid_lambda',
                'config': {
                    'PathIntegralMode': 'competitive',
                    'PathIntegralLambda': -1.0,  # Geçersiz değer
                    'PathIntegralSamples': 50
                },
                'expected_success': False  # Hata bekleniyor
            }
        ]
        
        # Test pozisyonları
        self.test_positions = [
            {
                'name': 'startpos',
                'fen': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
                'moves': []
            },
            {
                'name': 'sicilian',
                'fen': 'rnbqkb1r/pp1ppppp/5n2/2p5/2P5/8/PP1PPPPP/RNBQKBNR w KQkq - 0 3',
                'moves': ['e2e4', 'c7c5', 'g1f3']
            },
            {
                'name': 'tactical',
                'fen': 'r2qkb1r/ppp2ppp/2n1bn2/3pp3/3PP3/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 0 6',
                'moves': ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1c4', 'g8f6', 'b1c3', 'f8e7']
            }
        ]
    
    def check_dependencies(self) -> bool:
        """Gerekli bağımlılıkları kontrol et"""
        print("Bağımlılıklar kontrol ediliyor...")
        
        required_tools = ['meson', 'ninja', 'gcc', 'g++']
        missing_tools = []
        
        for tool in required_tools:
            if not shutil.which(tool):
                missing_tools.append(tool)
        
        if missing_tools:
            print(f"HATA: Eksik araçlar: {', '.join(missing_tools)}")
            print("Kurulum önerileri:")
            print("  Ubuntu/Debian: sudo apt install meson ninja-build build-essential")
            print("  CentOS/RHEL: sudo yum install meson ninja-build gcc gcc-c++")
            return False
        
        # CUDA kontrolü (opsiyonel)
        if shutil.which('nvcc'):
            print("✓ CUDA desteği mevcut")
        else:
            print("⚠ CUDA bulunamadı, CPU-only build yapılacak")
        
        print("✓ Tüm gerekli bağımlılıklar mevcut")
        return True
    
    def clean_build(self) -> bool:
        """Build dizinini temizle"""
        print("Build dizini temizleniyor...")
        
        if self.build_dir.exists():
            try:
                shutil.rmtree(self.build_dir)
                print(f"✓ {self.build_dir} temizlendi")
            except Exception as e:
                print(f"HATA: Build dizini temizlenemedi: {e}")
                return False
        
        return True
    
    def configure_build(self, enable_cuda: bool = True, debug: bool = False) -> bool:
        """Build'i konfigüre et"""
        print("Build konfigürasyonu...")
        
        # Meson setup komutu
        cmd = [
            'meson', 'setup', str(self.build_dir),
            '-Dpath_integral=true',  # Path Integral'ı etkinleştir
            '-Dbuildtype=release' if not debug else '-Dbuildtype=debug'
        ]
        
        # CUDA desteği
        if enable_cuda and shutil.which('nvcc'):
            cmd.append('-Dcudnn=true')
            print("CUDA desteği etkinleştirildi")
        
        # Konfigürasyonu çalıştır
        try:
            result = subprocess.run(cmd, cwd=self.source_dir, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"HATA: Meson setup başarısız")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return False
            
            print("✓ Build konfigürasyonu tamamlandı")
            return True
            
        except Exception as e:
            print(f"HATA: Build konfigürasyonu sırasında hata: {e}")
            return False
    
    def compile_project(self) -> bool:
        """Projeyi derle"""
        print("Proje derleniyor...")
        
        cmd = ['meson', 'compile', '-C', str(self.build_dir)]
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, cwd=self.source_dir, capture_output=True, text=True)
            compile_time = time.time() - start_time
            
            if result.returncode != 0:
                print(f"HATA: Derleme başarısız")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return False
            
            print(f"✓ Derleme tamamlandı ({compile_time:.1f}s)")
            
            # LC0 executable'ının varlığını kontrol et
            lc0_path = self.build_dir / 'lc0'
            if not lc0_path.exists():
                print(f"HATA: LC0 executable bulunamadı: {lc0_path}")
                return False
            
            print(f"✓ LC0 executable oluşturuldu: {lc0_path}")
            return True
            
        except Exception as e:
            print(f"HATA: Derleme sırasında hata: {e}")
            return False
    
    def test_basic_functionality(self) -> bool:
        """Temel fonksiyonalite testleri"""
        print("Temel fonksiyonalite testleri...")
        
        lc0_path = self.build_dir / 'lc0'
        
        # UCI protokol testi
        print("  UCI protokol testi...")
        try:
            cmd = [str(lc0_path)]
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE, text=True)
            
            stdout, stderr = process.communicate(input="uci\nquit\n", timeout=10)
            
            if "uciok" not in stdout:
                print(f"HATA: UCI protokol yanıtı alınamadı")
                print(f"STDOUT: {stdout}")
                print(f"STDERR: {stderr}")
                return False
            
            # Path Integral seçeneklerini kontrol et
            pi_options = ['PathIntegralMode', 'PathIntegralLambda', 'PathIntegralSamples', 'PathIntegralRewardMode']
            missing_options = []
            
            for option in pi_options:
                if option not in stdout:
                    missing_options.append(option)
            
            if missing_options:
                print(f"UYARI: Eksik Path Integral seçenekleri: {missing_options}")
            else:
                print("✓ Tüm Path Integral seçenekleri mevcut")
            
            print("✓ UCI protokol testi başarılı")
            return True
            
        except subprocess.TimeoutExpired:
            print("HATA: UCI protokol testi zaman aşımı")
            process.kill()
            return False
        except Exception as e:
            print(f"HATA: UCI protokol testi sırasında hata: {e}")
            return False
    
    def test_path_integral_configs(self) -> Dict[str, Any]:
        """Path Integral konfigürasyonlarını test et"""
        print("Path Integral konfigürasyon testleri...")
        
        lc0_path = self.build_dir / 'lc0'
        results = {}
        
        for test_config in self.test_configs:
            config_name = test_config['name']
            config = test_config['config']
            expected_success = test_config['expected_success']
            
            print(f"  Test: {config_name}")
            
            try:
                # UCI komutlarını hazırla
                uci_commands = ["uci"]
                
                # Konfigürasyon komutları
                for key, value in config.items():
                    uci_commands.append(f"setoption name {key} value {value}")
                
                # engine.py ile tutarlı LC0 ayarları ekle
                uci_commands.extend([
                    "setoption name PolicyTemperature value 0.7",
                    "setoption name CPuct value 1.0"
                    # Diğer seçenekler mevcut değil
                ])
                
                # Test pozisyonu ve analiz
                uci_commands.extend([
                    "position startpos",
                    "go movetime 3000",  # 3 saniye analiz
                    "quit"
                ])
                
                # Komutu çalıştır
                cmd = [str(lc0_path)]
                process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, text=True)
                
                input_text = "\n".join(uci_commands) + "\n"
                stdout, stderr = process.communicate(input=input_text, timeout=15)
                
                # Sonucu analiz et
                success = process.returncode == 0 and "bestmove" in stdout
                
                if success == expected_success:
                    status = "✓ BAŞARILI"
                else:
                    status = "✗ BAŞARISIZ"
                
                results[config_name] = {
                    'success': success,
                    'expected_success': expected_success,
                    'test_passed': success == expected_success,
                    'stdout': stdout,
                    'stderr': stderr,
                    'config': config
                }
                
                print(f"    {status} (beklenen: {expected_success}, sonuç: {success})")
                
                # Hata durumunda detay göster
                if not results[config_name]['test_passed']:
                    print(f"    STDERR: {stderr[:200]}...")
                
            except subprocess.TimeoutExpired:
                print(f"    ✗ ZAMAN AŞIMI")
                process.kill()
                results[config_name] = {
                    'success': False,
                    'expected_success': expected_success,
                    'test_passed': False,
                    'error': 'timeout',
                    'config': config
                }
            except Exception as e:
                print(f"    ✗ HATA: {e}")
                results[config_name] = {
                    'success': False,
                    'expected_success': expected_success,
                    'test_passed': False,
                    'error': str(e),
                    'config': config
                }
        
        return results
    
    def test_performance_benchmarks(self) -> Dict[str, Any]:
        """Performans benchmark testleri"""
        print("Performans benchmark testleri...")
        
        lc0_path = self.build_dir / 'lc0'
        results = {}
        
        # Baseline test (Path Integral kapalı)
        print("  Baseline test (Path Integral kapalı)...")
        baseline_result = self._run_performance_test(lc0_path, {}, "baseline")
        results['baseline'] = baseline_result
        
        # Path Integral testleri
        pi_configs = [
            {
                'name': 'competitive_fast',
                'config': {
                    'PathIntegralMode': 'competitive',
                    'PathIntegralLambda': 0.1,
                    'PathIntegralSamples': 25
                }
            },
            {
                'name': 'competitive_balanced',
                'config': {
                    'PathIntegralMode': 'competitive',
                    'PathIntegralLambda': 0.1,
                    'PathIntegralSamples': 50
                }
            },
            {
                'name': 'quantum_limit',
                'config': {
                    'PathIntegralMode': 'quantum_limit',
                    'PathIntegralRewardMode': 'hybrid',
                    'PathIntegralLambda': 0.1,
                    'PathIntegralSamples': 50
                }
            }
        ]
        
        for pi_config in pi_configs:
            config_name = pi_config['name']
            config = pi_config['config']
            
            print(f"  Test: {config_name}")
            result = self._run_performance_test(lc0_path, config, config_name)
            results[config_name] = result
            
            # Baseline ile karşılaştır
            if baseline_result['success'] and result['success']:
                if baseline_result['time'] > 0:
                    overhead = ((result['time'] - baseline_result['time']) / baseline_result['time']) * 100
                    print(f"    Ek yük: %{overhead:.1f}")
                
                if baseline_result['nps'] > 0:
                    nps_ratio = result['nps'] / baseline_result['nps']
                    print(f"    NPS oranı: {nps_ratio:.2f}")
        
        return results
    
    def _run_performance_test(self, lc0_path: Path, config: Dict, test_name: str) -> Dict[str, Any]:
        """Tek performans testi çalıştır"""
        try:
            # UCI komutları
            uci_commands = ["uci"]
            
            # Konfigürasyon
            for key, value in config.items():
                uci_commands.append(f"setoption name {key} value {value}")
            
            # engine.py ile tutarlı LC0 ayarları
            uci_commands.extend([
                "setoption name PolicyTemperature value 0.7",
                "setoption name CPuct value 1.0"
                # Diğer seçenekler mevcut değil
            ])
            
            # Test
            uci_commands.extend([
                "position startpos moves e2e4 e7e5",
                "go movetime 5000",  # 5 saniye
                "quit"
            ])
            
            # Çalıştır
            start_time = time.time()
            cmd = [str(lc0_path)]
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True)
            
            input_text = "\n".join(uci_commands) + "\n"
            stdout, stderr = process.communicate(input=input_text, timeout=20)
            end_time = time.time()
            
            # Sonuçları parse et
            success = process.returncode == 0 and "bestmove" in stdout
            total_time = end_time - start_time
            
            nodes = 0
            nps = 0
            
            if success:
                # Son info satırından nodes ve nps al
                info_lines = [line for line in stdout.split('\n') if line.startswith('info') and 'depth' in line]
                if info_lines:
                    last_info = info_lines[-1]
                    
                    # Nodes
                    if 'nodes' in last_info:
                        try:
                            nodes_part = last_info.split('nodes')[1].split()[0]
                            nodes = int(nodes_part)
                        except:
                            pass
                    
                    # NPS
                    if 'nps' in last_info:
                        try:
                            nps_part = last_info.split('nps')[1].split()[0]
                            nps = int(nps_part)
                        except:
                            pass
            
            result = {
                'success': success,
                'time': total_time,
                'nodes': nodes,
                'nps': nps,
                'config': config,
                'stdout_length': len(stdout),
                'stderr_length': len(stderr)
            }
            
            if success:
                print(f"    ✓ {total_time:.2f}s, {nodes:,} nodes, {nps:,} nps")
            else:
                print(f"    ✗ Başarısız")
            
            return result
            
        except Exception as e:
            print(f"    ✗ Hata: {e}")
            return {
                'success': False,
                'error': str(e),
                'config': config
            }
    
    def run_comparison_with_engine_py(self) -> Dict[str, Any]:
        """engine.py ile karşılaştırma testi"""
        print("engine.py ile karşılaştırma testi...")
        
        # engine.py'nin varlığını kontrol et
        engine_py_path = self.source_dir / 'engine.py'
        if not engine_py_path.exists():
            print("⚠ engine.py bulunamadı, karşılaştırma atlanıyor")
            return {'skipped': True, 'reason': 'engine.py not found'}
        
        # Karşılaştırma scriptini çalıştır
        comparison_script = self.source_dir / 'compare_path_integral_systems.py'
        if not comparison_script.exists():
            print("⚠ Karşılaştırma scripti bulunamadı")
            return {'skipped': True, 'reason': 'comparison script not found'}
        
        try:
            lc0_path = self.build_dir / 'lc0'
            cmd = [
                'python3', str(comparison_script),
                '--lc0-path', str(lc0_path),
                '--engine-py', str(engine_py_path),
                '--no-plots'  # Grafik oluşturmayı atla
            ]
            
            print("  Karşılaştırma scripti çalıştırılıyor...")
            result = subprocess.run(cmd, cwd=self.source_dir, capture_output=True, text=True, timeout=300)
            
            success = result.returncode == 0
            
            if success:
                print("✓ Karşılaştırma tamamlandı")
            else:
                print(f"✗ Karşılaştırma başarısız: {result.stderr}")
            
            return {
                'success': success,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
            
        except subprocess.TimeoutExpired:
            print("✗ Karşılaştırma zaman aşımı")
            return {'success': False, 'error': 'timeout'}
        except Exception as e:
            print(f"✗ Karşılaştırma hatası: {e}")
            return {'success': False, 'error': str(e)}
    
    def generate_deployment_report(self) -> str:
        """Deployment raporu oluştur"""
        report_filename = 'path_integral_deployment_report.md'
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("# LC0 Path Integral Deployment Raporu\n\n")
            f.write(f"**Rapor Tarihi**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Build bilgileri
            f.write("## Build Bilgileri\n\n")
            f.write(f"- **Kaynak Dizin**: {self.source_dir}\n")
            f.write(f"- **Build Dizin**: {self.build_dir}\n")
            f.write(f"- **LC0 Executable**: {self.build_dir / 'lc0'}\n\n")
            
            # Test sonuçları
            if hasattr(self, 'config_test_results'):
                f.write("## Konfigürasyon Test Sonuçları\n\n")
                
                passed_tests = sum(1 for r in self.config_test_results.values() if r.get('test_passed', False))
                total_tests = len(self.config_test_results)
                
                f.write(f"**Başarı Oranı**: {passed_tests}/{total_tests} ({100*passed_tests/total_tests:.1f}%)\n\n")
                
                f.write("| Test | Beklenen | Sonuç | Durum |\n")
                f.write("|------|----------|-------|-------|\n")
                
                for test_name, result in self.config_test_results.items():
                    expected = "✓" if result.get('expected_success', False) else "✗"
                    actual = "✓" if result.get('success', False) else "✗"
                    status = "BAŞARILI" if result.get('test_passed', False) else "BAŞARISIZ"
                    f.write(f"| {test_name} | {expected} | {actual} | {status} |\n")
                
                f.write("\n")
            
            # Performans sonuçları
            if hasattr(self, 'performance_results'):
                f.write("## Performans Test Sonuçları\n\n")
                
                f.write("| Test | Süre (s) | Nodes | NPS | Durum |\n")
                f.write("|------|----------|-------|-----|-------|\n")
                
                for test_name, result in self.performance_results.items():
                    if result.get('success', False):
                        time_val = result.get('time', 0)
                        nodes = result.get('nodes', 0)
                        nps = result.get('nps', 0)
                        status = "✓"
                    else:
                        time_val = nodes = nps = 0
                        status = "✗"
                    
                    f.write(f"| {test_name} | {time_val:.2f} | {nodes:,} | {nps:,} | {status} |\n")
                
                f.write("\n")
            
            # Öneriler
            f.write("## Deployment Önerileri\n\n")
            
            if hasattr(self, 'config_test_results'):
                failed_tests = [name for name, result in self.config_test_results.items() 
                              if not result.get('test_passed', False)]
                
                if failed_tests:
                    f.write("### Başarısız Testler\n\n")
                    for test_name in failed_tests:
                        f.write(f"- **{test_name}**: Konfigürasyonu gözden geçirin\n")
                    f.write("\n")
            
            f.write("### Genel Öneriler\n\n")
            f.write("1. **GPU Desteği**: CUDA backend'i etkinleştirin\n")
            f.write("2. **Bellek**: Yeterli sistem belleği sağlayın (8GB+)\n")
            f.write("3. **Thread Sayısı**: CPU çekirdek sayısına göre ayarlayın\n")
            f.write("4. **Monitoring**: Performans metrikleri izleyin\n\n")
            
            # Sonuç
            f.write("## Sonuç\n\n")
            
            overall_success = True
            if hasattr(self, 'config_test_results'):
                config_success = all(r.get('test_passed', False) for r in self.config_test_results.values())
                overall_success = overall_success and config_success
            
            if hasattr(self, 'performance_results'):
                perf_success = all(r.get('success', False) for r in self.performance_results.values())
                overall_success = overall_success and perf_success
            
            if overall_success:
                f.write("✅ **LC0 Path Integral başarıyla deploy edildi ve tüm testler geçti.**\n\n")
                f.write("Sistem production kullanıma hazırdır.\n")
            else:
                f.write("⚠️ **Bazı testler başarısız oldu.**\n\n")
                f.write("Sorunları giderdikten sonra testleri tekrar çalıştırın.\n")
        
        print(f"Deployment raporu oluşturuldu: {report_filename}")
        return report_filename
    
    def run_full_deployment(self, enable_cuda: bool = True, debug: bool = False, 
                          run_comparison: bool = False) -> bool:
        """Tam deployment sürecini çalıştır"""
        print("=" * 60)
        print("LC0 PATH INTEGRAL DEPLOYMENT BAŞLIYOR")
        print("=" * 60)
        
        try:
            # 1. Bağımlılık kontrolü
            if not self.check_dependencies():
                return False
            
            # 2. Build temizleme
            if not self.clean_build():
                return False
            
            # 3. Build konfigürasyonu
            if not self.configure_build(enable_cuda, debug):
                return False
            
            # 4. Derleme
            if not self.compile_project():
                return False
            
            # 5. Temel fonksiyonalite testleri
            if not self.test_basic_functionality():
                return False
            
            # 6. Konfigürasyon testleri
            print("\n" + "-" * 40)
            self.config_test_results = self.test_path_integral_configs()
            
            # 7. Performans testleri
            print("\n" + "-" * 40)
            self.performance_results = self.test_performance_benchmarks()
            
            # 8. engine.py karşılaştırması (opsiyonel)
            if run_comparison:
                print("\n" + "-" * 40)
                self.comparison_results = self.run_comparison_with_engine_py()
            
            # 9. Rapor oluştur
            print("\n" + "-" * 40)
            report_file = self.generate_deployment_report()
            
            print("\n" + "=" * 60)
            print("DEPLOYMENT TAMAMLANDI!")
            print("=" * 60)
            
            # Özet
            config_passed = sum(1 for r in self.config_test_results.values() if r.get('test_passed', False))
            config_total = len(self.config_test_results)
            
            perf_passed = sum(1 for r in self.performance_results.values() if r.get('success', False))
            perf_total = len(self.performance_results)
            
            print(f"Konfigürasyon Testleri: {config_passed}/{config_total}")
            print(f"Performans Testleri: {perf_passed}/{perf_total}")
            print(f"Rapor: {report_file}")
            
            if run_comparison and hasattr(self, 'comparison_results'):
                comp_success = self.comparison_results.get('success', False)
                print(f"engine.py Karşılaştırması: {'✓' if comp_success else '✗'}")
            
            return True
            
        except KeyboardInterrupt:
            print("\nKullanıcı tarafından iptal edildi.")
            return False
        except Exception as e:
            print(f"Beklenmeyen hata: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    parser = argparse.ArgumentParser(description='LC0 Path Integral deployment ve test')
    parser.add_argument('--build', action='store_true', help='Build işlemini çalıştır')
    parser.add_argument('--test', action='store_true', help='Testleri çalıştır')
    parser.add_argument('--compare', action='store_true', help='engine.py ile karşılaştır')
    parser.add_argument('--clean', action='store_true', help='Build dizinini temizle')
    parser.add_argument('--debug', action='store_true', help='Debug build yap')
    parser.add_argument('--no-cuda', action='store_true', help='CUDA desteğini devre dışı bırak')
    parser.add_argument('--source-dir', default='.', help='Kaynak kod dizini')
    parser.add_argument('--build-dir', default='builddir', help='Build dizini')
    
    args = parser.parse_args()
    
    # Varsayılan olarak build ve test yap
    if not any([args.build, args.test, args.compare, args.clean]):
        args.build = True
        args.test = True
    
    # Deployer oluştur
    deployer = PathIntegralDeployer(args.source_dir, args.build_dir)
    
    # Clean işlemi
    if args.clean:
        deployer.clean_build()
        return
    
    # Full deployment
    if args.build and args.test:
        success = deployer.run_full_deployment(
            enable_cuda=not args.no_cuda,
            debug=args.debug,
            run_comparison=args.compare
        )
        sys.exit(0 if success else 1)
    
    # Sadece build
    if args.build:
        if not deployer.check_dependencies():
            sys.exit(1)
        if not deployer.clean_build():
            sys.exit(1)
        if not deployer.configure_build(not args.no_cuda, args.debug):
            sys.exit(1)
        if not deployer.compile_project():
            sys.exit(1)
        print("Build tamamlandı!")
    
    # Sadece test
    if args.test:
        lc0_path = Path(args.build_dir) / 'lc0'
        if not lc0_path.exists():
            print(f"HATA: LC0 executable bulunamadı: {lc0_path}")
            print("Önce --build ile derleme yapın.")
            sys.exit(1)
        
        if not deployer.test_basic_functionality():
            sys.exit(1)
        
        config_results = deployer.test_path_integral_configs()
        perf_results = deployer.test_performance_benchmarks()
        
        print("Testler tamamlandı!")
    
    # Sadece karşılaştırma
    if args.compare and not args.test:
        comparison_results = deployer.run_comparison_with_engine_py()
        if not comparison_results.get('success', False):
            sys.exit(1)
        print("Karşılaştırma tamamlandı!")

if __name__ == '__main__':
    main()