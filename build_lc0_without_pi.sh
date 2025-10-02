#!/bin/bash
# LC0'ı Path Integral olmadan derle

echo "LC0 Path Integral olmadan derleniyor..."

# Eski build'i temizle
rm -rf builddir_normal

# Yeni build dizini oluştur
meson setup builddir_normal -Dpath_integral=false -Dcudnn=true

# Derle
meson compile -C builddir_normal

echo "Derleme tamamlandı: builddir_normal/lc0"

# Test et
echo "Normal LC0 test ediliyor..."
echo -e "uci\nposition startpos\ngo nodes 100\nquit" | ./builddir_normal/lc0