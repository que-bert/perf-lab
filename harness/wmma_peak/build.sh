#!/usr/bin/env bash
# Build wmma_peak next to its shaders (shaders are compiled with glslc at run time).
set -euo pipefail
cd "$(dirname "$0")"
g++ -O2 -std=c++17 -o wmma_peak wmma_peak.cpp -lvulkan
