#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ТЕСТ 2: ВАЛИДАЦИЯ МАСШТАБИРОВАНИЯ
===================================

Цель: Проверить как размеры системы влияют на проводимость.

Подтест A: Варьирование отношения L_tube/L_field
Подтест B: Варьирование абсолютных размеров при фиксированном отношении

Все симуляции при φ > φ_c чтобы избежать near-percolation эффектов.
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime

try:
    from ФИНАЛ import EnhancedNanotubeSimulator
    print("✅ Симулятор загружен")
except ImportError:
    print("❌ Не найден ФИНАЛ.py!")
    sys.exit(1)

print("="*80)
print("🔬 ТЕСТ 2: ВАЛИДАЦИЯ МАСШТАБИРОВАНИЯ")
print("="*80)

results_A = []
results_B = []
base_seed = 100

# ==========================================
# ПОДТЕСТ A: ВАРЬИРОВАНИЕ ОТНОШЕНИЯ L_tube/L_field
# ==========================================

print("\n" + "="*80)
print("📐 ПОДТЕСТ A: ВАРЬИРОВАНИЕ ОТНОШЕНИЯ L_tube/L_field")
print("="*80)
print("\nСтратегия:")
print("  • Фиксируем: L_field = 400 нм")
print("  • Варьируем: L_tube от 150 до 500 нм (отношение 0.375 → 1.25)")
print("  • Подбираем φ > φ_c для каждого L_tube (чтобы была перколяция)")
print("  • 5 повторов на каждую точку для статистики")
print("  • Ожидаемый результат: σ растёт с увеличением L_tube/L_field")
print("\n" + "="*80)

# Параметры для подтеста A
field_size_A = 400.0  # нм (фиксировано)

# Конфигурации: (L_tube, num_tubes)
# num_tubes подобраны чтобы φ была выше порога (~0.5-1%)
configs_A = [
    # L_tube, num_tubes, описание
    (150, 1800, "L/L_field=0.375, много коротких"),
    (200, 1200, "L/L_field=0.50"),
    (250, 1000, "L/L_field=0.625, baseline"),
    (300, 900,  "L/L_field=0.75"),
    (400, 700,  "L/L_field=1.0"),
    (500, 600,  "L/L_field=1.25, мало длинных"),
]

n_repeats_A = 5

print(f"\n🔄 Запуск {len(configs_A)} конфигураций × {n_repeats_A} повторов = {len(configs_A) * n_repeats_A} симуляций")
print(f"⏱️  Ожидаемое время: ~{len(configs_A) * n_repeats_A * 1.2 / 60:.1f} мин\n")

run_id = 1
for config_id, (L_tube, num_tubes, desc) in enumerate(configs_A, 1):
    ratio = L_tube / field_size_A
    
    print(f"\n{'─'*80}")
    print(f"⚙️  Конфигурация {config_id}/{len(configs_A)}: {desc}")
    print(f"   L_tube = {L_tube} нм, L_field = {field_size_A} нм, ratio = {ratio:.3f}")
    print(f"   Количество трубок: {num_tubes}")
    print(f"{'─'*80}")
    
    for rep in range(1, n_repeats_A + 1):
        print(f"\n   [{run_id}] Повтор {rep}/{n_repeats_A}...", end=" ")
        
        start_time = time.time()
        
        sim = EnhancedNanotubeSimulator()
        sim.num_tubes = num_tubes
        sim.tube_length = L_tube
        sim.field_size = field_size_A
        sim.voxel_size = 2.0
        sim.aligned_jitter_deg = 4.0
        sim.enable_particles = False  # БЕЗ аэрогеля для чистоты
        sim.enable_conductivity = True
        sim.show_percolation_analysis = False
        sim.random_seed = base_seed + run_id
        sim.set_random_seed()
        
        sim.generate_all_objects()
        sim.calculate_conductivity()
        
        elapsed = time.time() - start_time
        res = sim.conductivity_results
        
        result = {
            'test': 'A',
            'run_id': run_id,
            'config_id': config_id,
            'repeat': rep,
            'L_tube_nm': L_tube,
            'L_field_nm': field_size_A,
            'ratio': ratio,
            'num_tubes': num_tubes,
            'percolates': res.get('percolates', False),
            'phi_percent': res.get('volume_fraction', 0) * 100,
            'sigma_Sm': res.get('sigma_effective', 0),
            'method': res.get('method', 'N/A'),
            'time_sec': round(elapsed, 1),
        }
        
        results_A.append(result)
        
        perc = "✓" if result['percolates'] else "✗"
        print(f"{perc} φ={result['phi_percent']:.2f}% σ={result['sigma_Sm']:.1e} ({elapsed:.1f}s)")
        
        run_id += 1

# Анализ подтеста A
print("\n" + "="*80)
print("📊 АНАЛИЗ ПОДТЕСТА A")
print("="*80)

df_A = pd.DataFrame(results_A)
summary_A = df_A.groupby('ratio').agg({
    'phi_percent': ['mean', 'std'],
    'sigma_Sm': ['mean', 'std'],
    'percolates': 'sum'
}).round(4)

print("\nСводка по конфигурациям:")
print(summary_A)

print(f"\nПерколяция: {df_A['percolates'].sum()}/{len(df_A)} симуляций")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename_A = f"test_2A_ratio_validation_{timestamp}.csv"
df_A.to_csv(filename_A, index=False, encoding='utf-8-sig')
print(f"\n✅ Результаты подтеста A: {filename_A}")

# ==========================================
# ПОДТЕСТ B: ФИКСИРОВАННОЕ ОТНОШЕНИЕ, РАЗНЫЕ РАЗМЕРЫ
# ==========================================

print("\n\n" + "="*80)
print("📏 ПОДТЕСТ B: ФИКСИРОВАННОЕ ОТНОШЕНИЕ, РАЗНЫЕ АБСОЛЮТНЫЕ РАЗМЕРЫ")
print("="*80)
print("\nСтратегия:")
print("  • Фиксируем: L_tube/L_field = 0.625 (как baseline)")
print("  • Варьируем: (L_tube, L_field) = (125, 200), (250, 400), (500, 800)")
print("  • Подбираем φ ≈ const для всех размеров (через num_tubes)")
print("  • 5 повторов на размер")
print("  • Ожидаемый результат: σ примерно одинаковая (масштабная инвариантность)")
print("\n" + "="*80)

# Параметры для подтеста B
target_ratio = 0.625  # L_tube/L_field
target_phi = 0.007    # ~0.7% (выше порога)

# Конфигурации: (L_tube, L_field, num_tubes)
# num_tubes рассчитаны для φ ≈ 0.7%
configs_B = [
    # (L_tube, L_field, num_tubes, описание)
    (125, 200, 500,  "Малый масштаб"),
    (250, 400, 1000, "Средний масштаб (baseline)"),
    (500, 800, 2000, "Большой масштаб"),
]

n_repeats_B = 5

print(f"\n🔄 Запуск {len(configs_B)} конфигураций × {n_repeats_B} повторов = {len(configs_B) * n_repeats_B} симуляций")
print(f"⏱️  Ожидаемое время: ~{len(configs_B) * n_repeats_B * 1.5 / 60:.1f} мин\n")

run_id = 1
for config_id, (L_tube, L_field, num_tubes, desc) in enumerate(configs_B, 1):
    actual_ratio = L_tube / L_field
    
    print(f"\n{'─'*80}")
    print(f"⚙️  Конфигурация {config_id}/{len(configs_B)}: {desc}")
    print(f"   L_tube = {L_tube} нм, L_field = {L_field} нм, ratio = {actual_ratio:.3f}")
    print(f"   Количество трубок: {num_tubes}")
    print(f"{'─'*80}")
    
    for rep in range(1, n_repeats_B + 1):
        print(f"\n   [{run_id}] Повтор {rep}/{n_repeats_B}...", end=" ")
        
        start_time = time.time()
        
        sim = EnhancedNanotubeSimulator()
        sim.num_tubes = num_tubes
        sim.tube_length = L_tube
        sim.field_size = L_field
        sim.voxel_size = 2.0
        sim.aligned_jitter_deg = 4.0
        sim.enable_particles = False
        sim.enable_conductivity = True
        sim.show_percolation_analysis = False
        sim.random_seed = base_seed + 1000 + run_id  # Другой диапазон seed
        sim.set_random_seed()
        
        sim.generate_all_objects()
        sim.calculate_conductivity()
        
        elapsed = time.time() - start_time
        res = sim.conductivity_results
        
        result = {
            'test': 'B',
            'run_id': run_id,
            'config_id': config_id,
            'repeat': rep,
            'L_tube_nm': L_tube,
            'L_field_nm': L_field,
            'ratio': actual_ratio,
            'num_tubes': num_tubes,
            'percolates': res.get('percolates', False),
            'phi_percent': res.get('volume_fraction', 0) * 100,
            'sigma_Sm': res.get('sigma_effective', 0),
            'method': res.get('method', 'N/A'),
            'time_sec': round(elapsed, 1),
        }
        
        results_B.append(result)
        
        perc = "✓" if result['percolates'] else "✗"
        print(f"{perc} φ={result['phi_percent']:.2f}% σ={result['sigma_Sm']:.1e} ({elapsed:.1f}s)")
        
        run_id += 1

# Анализ подтеста B
print("\n" + "="*80)
print("📊 АНАЛИЗ ПОДТЕСТА B")
print("="*80)

df_B = pd.DataFrame(results_B)
summary_B = df_B.groupby('L_field_nm').agg({
    'phi_percent': ['mean', 'std'],
    'sigma_Sm': ['mean', 'std'],
    'percolates': 'sum'
}).round(4)

print("\nСводка по размерам (при фиксированном отношении 0.625):")
print(summary_B)

# Проверка масштабной инвариантности
sigma_means = df_B.groupby('L_field_nm')['sigma_Sm'].mean()
sigma_cv = (sigma_means.std() / sigma_means.mean()) * 100

print(f"\n🎯 Проверка масштабной инвариантности:")
print(f"   CV(σ) = {sigma_cv:.1f}%")
if sigma_cv < 30:
    print("   ✅ ХОРОШО: σ примерно одинаковая при разных размерах")
elif sigma_cv < 60:
    print("   ⚠️  СРЕДНЕ: Есть зависимость от размера (~{sigma_cv:.0f}%)")
else:
    print("   ❌ ПЛОХО: Сильная зависимость от размера")

print(f"\nПерколяция: {df_B['percolates'].sum()}/{len(df_B)} симуляций")

filename_B = f"test_2B_size_validation_{timestamp}.csv"
df_B.to_csv(filename_B, index=False, encoding='utf-8-sig')
print(f"\n✅ Результаты подтеста B: {filename_B}")

# ==========================================
# ОБЩАЯ СТАТИСТИКА
# ==========================================

print("\n" + "="*80)
print("✅ ОБА ПОДТЕСТА ЗАВЕРШЕНЫ")
print("="*80)

total_time_A = df_A['time_sec'].sum()
total_time_B = df_B['time_sec'].sum()

print(f"\n⏱️  ВРЕМЯ:")
print(f"   Подтест A: {total_time_A/60:.1f} мин ({len(df_A)} симуляций)")
print(f"   Подтест B: {total_time_B/60:.1f} мин ({len(df_B)} симуляций)")
print(f"   ИТОГО: {(total_time_A + total_time_B)/60:.1f} мин")

print(f"\n📁 ФАЙЛЫ:")
print(f"   • {filename_A}")
print(f"   • {filename_B}")

print("\n💡 ЧТО СМОТРЕТЬ:")
print("\n   ПОДТЕСТ A (варьирование отношения):")
print("      → График σ(L_tube/L_field) должен показать рост σ с увеличением отношения")
print("      → Это основная валидация: нужно найти оптимальное отношение")
print("\n   ПОДТЕСТ B (фиксированное отношение):")
print("      → σ должна быть примерно одинаковой для всех размеров")
print("      → CV < 30% = хорошая масштабная инвариантность")
print("      → Если CV > 50% = есть finite-size эффекты")

print("\n" + "="*80)
