#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ТЕСТ 1: ПРОВОДИМОСТЬ МАТРИЦЫ (БЕЗ ПЕРКОЛЯЦИИ УНТ)
===================================================

Цель: Проверить что программа правильно считает проводимость через матрицу
      когда нет перколяции по УНТ.

Сценарий: 10 симуляций от очень малого числа УНТ до порога перколяции
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime

# Импортируем исправленную версию
try:
    # Замени путь на свой!
    from ФИНАЛ import EnhancedNanotubeSimulator
    print("✅ Загружена версия с проводимостью матрицы")
except ImportError:
    print("❌ Не найден ФИНАЛ.py! Положи файл в ту же папку.")
    sys.exit(1)

print("="*70)
print("🧪 ТЕСТ: ПРОВОДИМОСТЬ МАТРИЦЫ БЕЗ ПЕРКОЛЯЦИИ УНТ")
print("="*70)
print("\nСценарий:")
print("  • 10 концентраций УНТ: от очень малой до порога перколяции")
print("  • Все симуляции С аэрогелем (enable_particles=True)")
print("  • Ожидается: σ ≈ 1e-6 См/м для систем без перколяции")
print("  • Ожидается: σ >> 1e-6 См/м когда появляется перколяция")
print("\n" + "="*70)

# Параметры
results = []
base_seed = 56587

# Концентрации: от 100 до 1000 трубок (порог где-то ~600-800)
num_tubes_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

print(f"\n🔄 Запуск {len(num_tubes_list)} симуляций...")
print("="*70)

for i, num_tubes in enumerate(num_tubes_list, 1):
    print(f"\n[{i}/{len(num_tubes_list)}] Трубок: {num_tubes}")
    print("-"*50)
    
    start_time = time.time()
    
    sim = EnhancedNanotubeSimulator()
    
    # Параметры
    sim.num_tubes = num_tubes
    sim.tube_length = 250.0  # нм
    sim.outer_radius = 1.0
    sim.inner_radius = 0.66
    sim.field_size = 400.0  # нм
    sim.voxel_size = 2.0
    sim.aligned_jitter_deg = 4.0
    sim.orientation_mode = 'aligned'
    
    # ВАЖНО: Включаем аэрогель
    sim.enable_particles = True
    sim.particle_to_total_ratio = 0.05  # 30% аэрогель
    
    sim.enable_conductivity = True
    sim.show_percolation_analysis = False
    sim.random_seed = base_seed + i
    sim.set_random_seed()
    
    # Расчёт
    sim.generate_all_objects()
    sim.calculate_conductivity()
    
    elapsed = time.time() - start_time
    
    # Результаты
    res = sim.conductivity_results
    
    result = {
        'run_id': i,
        'num_tubes': num_tubes,
        'percolates': res.get('percolates', False),
        'phi_percent': res.get('volume_fraction', 0) * 100,
        'sigma_Sm': res.get('sigma_effective', 0),
        'method': res.get('method', 'N/A'),
        'time_sec': round(elapsed, 1),
        'seed': sim.random_seed
    }
    
    results.append(result)
    
    # Вывод
    perc_icon = "✅" if result['percolates'] else "❌"
    print(f"{perc_icon} Перколяция: {result['percolates']}")
    print(f"   φ = {result['phi_percent']:.3f}%")
    print(f"   σ = {result['sigma_Sm']:.2e} См/м")
    print(f"   Метод: {result['method']}")
    print(f"   ⏱️  {elapsed:.1f} сек")

# Сохранение
print("\n" + "="*70)
print("💾 СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
print("="*70)

df = pd.DataFrame(results)

# Определяем порог перколяции
percolating_runs = df[df['percolates'] == True]
if len(percolating_runs) > 0:
    phi_c = percolating_runs['phi_percent'].min()
    n_c = percolating_runs['num_tubes'].min()
else:
    phi_c = None
    n_c = None

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"test_matrix_conductivity_{timestamp}.csv"
df.to_csv(filename, index=False, encoding='utf-8-sig')

print(f"✅ Файл сохранён: {filename}")
print(f"\nСтроки: {len(df)}")

# Анализ
print("\n" + "="*70)
print("📊 АНАЛИЗ РЕЗУЛЬТАТОВ")
print("="*70)

non_perc = df[df['percolates'] == False]
perc = df[df['percolates'] == True]

print(f"\n🔴 БЕЗ ПЕРКОЛЯЦИИ ({len(non_perc)} симуляций):")
if len(non_perc) > 0:
    print(f"   • Количество трубок: {non_perc['num_tubes'].min()}-{non_perc['num_tubes'].max()}")
    print(f"   • Проводимость: σ = {non_perc['sigma_Sm'].mean():.2e} ± {non_perc['sigma_Sm'].std():.2e} См/м")
    print(f"   • Методы: {non_perc['method'].unique()}")
    print(f"\n   ✅ ПРОВЕРКА: σ ≈ 1e-6? {non_perc['sigma_Sm'].mean():.2e} (ожидалось 1.00e-06)")
    
    # Проверка что σ действительно ~1e-6
    if abs(non_perc['sigma_Sm'].mean() - 1e-6) < 1e-7:
        print("   ✅ ТЕСТ ПРОЙДЕН: Проводимость матрицы работает правильно!")
    else:
        print("   ⚠️  ВНИМАНИЕ: Проводимость отличается от ожидаемой!")
else:
    print("   (нет данных)")

print(f"\n🟢 С ПЕРКОЛЯЦИЕЙ ({len(perc)} симуляций):")
if len(perc) > 0:
    print(f"   • Количество трубок: {perc['num_tubes'].min()}-{perc['num_tubes'].max()}")
    print(f"   • Проводимость: σ = {perc['sigma_Sm'].mean():.2e} ± {perc['sigma_Sm'].std():.2e} См/м")
    print(f"   • Методы: {perc['method'].unique()}")
    print(f"\n   ✅ ПРОВЕРКА: σ >> 1e-6? {perc['sigma_Sm'].mean():.2e} (ожидалось ~10-1000 См/м)")
else:
    print("   (нет данных)")

if phi_c is not None:
    print(f"\n🎯 ПОРОГ ПЕРКОЛЯЦИИ:")
    print(f"   • φ_c ≈ {phi_c:.3f}%")
    print(f"   • n_c ≈ {n_c} трубок (при L=250 нм, V=400³ нм³)")

print(f"\n⏱️  Общее время: {df['time_sec'].sum()/60:.1f} мин")

print("\n" + "="*70)
print("✅ ТЕСТ ЗАВЕРШЁН")
print("="*70)
print(f"\n📁 Результаты: {filename}")
print("\n💡 Что смотреть:")
print("   1. Для систем БЕЗ перколяции: σ должна быть ≈ 1e-6 См/м")
print("   2. Для систем С перколяцией: σ должна быть >> 1e-6 См/м")
print("   3. S-образная кривая σ(φ) с плавным переходом")
