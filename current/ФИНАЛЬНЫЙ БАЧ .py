#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BATCH АНАЛИЗ: Калибровка модели УНТ (с повторами и детальной областью)
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    if not os.path.exists('ФИНАЛ.py'):
        print("❌ Файл ФИНАЛ.py не найден!")
        sys.exit(1)
    
    from ФИНАЛ import EnhancedNanotubeSimulator, PhysicsConfig
    print("✅ Симулятор загружен")
except ImportError as e:
    print(f"❌ Ошибка: {e}")
    sys.exit(1)


class SensitivityAnalyzer:
    def __init__(self, output_dir="./results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = []
        self.current_seed = None
        
        # БАЗОВЫЕ ПАРАМЕТРЫ
        self.baseline_params = {
            'num_tubes': 1000,
            'tube_length': 250.0,
            'outer_radius': 1.0,
            'inner_radius': 0.66,
            'field_size': 400.0,
            'voxel_size': 2.0,
            'aligned_jitter_deg': 4.0,
            'orientation_mode': 'aligned',
            'enable_particles': True,
            'particle_to_total_ratio': 0.0,
            'min_gap': 0.34,
            'min_gap_particles': 0.5,
            'enable_conductivity': True,
            'show_percolation_analysis': False,
        }
    
    def run_single_simulation(self, params, label="", run_id=0):
        print(f"\n{'='*60}")
        print(f"🔬 #{run_id}: {label}")
        
        start_time = time.time()
        
        try:
            sim = EnhancedNanotubeSimulator()
            
            for key, value in params.items():
                if hasattr(sim, key):
                    setattr(sim, key, value)
            
            if self.current_seed is not None:
                sim.random_seed = self.current_seed
            else:
                seed = int(time.time() * 1000) % (2**31)
                sim.random_seed = seed
                self.current_seed = seed

            sim.set_random_seed()
            
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                sim.generate_all_objects()
                if sim.enable_conductivity:
                    sim.calculate_conductivity()
            
            elapsed_time = time.time() - start_time
            
            volume_tubes = len(sim.tubes) * np.pi * (sim.outer_radius**2 - sim.inner_radius**2) * sim.tube_length
            volume_field = sim.field_size ** 3
            packing_density_theoretical = volume_tubes / volume_field
            
            result = {
                'run_id': run_id,
                'label': label,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'elapsed_time_sec': round(elapsed_time, 1),
                'random_seed': self.current_seed,
                'num_tubes': params.get('num_tubes'),
                'tube_length_nm': params.get('tube_length'),
                'outer_radius_nm': params.get('outer_radius'),
                'field_size_nm': params.get('field_size'),
                'voxel_size_nm': params.get('voxel_size'),
                'aligned_jitter_deg': params.get('aligned_jitter_deg'),
                'tubes_generated': len(sim.tubes),
                'packing_density_theoretical': round(packing_density_theoretical, 6),
            }
            
            if sim.conductivity_results:
                res = sim.conductivity_results
                result.update({
                    'percolates': res.get('percolates', False),
                    'volume_fraction_real': round(res.get('volume_fraction', 0.0), 6),
                    'sigma_effective_Sm': res.get('sigma_effective', 0.0),
                    'sigma_kirchhoff_Sm': res.get('sigma_kirchhoff', None),
                    'calculation_method': res.get('method', 'N/A'),
                    'tortuosity': round(res.get('tortuosity', 0.0), 2),
                    'contact_factor': round(res.get('contact_resistance_factor', 0.0), 3),
                })
            else:
                result.update({
                    'percolates': False,
                    'volume_fraction_real': 0.0,
                    'sigma_effective_Sm': 0.0,
                    'sigma_kirchhoff_Sm': None,
                    'calculation_method': 'No percolation',
                    'tortuosity': 0.0,
                    'contact_factor': 0.0,
                })
            
            print(f"✅ {elapsed_time:.1f} сек | Seed: {self.current_seed} | "
                  f"Трубки: {result['tubes_generated']} | "
                  f"φ: {result['volume_fraction_real']*100:.3f}% | "
                  f"Перк: {'✓' if result['percolates'] else '✗'} | "
                  f"σ = {result['sigma_effective_Sm']:.2e} См/м")
            
            return result
            
        except Exception as e:
            print(f"❌ ОШИБКА: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def vary_parameter(self, param_name, values, new_seed=True):
        if new_seed:
            self.current_seed = int(time.time() * 1000) % (2**31)
        
        print(f"\n{'#'*70}")
        print(f"📊 {param_name} → {len(values)} точек")
        print(f"   🎲 Seed: {self.current_seed}")
        print(f"{'#'*70}")
        
        series_results = []
        
        for value in values:
            params = self.baseline_params.copy()
            params[param_name] = value
            label = f"{param_name}={value}"
            
            result = self.run_single_simulation(params, label=label, run_id=len(self.results)+1)
            
            if result:
                result['varied_parameter'] = param_name
                result['varied_value'] = value
                series_results.append(result)
                self.results.append(result)
        
        return series_results
    
    def save_results_csv(self, filename=None):
        if not self.results:
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"calibration_{timestamp}.csv"
        
        df = pd.DataFrame(self.results)
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"✅ CSV: {filepath}")
        return filepath
    
    def print_summary(self):
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        
        print(f"\n{'='*70}")
        print("📊 СВОДКА")
        print(f"{'='*70}")
        print(f"Симуляций: {len(self.results)}")
        print(f"Перколяция: {df['percolates'].sum()}/{len(df)}")
        
        if df['percolates'].sum() > 0:
            perc_df = df[df['percolates']]
            print(f"\nПроводимость:")
            print(f"  Среднее: {perc_df['sigma_effective_Sm'].mean():.2e} См/м")
            print(f"  Мин:     {perc_df['sigma_effective_Sm'].min():.2e} См/м")
            print(f"  Макс:    {perc_df['sigma_effective_Sm'].max():.2e} См/м")
        
        print(f"\nВремя: {df['elapsed_time_sec'].sum()/60:.1f} мин")
        print(f"{'='*70}")


# СЦЕНАРИИ
def scenario_1_concentration(analyzer):
    """Концентрация УНТ (стандартная)"""
    print("\n🎯 СЦЕНАРИЙ 1: КОНЦЕНТРАЦИЯ УНТ")
    print("   8 симуляций, ~10-15 мин")
    values = [400, 600, 800, 1000, 1200, 1500, 1800, 2000]
    analyzer.vary_parameter('num_tubes', values)


def scenario_1_detailed(analyzer):
    """Концентрация УНТ (детальная критическая зона)"""
    print("\n🎯 СЦЕНАРИЙ 1-ДЕТАЛЬНЫЙ: КРИТИЧЕСКАЯ ЗОНА")
    print("   14 симуляций, ~15-20 мин")
    
    # Критическая зона (шаг 50)
    critical = list(range(350, 900, 50))  # 350, 400, ..., 850
    # Выше критической (реже)
    above = [1000, 1500, 2000]
    
    values = critical + above
    print(f"   Точки: {values}")
    analyzer.vary_parameter('num_tubes', values)


def scenario_2_geometry(analyzer):
    """Геометрия: длина + джиттер"""
    print("\n📐 СЦЕНАРИЙ 2: ГЕОМЕТРИЯ")
    print("   11 симуляций, ~18-25 мин")
    
    print("\n--- Длина трубки ---")
    values = [150, 200, 250, 300, 400, 500]
    analyzer.vary_parameter('tube_length', values)
    
    print("\n--- Джиттер ориентации ---")
    values = [0, 4, 8, 15, 30]
    analyzer.vary_parameter('aligned_jitter_deg', values)


def scenario_3_convergence(analyzer):
    """Сходимость метода"""
    print("\n🔢 СЦЕНАРИЙ 3: СХОДИМОСТЬ")
    print("   3 симуляции, ~5 мин")
    values = [3, 4, 5]
    analyzer.vary_parameter('voxel_size', values)


def scenario_4_full(analyzer):
    """Всё вместе (старый полный)"""
    print("\n🚀 СЦЕНАРИЙ 4: ПОЛНАЯ КАЛИБРОВКА")
    print("   22 симуляции, ~35-45 мин")
    scenario_1_concentration(analyzer)
    scenario_2_geometry(analyzer)
    scenario_3_convergence(analyzer)


def scenario_5_detailed_repeats(n_repeats=10):
    """10 повторов детального анализа"""
    print("\n🔥 СЦЕНАРИЙ 5: ДЕТАЛЬНЫЙ АНАЛИЗ С ПОВТОРАМИ")
    print(f"   {n_repeats} повторов × (14 концентр. + 11 геометрия) = {n_repeats * 25} симуляций")
    print(f"   ⏱️ Время: ~{n_repeats * 35 / 60:.1f} часа")
    print("\n   ⚠️ ДОЛГО! Рекомендуется запускать на ночь!")
    
    confirm = input("\n   Продолжить? (y/n): ").strip().lower()
    if confirm != 'y':
        print("   Отменено")
        return
    
    total_start = time.time()
    
    for i in range(1, n_repeats + 1):
        print(f"\n{'='*70}")
        print(f"🔁 ПОВТОР {i}/{n_repeats}")
        print(f"{'='*70}")
        
        analyzer = SensitivityAnalyzer()
        
        # Детальная критическая зона концентраций
        scenario_1_detailed(analyzer)
        
        # Геометрия (длина + джиттер)
        scenario_2_geometry(analyzer)
        
        # Сохраняем результаты повтора
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detailed_repeat_{i:02d}_{timestamp}.csv"
        analyzer.save_results_csv(filename)
        
        analyzer.print_summary()
        
        elapsed_total = (time.time() - total_start) / 60
        remaining = (elapsed_total / i) * (n_repeats - i)
        
        print(f"\n   ⏱️ Прошло: {elapsed_total:.1f} мин | Осталось: ~{remaining:.1f} мин")
    
    print(f"\n{'='*70}")
    print(f"✅ ЗАВЕРШЕНО! {n_repeats} повторов за {(time.time() - total_start) / 3600:.2f} часов")
    print(f"{'='*70}")


def main():
    print("="*70)
    print("🔬 КАЛИБРОВКА МОДЕЛИ УНТ")
    print("="*70)
    print("\nСЦЕНАРИИ:")
    print("  1 - Концентрация [8 точек, ~15 мин]")
    print("  2 - Геометрия (length+jitter) [11 точек, ~25 мин]")
    print("  3 - Сходимость (voxel) [3 точки, ~5 мин]")
    print("  4 - Всё вместе (1+2+3) [22 точки, ~45 мин]")
    print("  5 - 🔥 ДЕТАЛЬНЫЙ ПОВТОР [10× (14+11) = 250 сим., ~4-5 часов] ⭐")
    print("  0 - Выход")
    print("="*70)
    
    choice = input("\nВыбери сценарий (1-5): ").strip()
    
    if choice == "5":
        n = input("Количество повторов (по умолчанию 10): ").strip()
        n_repeats = int(n) if n.isdigit() else 10
        scenario_5_detailed_repeats(n_repeats)
        return
    
    analyzer = SensitivityAnalyzer()
    
    if choice == "1":
        scenario_1_concentration(analyzer)
    elif choice == "2":
        scenario_2_geometry(analyzer)
    elif choice == "3":
        scenario_3_convergence(analyzer)
    elif choice == "4":
        scenario_4_full(analyzer)
    elif choice == "0":
        print("Выход.")
        return
    else:
        print(f"❌ Неизвестный: {choice}")
        return
    
    analyzer.print_summary()
    
    print(f"\n{'='*70}")
    print("💾 СОХРАНЕНИЕ")
    print(f"{'='*70}")
    analyzer.save_results_csv()
    
    print("\n✅ ГОТОВО!")


if __name__ == "__main__":
    main()
