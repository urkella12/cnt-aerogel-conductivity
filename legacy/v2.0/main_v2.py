#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
UNIFIED NANOTUBE SIMULATOR - PRODUCTION VERSION
=============================================================================
Комплексный симулятор композитных материалов на основе углеродных нанотрубок
и частиц аэрогеля с расчётом электрической проводимости.

Основные возможности:
- Генерация углеродных нанотрубок с настраиваемой ориентацией
- Генерация частиц аэрогеля с настраиваемыми параметрами
- CPU-ускорение вычислений (Numba JIT)
- Расчёт проводимости методом Кирхгоффа (SciPy)
- Анализ перколяционных кластеров
- Визуализация проводящих путей (2 режима)
=============================================================================
"""

import pyvista as pv
import numpy as np
import time
import sys
from collections import defaultdict
import tkinter as tk
from tkinter import ttk, messagebox

# ==========================================
# ИНИЦИАЛИЗАЦИЯ БИБЛИОТЕК
# ==========================================
try:
    from numba import jit
    NUMBA_AVAILABLE = True
    print("✅ Numba доступна - CPU ускорение (JIT) включено")
except ImportError:
    NUMBA_AVAILABLE = False
    print("⚠️ Numba не установлена, работа без ускорения")
    def jit(nopython=True):
        def decorator(func):
            return func
        return decorator

try:
    from scipy.sparse import lil_matrix, csc_matrix
    from scipy.sparse.linalg import spsolve
    SCIPY_AVAILABLE = True
    print("✅ SciPy доступна - расчёты Кирхгоффа включены")
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ SciPy не установлена, расчёты Кирхгоффа будут ограничены")

# ==========================================
# ОПТИМИЗИРОВАННЫЕ МАТЕМАТИЧЕСКИЕ ФУНКЦИИ (Numba JIT для CPU)
# ==========================================
if NUMBA_AVAILABLE:
    @jit(nopython=True)
    def _point_to_line_distance_numba(point, line_start, line_end):
        """Оптимизированный расчёт расстояния от точки до отрезка (Numba JIT)"""
        lx = line_end[0] - line_start[0]
        ly = line_end[1] - line_start[1]
        lz = line_end[2] - line_start[2]
        px = point[0] - line_start[0]
        py = point[1] - line_start[1]
        pz = point[2] - line_start[2]
        line_len_sq = lx*lx + ly*ly + lz*lz
        
        if line_len_sq < 1e-24:
            return np.sqrt(px*px + py*py + pz*pz)
        
        t = (px*lx + py*ly + pz*lz) / line_len_sq
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        
        nearest_x = line_start[0] + t * lx
        nearest_y = line_start[1] + t * ly
        nearest_z = line_start[2] + t * lz
        
        dx = point[0] - nearest_x
        dy = point[1] - nearest_y
        dz = point[2] - nearest_z
        
        return np.sqrt(dx*dx + dy*dy + dz*dz)
    
    @jit(nopython=True)
    def _calc_fill(distance, radius, voxel_size_half):
        """Расчёт коэффициента заполнения воксела"""
        if distance <= radius:
            return 1.0
        elif distance <= radius + voxel_size_half:
            return 1.0 - (distance - radius) / voxel_size_half
        else:
            return 0.0
    
    @jit(nopython=True)
    def _distance_between_segments_numba(p1, p2, p3, p4):
        """Оптимизированный расчёт расстояния между отрезками (Numba JIT для коллизий)"""
        d1x = p2[0] - p1[0]
        d1y = p2[1] - p1[1]
        d1z = p2[2] - p1[2]
        
        d2x = p4[0] - p3[0]
        d2y = p4[1] - p3[1]
        d2z = p4[2] - p3[2]
        
        rx = p1[0] - p3[0]
        ry = p1[1] - p3[1]
        rz = p1[2] - p3[2]
        
        a = d1x*d1x + d1y*d1y + d1z*d1z
        e = d2x*d2x + d2y*d2y + d2z*d2z
        f = d2x*rx + d2y*ry + d2z*rz
        
        s = 0.0
        t = 0.0
        
        if a <= 1e-12 and e <= 1e-12:
            return np.sqrt(rx*rx + ry*ry + rz*rz)
        
        if a <= 1e-12:
            t = max(0.0, min(1.0, f / e))
        elif e <= 1e-12:
            c = d1x*rx + d1y*ry + d1z*rz
            s = max(0.0, min(1.0, -c / a))
        else:
            b = d1x*d2x + d1y*d2y + d1z*d2z
            c = d1x*rx + d1y*ry + d1z*rz
            denom = a*e - b*b
            
            if abs(denom) > 1e-12:
                s = (b*f - c*e) / denom
                s = max(0.0, min(1.0, s))
                t = (b*s + f) / e
                
                if t < 0.0:
                    t = 0.0
                    s = max(0.0, min(1.0, -c / a))
                elif t > 1.0:
                    t = 1.0
                    s = max(0.0, min(1.0, (b - c) / a))
            else:
                s = 0.0
                t = max(0.0, min(1.0, f / e))
        
        c1x = p1[0] + s * d1x
        c1y = p1[1] + s * d1y
        c1z = p1[2] + s * d1z
        
        c2x = p3[0] + t * d2x
        c2y = p3[1] + t * d2y
        c2z = p3[2] + t * d2z
        
        dx = c1x - c2x
        dy = c1y - c2y
        dz = c1z - c2z
        
        return np.sqrt(dx*dx + dy*dy + dz*dz)

else:
    # Fallback версии без Numba ускорения
    def _point_to_line_distance_numba(point, line_start, line_end):
        line_vec = line_end - line_start
        line_len_sq = np.dot(line_vec, line_vec)
        if line_len_sq < 1e-12:
            return np.linalg.norm(point - line_start)
        t = np.dot(point - line_start, line_vec) / line_len_sq
        t = max(0.0, min(1.0, t))
        nearest = line_start + t * line_vec
        return np.linalg.norm(point - nearest)
    
    def _calc_fill(distance, radius, voxel_size_half):
        """Расчёт коэффициента заполнения воксела"""
        if distance <= radius:
            return 1.0
        elif distance <= radius + voxel_size_half:
            return 1.0 - (distance - radius) / voxel_size_half
        else:
            return 0.0    
    
    def _distance_between_segments_numba(p1, p2, p3, p4):
        d1 = p2 - p1
        d2 = p4 - p3
        r = p1 - p3
        a = np.dot(d1, d1)
        e = np.dot(d2, d2)
        f = np.dot(d2, r)
        
        if a <= 1e-12 and e <= 1e-12:
            return np.linalg.norm(r)
        
        if a <= 1e-12:
            t = np.clip(f / e, 0.0, 1.0)
            s = 0.0
        elif e <= 1e-12:
            s = np.clip(-np.dot(d1, r) / a, 0.0, 1.0)
            t = 0.0
        else:
            b = np.dot(d1, d2)
            c = np.dot(d1, r)
            denom = a*e - b*b
            
            if abs(denom) > 1e-12:
                s = np.clip((b*f - c*e) / denom, 0.0, 1.0)
                t = (b*s + f) / e
                if t < 0.0:
                    t = 0.0
                    s = np.clip(-c / a, 0.0, 1.0)
                elif t > 1.0:
                    t = 1.0
                    s = np.clip((b - c) / a, 0.0, 1.0)
            else:
                s = 0.0
                t = np.clip(f / e, 0.0, 1.0)
        
        closest1 = p1 + s * d1
        closest2 = p3 + t * d2
        return np.linalg.norm(closest1 - closest2)

# ==========================================
# ФИЗИЧЕСКИЕ ПАРАМЕТРЫ МОДЕЛИ
# ==========================================
class PhysicsConfig:
    """Централизованное хранение физических параметров модели"""
    
    # Проводимости материалов (См/м)
    SIGMA_CNT_PARALLEL = 5e6  # Проводимость ОУНТ вдоль оси
    SIGMA_CNT_PERPENDICULAR = 1e-3  # Проводимость ОУНТ поперёк оси
    SIGMA_AEROGEL = 1e-6  # Проводимость аэрогеля (изолятор)
    
    # Параметры туннелирования
    TUNNELING_BETA = 3  # нм^-1 - параметр затухания волновой функции
    TUNNELING_G0 = 2e-5  # См - базовая туннельная проводимость при нулевом зазоре
    MAX_TUNNEL_DISTANCE = 1.2  # нм - максимальное расстояние туннелирования
    
    # Параметры перколяции
    PERCOLATION_THRESHOLD = 0.15  # Критическая объёмная доля (эмпирика для изотропных стержней)
    PERCOLATION_EXPONENT = 2.0  # Критический показатель
    
    # Контактное сопротивление (эмпирические коэффициенты)
    CONTACT_FACTOR_PURE_CNT = 0.1  # Без аэрогеля
    CONTACT_FACTOR_LOW_AEROGEL = 0.05  # Аэрогель < 50%
    CONTACT_FACTOR_HIGH_AEROGEL = 0.01  # Аэрогель > 50%
    
    # Извилистость проводящего пути (зависит от ориентации)
    TORTUOSITY_ALIGNED_LOW = 1.1  # Джиттер < 5°
    TORTUOSITY_ALIGNED_MED = 1.3  # Джиттер 5-15°
    TORTUOSITY_ALIGNED_HIGH = 1.7  # Джиттер > 15°
    TORTUOSITY_RANDOM = 2.5  # Случайная ориентация


# ==========================================
# КЛАСС РАСЧЁТА ПРОВОДИМОСТИ (с корректной физикой)
# ==========================================
class ConductivityCalculator:
    """
    Класс для расчёта электрической проводимости композита.
    
    Физически корректная реализация:
    - Геометрия хранится в нм, проводимость в См/м
    - Разделены σ (проводимость материала, См/м) и G (проводимость контакта, См)
    - Учитывается анизотропия относительно направления связи между вокселями
    - Туннельный зазор рассчитывается через реальное расстояние между осями трубок
    """
    
    def __init__(self, field_size, voxel_size):
        self.field_size = field_size  # нм
        self.voxel_size = voxel_size  # нм
        self.n = int(field_size / voxel_size)
        
        # Разреженное хранение вокселей
        self.voxels = {}
        
        # Физические константы из конфига
        self.config = PhysicsConfig()
        
        # Хранение осей трубок для расчёта истинного зазора
        self.tube_axes = {}  # {tube_id: (p1, p2)}
        
        print(f"📐 Вокселизация: {self.n}×{self.n}×{self.n} = {self.n**3:,} вокселей")
        print(f"   Размер воксела: {self.voxel_size} нм")
    
    def idx_to_coord(self, i, j, k):
        """Преобразование индекса воксела в координату центра (нм)"""
        return np.array([
            (i + 0.5) * self.voxel_size,
            (j + 0.5) * self.voxel_size,
            (k + 0.5) * self.voxel_size
        ])
    
    def coord_to_idx(self, coord):
        """Преобразование координаты в индекс воксела"""
        i = int(coord[0] / self.voxel_size)
        j = int(coord[1] / self.voxel_size)
        k = int(coord[2] / self.voxel_size)
        return (i, j, k)
    
    def voxelize_nanotube(self, axis, radius, tube_id):
        """Вокселизация одной нанотрубки"""
        p1, p2 = axis
        
        # ИСПРАВЛЕНО: Сохраняем ось трубки для расчёта истинного зазора
        self.tube_axes[tube_id] = (p1, p2)
        
        length = np.linalg.norm(p2 - p1)
        direction = (p2 - p1) / (length + 1e-12)
        
        n_steps = int(length / (self.voxel_size / 2)) + 1
        voxels_added = 0
        
        for step in range(n_steps):
            t = step / max(n_steps - 1, 1)
            point = p1 + t * (p2 - p1)
            
            idx = self.coord_to_idx(point)
            if not self._is_valid_index(idx):
                continue
            
            r_vox = int(radius / self.voxel_size) + 1
            
            for di in range(-r_vox, r_vox + 1):
                for dj in range(-r_vox, r_vox + 1):
                    for dk in range(-r_vox, r_vox + 1):
                        voxel_idx = (idx[0] + di, idx[1] + dj, idx[2] + dk)
                        
                        if not self._is_valid_index(voxel_idx):
                            continue
                        
                        voxel_center = self.idx_to_coord(*voxel_idx)
                        dist_to_axis = _point_to_line_distance_numba(voxel_center, p1, p2)
                        
                        if dist_to_axis <= radius + self.voxel_size * 0.5:
                            fill_fraction = _calc_fill(dist_to_axis, radius, self.voxel_size * 0.5)
                            
                            if voxel_idx not in self.voxels:
                                self.voxels[voxel_idx] = {
                                    'type': 1,
                                    'fill': fill_fraction,
                                    'orientation': direction.copy(),
                                    'sigma': 0.0,
                                    'tube_ids': {tube_id}
                                    # ИСПРАВЛЕНО: Не сохраняем один радиус, т.к. через воксель может проходить несколько трубок
                                }
                                voxels_added += 1
                            else:
                                existing = self.voxels[voxel_idx]
                                if 'tube_ids' not in existing:
                                    existing['tube_ids'] = set()
                                existing['tube_ids'].add(tube_id)
                                
                                total_fill = existing['fill'] + fill_fraction
                                existing['orientation'] = (
                                    existing['orientation'] * existing['fill'] + 
                                    direction * fill_fraction
                                ) / (total_fill + 1e-12)
                                existing['fill'] = min(1.0, total_fill)
        
        return voxels_added
    
    def voxelize_particle(self, axis, radius, length, particle_type):
        """Вокселизация частицы аэрогеля"""
        p1, p2 = axis
        center = (p1 + p2) / 2
        effective_radius = radius + length / 2
        
        r_vox = int(effective_radius / self.voxel_size) + 2
        center_idx = self.coord_to_idx(center)
        
        voxels_added = 0
        
        for di in range(-r_vox, r_vox + 1):
            for dj in range(-r_vox, r_vox + 1):
                for dk in range(-r_vox, r_vox + 1):
                    voxel_idx = (center_idx[0] + di, center_idx[1] + dj, center_idx[2] + dk)
                    
                    if not self._is_valid_index(voxel_idx):
                        continue
                    
                    voxel_center = self.idx_to_coord(*voxel_idx)
                    dist = np.linalg.norm(voxel_center - center)
                    
                    if dist <= effective_radius:
                        if voxel_idx not in self.voxels:
                            self.voxels[voxel_idx] = {
                                'type': 2 + particle_type,
                                'fill': 0.3,
                                'orientation': np.array([0, 0, 1]),
                                'sigma': 0.0
                            }
                            voxels_added += 1
        
        return voxels_added
    
    def calculate_local_conductivities(self):
        """
        Расчёт локальных проводимостей вокселей с учётом анизотропии.
        
        Анизотропия будет учитываться при расчёте контактов между вокселями
        в зависимости от направления связи (X, Y или Z).
        """
        print("\n⚡ Расчёт локальных проводимостей с анизотропией...")
        sys.stdout.flush()
        
        for idx, data in self.voxels.items():
            if data['type'] == 1:  # ОУНТ
                orientation = data['orientation'] / (np.linalg.norm(data['orientation']) + 1e-12)
                fill = data['fill']
                
                # Сохраняем нормализованную ориентацию
                data['orientation'] = orientation
                
                # Усреднённая проводимость для воксела (будет уточняться для каждой связи)
                sigma_isotropic = (
                    self.config.SIGMA_CNT_PARALLEL + 
                    2 * self.config.SIGMA_CNT_PERPENDICULAR
                ) / 3
                
                data['sigma'] = sigma_isotropic * fill
            
            elif data['type'] >= 2:  # Аэрогель
                data['sigma'] = self.config.SIGMA_AEROGEL * data['fill']
    
    def calculate_contact_conductance(self, idx1, idx2):
        """
        Расчёт проводимости контакта между двумя вокселями.
        
        ИСПРАВЛЕНО: 
        1. Возвращает G [См] (проводимость контакта), а не σ [См/м]
        2. Учитывает направление связи для анизотропии
        3. Туннельный зазор через реальное расстояние между осями трубок
        """
        if idx1 not in self.voxels or idx2 not in self.voxels:
            return 0.0
        
        data1 = self.voxels[idx1]
        data2 = self.voxels[idx2]
        
        # Только между ОУНТ
        if data1['type'] != 1 or data2['type'] != 1:
            return 0.0
        
        # Определяем направление связи между вокселями
        link_direction = np.array([
            idx2[0] - idx1[0],
            idx2[1] - idx1[1],
            idx2[2] - idx1[2]
        ], dtype=float)
        link_direction /= (np.linalg.norm(link_direction) + 1e-12)
        
        # СЛУЧАЙ 1: Воксели принадлежат одной трубке
        if 'tube_ids' in data1 and 'tube_ids' in data2:
            common_tubes = data1['tube_ids'] & data2['tube_ids']
            if common_tubes:
                # Проводимость по скелету трубки: G = σ * A / L
                # С учётом анизотропии относительно направления связи
                
                # Проекция ориентации на направление связи
                orientation1 = data1['orientation']
                alignment1 = abs(np.dot(orientation1, link_direction))
                
                # Анизотропная проводимость
                sigma1 = (alignment1 * self.config.SIGMA_CNT_PARALLEL + 
                         (1 - alignment1) * self.config.SIGMA_CNT_PERPENDICULAR) * data1['fill']
                
                orientation2 = data2['orientation']
                alignment2 = abs(np.dot(orientation2, link_direction))
                sigma2 = (alignment2 * self.config.SIGMA_CNT_PARALLEL + 
                         (1 - alignment2) * self.config.SIGMA_CNT_PERPENDICULAR) * data2['fill']
                
                sigma_avg = (sigma1 + sigma2) / 2
                
                # Площадь сечения воксела (нм²)
                A_voxel = self.voxel_size ** 2
                
                # Расстояние между центрами (нм)
                L = self.voxel_size
                
                # Конвертация: σ [См/м], A [нм²], L [нм] -> G [См]
                G_intrinsic = sigma_avg * (A_voxel * 1e-18) / (L * 1e-9)
                
                return G_intrinsic
        
        # СЛУЧАЙ 2: Разные трубки - туннелирование
        # ИСПРАВЛЕНО: Используем истинное расстояние между осями трубок
        
        # Находим все трубки в обоих вокселях
        tubes1 = data1.get('tube_ids', set())
        tubes2 = data2.get('tube_ids', set())
        
        if not tubes1 or not tubes2:
            return 0.0
        
        # Ищем минимальное расстояние между осями всех пар трубок
        min_gap = float('inf')
        
        for tid1 in tubes1:
            for tid2 in tubes2:
                if tid1 == tid2:
                    continue
                
                if tid1 in self.tube_axes and tid2 in self.tube_axes:
                    axis1 = self.tube_axes[tid1]
                    axis2 = self.tube_axes[tid2]
                    
                    # Реальное расстояние между осями (нм)
                    distance = _distance_between_segments_numba(
                        np.array(axis1[0], dtype=np.float64),
                        np.array(axis1[1], dtype=np.float64),
                        np.array(axis2[0], dtype=np.float64),
                        np.array(axis2[1], dtype=np.float64)
                    )
                    
                    # Зазор с учётом радиусов
                    # Используем радиус, переданный из simulator
                    tube_radius = getattr(self, 'tube_radius', 1.0)  # нм, дефолт если не задан
                    
                    gap = max(0.0, distance - 2 * tube_radius)
                    min_gap = min(min_gap, gap)
        
        # Если не нашли пары трубок, используем расстояние между центрами вокселей
        if min_gap == float('inf'):
            coord1 = self.idx_to_coord(*idx1)
            coord2 = self.idx_to_coord(*idx2)
            distance = np.linalg.norm(coord2 - coord1)  # нм
            
            # Эффективный радиус из fill
            r_eff = self.voxel_size * 0.5 * (data1['fill'] + data2['fill']) / 2
            min_gap = max(0.0, distance - 2 * r_eff)
        
        if min_gap > self.config.MAX_TUNNEL_DISTANCE:
            return 0.0
        
        # Туннельная проводимость: G = G0 * exp(-β * gap)
        G_tunnel = self.config.TUNNELING_G0 * np.exp(-self.config.TUNNELING_BETA * min_gap)
        
        # Учёт ориентации (параллельные трубки - лучший контакт)
        if 'orientation' in data1 and 'orientation' in data2:
            dot_product = np.abs(np.dot(data1['orientation'], data2['orientation']))
            orientation_factor = 0.5 + 0.5 * dot_product
            G_tunnel *= orientation_factor
        
        return G_tunnel
    
    def find_percolating_clusters(self):
        """Поиск перколяционных кластеров методом Union-Find"""
        print("\n🔍 Поиск перколяционных кластеров...")
        sys.stdout.flush()
        
        if not self.voxels:
            return False, {}, set()
        
        parent = {idx: idx for idx in self.voxels}
        
        def find(idx):
            if parent[idx] != idx:
                parent[idx] = find(parent[idx])
            return parent[idx]
        
        def union(idx1, idx2):
            root1 = find(idx1)
            root2 = find(idx2)
            if root1 != root2:
                parent[root1] = root2
        
        shifts = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
        connections = 0
        
        for idx in self.voxels:
            for dx, dy, dz in shifts:
                neighbor_idx = (idx[0] + dx, idx[1] + dy, idx[2] + dz)
                if neighbor_idx in self.voxels:
                    G = self.calculate_contact_conductance(idx, neighbor_idx)
                    if G > 1e-15:
                        union(idx, neighbor_idx)
                        connections += 1
        
        print(f"   ✓ Найдено связей: {connections:,}")
        sys.stdout.flush()
        
        cluster_map = {idx: find(idx) for idx in self.voxels}
        
        left_clusters = set()
        right_clusters = set()
        
        for idx, cluster_id in cluster_map.items():
            if self.voxels[idx]['type'] == 1:
                if idx[0] == 0:
                    left_clusters.add(cluster_id)
                if idx[0] == self.n - 1:
                    right_clusters.add(cluster_id)
        
        percolating_clusters = left_clusters & right_clusters
        
        print(f"   ✓ Кластеров на левой границе: {len(left_clusters)}")
        print(f"   ✓ Кластеров на правой границе: {len(right_clusters)}")
        print(f"   ✓ Перколяционных кластеров: {len(percolating_clusters)}")
        sys.stdout.flush()
        
        return len(percolating_clusters) > 0, cluster_map, percolating_clusters
    
    def calculate_effective_conductivity_kirchhoff(self, percolating_cluster_nodes):
        """
        Расчёт проводимости методом Кирхгоффа.
        
        ИСПРАВЛЕНО: Правильные единицы измерения.
        - G [См] - проводимость контактов
        - Потенциал V [В] - безразмерный (0 или 1)
        - Ток I [А] = G * V
        - σ [См/м] = I / (∆V * L_m)
        """
        if not SCIPY_AVAILABLE:
            print("   ⚠️ SciPy недоступна, используется упрощённый метод")
            return None
        
        print("   🧮 Решение уравнений Кирхгоффа...")
        sys.stdout.flush()
        
        nodes = list(percolating_cluster_nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        N = len(nodes)
        
        A = lil_matrix((N, N))
        b = np.zeros(N)
        
        for i, node in enumerate(nodes):
            # Граничные условия: V = 1 на левой границе, V = 0 на правой
            if node[0] == 0:
                A[i, i] = 1.0
                b[i] = 1.0
                continue
            
            if node[0] == self.n - 1:
                A[i, i] = 1.0
                b[i] = 0.0
                continue
            
            # Внутренние узлы: закон Кирхгофа (сумма токов = 0)
            sum_conductances = 0.0
            
            for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                neighbor = (node[0] + dx, node[1] + dy, node[2] + dz)
                if neighbor in node_to_idx:
                    j = node_to_idx[neighbor]
                    G = self.calculate_contact_conductance(node, neighbor)  # [См]
                    if G > 1e-15:
                        A[i, j] = -G
                        sum_conductances += G
            
            A[i, i] = sum_conductances if sum_conductances > 0 else 1.0
        
        try:
            potentials = spsolve(A.tocsc(), b)
            
            # Вычисляем полный ток через левую границу
            total_current = 0.0  # [А] = [См] * [В]
            for i, node in enumerate(nodes):
                if node[0] == 0:
                    for dx, dy, dz in [(1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                        neighbor = (node[0] + dx, node[1] + dy, node[2] + dz)
                        if neighbor in node_to_idx:
                            j = node_to_idx[neighbor]
                            G = self.calculate_contact_conductance(node, neighbor)  # [См]
                            total_current += G * (1.0 - potentials[j])  # [А]
            
            # ИСПРАВЛЕНО: Правильная конвертация единиц
            # σ = I / (∆V * A / L) = I * L / (∆V * A)
            # Где A - поперечное сечение, L - длина в направлении тока
            
            # Длина поля в метрах
            L_m = self.field_size * 1e-9  # м
            
            # Площадь поперечного сечения в метрах²
            A_m = (self.field_size * 1e-9) ** 2  # м²
            
            # ∆V = 1 В
            delta_V = 1.0
            
            # σ [См/м] = I [А] * L [м] / (∆V [В] * A [м²])
            sigma_kirchhoff = total_current * L_m / (delta_V * A_m)
            
            print(f"   ✓ Расчёт Кирхгоффа завершён: {sigma_kirchhoff:.2e} См/м")
            sys.stdout.flush()
            
            return sigma_kirchhoff
        
        except Exception as e:
            print(f"   ❌ Ошибка расчёта Кирхгоффа: {e}")
            sys.stdout.flush()
            return None
    
    def calculate_effective_conductivity_simple(self, percolates, percolating_clusters, cluster_map):
        """Расчёт эффективной проводимости"""
        print("\n⚡ РАСЧЁТ ЭФФЕКТИВНОЙ ПРОВОДИМОСТИ")
        print("="*50)
        sys.stdout.flush()
        
        filled_voxels = [v for v in self.voxels.values() if v['type'] == 1]
        voxel_volume = self.voxel_size ** 3
        real_volume_nm3 = sum(voxel_volume * v.get('fill', 1.0) for v in filled_voxels)

        # Геометрический расчёт
        if hasattr(self, 'simulator'):
            volumes = self.simulator.calculate_actual_volumes()
            volume_fraction = volumes['tube_fraction']
        else:
            volume_fraction = real_volume_nm3 / (self.field_size ** 3)

        if not percolates:
            print("   ❌ Перколяция УНТ не обнаружена")
            
            # НОВАЯ ЛОГИКА: Проверяем есть ли аэрогель
            if hasattr(self, 'simulator') and getattr(self.simulator, 'enable_particles', False):
                # Есть частицы аэрогеля - используем его проводимость
                sigma_matrix = self.config.SIGMA_AEROGEL
                print(f"   ℹ️  Проводимость через матрицу аэрогеля: {sigma_matrix:.2e} См/м")
                sys.stdout.flush()
                return sigma_matrix, {
                    'percolates': False, 
                    'volume_fraction': volume_fraction,
                    'sigma_effective': sigma_matrix,
                    'method': 'Проводимость матрицы (аэрогель)'
                }
            else:
                # Нет аэрогеля - изолятор (перестраховка: используем 1e-6)
                sigma_insulator = 1e-6  # Типичное для изоляторов
                print(f"   ℹ️  Система изолятор: σ = {sigma_insulator:.2e} См/м")
                sys.stdout.flush()
                return sigma_insulator, {
                    'percolates': False, 
                    'volume_fraction': volume_fraction,
                    'sigma_effective': sigma_insulator,
                    'method': 'Проводимость изолятора (по умолчанию)'
                }
        
        percolating_voxels = []
        percolating_sigmas = []
        
        for idx, cluster_id in cluster_map.items():
            if cluster_id in percolating_clusters:
                if self.voxels[idx]['type'] == 1:
                    percolating_voxels.append(idx)
                    percolating_sigmas.append(self.voxels[idx]['sigma'])
        
        print(f"   ℹ️  Проводящих вокселей: {len(percolating_voxels):,}")
        sys.stdout.flush()

        filled_voxels = [v for v in self.voxels.values() if v['type'] == 1]
        real_volume_nm3 = 0.0
        voxel_volume = self.voxel_size ** 3

        for voxel in filled_voxels:
            fill_fraction = voxel.get('fill', 1.0)
            real_volume_nm3 += voxel_volume * fill_fraction

        volume_fraction_voxels = real_volume_nm3 / (self.field_size ** 3)

        # Расчёт с учётом обрезки
        if hasattr(self, 'simulator'):
            volumes = self.simulator.calculate_actual_volumes()
            volume_fraction = volumes['tube_fraction']
            
            print(f"   📊 Объёмные доли:")
            print(f"      • Из вокселей:        {volume_fraction_voxels:.4%}")
            print(f"      • УНТ (с обрезкой):   {volumes['tube_fraction']:.4%}")
            if volumes['particle_fraction'] > 0:
                print(f"      • Аэрогель:           {volumes['particle_fraction']:.4%}")
                print(f"      • Всего:              {volumes['total_fraction']:.4%}")
            sys.stdout.flush()
        else:
            volume_fraction = volume_fraction_voxels

        percolating_fraction = len(percolating_voxels) / max(1, len(filled_voxels))
        phi_c = self.config.PERCOLATION_THRESHOLD
        
        if volume_fraction > phi_c:
            t = self.config.PERCOLATION_EXPONENT
            sigma_percolation = self.config.SIGMA_CNT_PARALLEL * ((volume_fraction - phi_c) / phi_c) ** t
        else:
            sigma_percolation = 0.0
        
        if len(percolating_sigmas) > 0:
            sigma_geometric = np.exp(np.mean(np.log(np.array(percolating_sigmas) + 1e-12)))
            
            # Контактное сопротивление из config
            if hasattr(self, 'simulator'):
                has_aerogel = getattr(self.simulator, 'enable_particles', False)
                if has_aerogel:
                    aerogel_ratio = getattr(self.simulator, 'particle_to_total_ratio', 0.0)
                    if aerogel_ratio > 0.5:
                        contact_resistance_factor = self.config.CONTACT_FACTOR_HIGH_AEROGEL
                    else:
                        contact_resistance_factor = self.config.CONTACT_FACTOR_LOW_AEROGEL
                else:
                    contact_resistance_factor = self.config.CONTACT_FACTOR_PURE_CNT
            else:
                contact_resistance_factor = self.config.CONTACT_FACTOR_LOW_AEROGEL
            
            sigma_geometric_with_contacts = sigma_geometric * contact_resistance_factor
        else:
            sigma_geometric = 0.0
            sigma_geometric_with_contacts = 0.0
            contact_resistance_factor = self.config.CONTACT_FACTOR_LOW_AEROGEL
        
        # Извилистость из config
        if hasattr(self, 'simulator'):
            orientation_mode = self.simulator.orientation_mode
            if orientation_mode == 'aligned':
                jitter = getattr(self.simulator, 'aligned_jitter_deg', 5.0)
                if jitter < 5:
                    tortuosity = self.config.TORTUOSITY_ALIGNED_LOW
                elif jitter < 15:
                    tortuosity = self.config.TORTUOSITY_ALIGNED_MED
                else:
                    tortuosity = self.config.TORTUOSITY_ALIGNED_HIGH
            else:
                tortuosity = self.config.TORTUOSITY_RANDOM
        else:
            tortuosity = self.config.TORTUOSITY_ALIGNED_MED
        
        sigma_kirchhoff = None
        if SCIPY_AVAILABLE and len(percolating_voxels) > 0 and len(percolating_voxels) < 1000000:
            sigma_kirchhoff = self.calculate_effective_conductivity_kirchhoff(set(percolating_voxels))
        
        # ИСПРАВЛЕНО: Проверка на None, а не на truthiness
        if sigma_kirchhoff is not None and sigma_kirchhoff > 0:
            sigma_effective = sigma_kirchhoff
            method = "Кирхгофф"
        else:
            sigma_effective = sigma_geometric_with_contacts / tortuosity
            method = "Геометрическое среднее"
        
        results = {
            'sigma_percolation': sigma_percolation,
            'sigma_geometric': sigma_geometric,
            'sigma_geometric_with_contacts': sigma_geometric_with_contacts,
            'sigma_effective': sigma_effective,
            'sigma_kirchhoff': sigma_kirchhoff,
            'volume_fraction': volume_fraction,
            'percolating_fraction': percolating_fraction,
            'percolating_voxels': percolating_voxels,
            'phi_c': phi_c,
            'tortuosity': tortuosity,
            'contact_resistance_factor': contact_resistance_factor,
            'percolates': True,
            'method': method
        }
        
        print(f"\n   📈 РЕЗУЛЬТАТЫ РАСЧЁТА:")
        print(f"   ├─ Метод: {method}")
        print(f"   ├─ Модель перколяции: {sigma_percolation:.2e} См/м")
        print(f"   ├─ Геометрическое среднее: {sigma_geometric:.2e} См/м")
        print(f"   ├─ С учётом контактов (×{contact_resistance_factor}): {sigma_geometric_with_contacts:.2e} См/м")
        if sigma_kirchhoff is not None:
            print(f"   ├─ Расчёт Кирхгоффа: {sigma_kirchhoff:.2e} См/м")
        print(f"   ├─ Извилистость пути: {tortuosity:.2f}")
        print(f"   └─ ЭФФЕКТИВНАЯ ПРОВОДИМОСТЬ: {sigma_effective:.2e} См/м")
        sys.stdout.flush()
        
        return sigma_effective, results
    
    def _is_valid_index(self, idx):
        """Проверка валидности индекса воксела"""
        i, j, k = idx
        return 0 <= i < self.n and 0 <= j < self.n and 0 <= k < self.n


# ==========================================
# СИМУЛЯТОР НАНОТРУБОК
# ==========================================
class EnhancedNanotubeSimulator:
    def __init__(self):
        """Инициализация симулятора с параметрами по умолчанию"""
        
        # Параметры генерации случайных чисел
        self.random_seed = None
        
        # Параметры нанотрубок
        self.num_tubes = 1000
        self.outer_radius = 1.0
        self.inner_radius = 0.66
        self.tube_length = 250.0
        
        # Параметры частиц аэрогеля
        self.enable_particles = True
        self.num_particle_types = 2
        
        self.particle_params = [
            {'radius': 15.0, 'length': 40.0, 'color': 'gray', 'opacity': 0.4},
            {'radius': 25.0, 'length': 60.0, 'color': 'lightgray', 'opacity': 0.3},
        ]
        
        self.particle_to_total_ratio = 0.0
        self.particle_type_distribution = [0.6, 0.4]
        
        # Параметры поля
        self.field_size = 400.0
        
        # Параметры границ генерации
        self.clip_protruding_parts = True
        
        # Параметры электропроводности
        self.enable_conductivity = True
        self.voxel_size = 2.0
        self.conductivity_calculator = None
        self.conductivity_results = None
        
        # Параметры коллизий
        self.min_gap = 0.34
        self.min_gap_floor = self.min_gap
        self.min_gap_particles = 0.5
        
        # Параметры генерации
        self.max_attempts_multiplier = 1000
        self.max_attempts_custom = None
        self.max_attempts_ceiling = 50000
        
        # Параметры отображения
        self.show_percolation_analysis = False
        self.show_conductive_paths = True
        self.cluster_highlight_mode = 'tubes'
        
        # Параметры ориентации нанотрубок
        self.orientation_mode = 'aligned'
        self.aligned_dir = (0, 0, 1)
        self.aligned_jitter_deg = 4.0
        
        # Параметры визуализации
        self.tube_color = 'cyan'
        self.tube_opacity = 0.3
        self.background_color = 'black'
        
        # Хранилища объектов
        self.tubes = []
        self.tube_axes = []
        self.particles = []
        self.particle_axes = []
        self.particle_types = []
        self.plotter = None
        
        # Данные кластеров
        self.last_cluster_map = None
        self.last_percolating_clusters = None
    
    def set_random_seed(self):
        """Установка сида генератора случайных чисел"""
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            print(f"🎲 Установлен сид: {self.random_seed}")
        else:
            self.random_seed = int(time.time() * 1000) % 2**32
            np.random.seed(self.random_seed)
            print(f"🎲 Используется случайный сид: {self.random_seed}")
    
    def create_ellipsoid_particle(self, radius, length):
        """Создание геометрии частицы аэрогеля (эллипсоид)"""
        n_circumference = 30
        n_length = 15
        n_hemisphere = 10
        
        all_points = []
        all_faces = []
        
        theta = np.linspace(0, 2*np.pi, n_circumference, endpoint=False)
        
        # Цилиндрическая часть
        z_cylinder = np.linspace(-length/2, length/2, n_length)
        cylinder_start = len(all_points)
        
        for z in z_cylinder:
            for t in theta:
                all_points.append([radius*np.cos(t), radius*np.sin(t), z])
        
        for i in range(n_length - 1):
            for j in range(n_circumference):
                jn = (j + 1) % n_circumference
                p0 = cylinder_start + i*n_circumference + j
                p1 = cylinder_start + i*n_circumference + jn
                p2 = cylinder_start + (i+1)*n_circumference + jn
                p3 = cylinder_start + (i+1)*n_circumference + j
                all_faces.extend([3, p0, p1, p2])
                all_faces.extend([3, p0, p2, p3])
        
        # Верхняя полусфера
        phi_top = np.linspace(0, np.pi/2, n_hemisphere)
        top_hemisphere_start = len(all_points)
        
        for p in phi_top[1:]:
            for t in theta:
                x = radius * np.sin(p) * np.cos(t)
                y = radius * np.sin(p) * np.sin(t)
                z = length/2 + radius * np.cos(p)
                all_points.append([x, y, z])
        
        top_point = len(all_points)
        all_points.append([0, 0, length/2 + radius])
        
        top_cylinder_ring = cylinder_start + (n_length-1)*n_circumference
        first_hemisphere_ring = top_hemisphere_start
        
        if n_hemisphere > 1:
            for j in range(n_circumference):
                jn = (j + 1) % n_circumference
                p0 = top_cylinder_ring + j
                p1 = top_cylinder_ring + jn
                p2 = first_hemisphere_ring + jn
                p3 = first_hemisphere_ring + j
                all_faces.extend([3, p0, p1, p2])
                all_faces.extend([3, p0, p2, p3])
        
        for i in range(n_hemisphere - 2):
            for j in range(n_circumference):
                jn = (j + 1) % n_circumference
                p0 = top_hemisphere_start + i*n_circumference + j
                p1 = top_hemisphere_start + i*n_circumference + jn
                p2 = top_hemisphere_start + (i+1)*n_circumference + jn
                p3 = top_hemisphere_start + (i+1)*n_circumference + j
                all_faces.extend([3, p0, p1, p2])
                all_faces.extend([3, p0, p2, p3])
        
        last_ring = top_hemisphere_start + (n_hemisphere-2)*n_circumference
        for j in range(n_circumference):
            jn = (j + 1) % n_circumference
            p0 = last_ring + j
            p1 = last_ring + jn
            all_faces.extend([3, p0, p1, top_point])
        
        # Нижняя полусфера
        phi_bottom = np.linspace(np.pi/2, np.pi, n_hemisphere)
        bottom_hemisphere_start = len(all_points)
        
        for p in phi_bottom[1:]:
            for t in theta:
                x = radius * np.sin(p) * np.cos(t)
                y = radius * np.sin(p) * np.sin(t)
                z = -length/2 + radius * np.cos(p)
                all_points.append([x, y, z])
        
        bottom_point = len(all_points)
        all_points.append([0, 0, -length/2 - radius])
        
        bottom_cylinder_ring = cylinder_start
        first_bottom_hemisphere_ring = bottom_hemisphere_start
        
        if n_hemisphere > 1:
            for j in range(n_circumference):
                jn = (j + 1) % n_circumference
                p0 = bottom_cylinder_ring + j
                p1 = bottom_cylinder_ring + jn
                p2 = first_bottom_hemisphere_ring + jn
                p3 = first_bottom_hemisphere_ring + j
                all_faces.extend([3, p0, p3, p2])
                all_faces.extend([3, p0, p2, p1])
        
        for i in range(n_hemisphere - 2):
            for j in range(n_circumference):
                jn = (j + 1) % n_circumference
                p0 = bottom_hemisphere_start + i*n_circumference + j
                p1 = bottom_hemisphere_start + i*n_circumference + jn
                p2 = bottom_hemisphere_start + (i+1)*n_circumference + jn
                p3 = bottom_hemisphere_start + (i+1)*n_circumference + j
                all_faces.extend([3, p0, p3, p2])
                all_faces.extend([3, p0, p2, p1])
        
        last_bottom_ring = bottom_hemisphere_start + (n_hemisphere-2)*n_circumference
        for j in range(n_circumference):
            jn = (j + 1) % n_circumference
            p0 = last_bottom_ring + j
            p1 = last_bottom_ring + jn
            all_faces.extend([3, p0, bottom_point, p1])
        
        points_array = np.array(all_points, dtype=np.float32)
        faces_array = np.array(all_faces, dtype=np.int32)
        return pv.PolyData(points_array, faces_array)
    
    def create_hollow_tube_template(self):
        """Создание шаблона полой нанотрубки"""
        n_circumference = 50
        n_length = 20

        all_points = []
        all_faces = []

        theta = np.linspace(0, 2*np.pi, n_circumference, endpoint=False)
        z_values = np.linspace(-self.tube_length/2, self.tube_length/2, n_length)

        # Внешняя поверхность
        outer_points_start = len(all_points)
        for z in z_values:
            for t in theta:
                all_points.append([self.outer_radius*np.cos(t),
                                   self.outer_radius*np.sin(t), z])

        for i in range(n_length - 1):
            for j in range(n_circumference):
                jn = (j + 1) % n_circumference
                p0 = outer_points_start + i*n_circumference + j
                p1 = outer_points_start + i*n_circumference + jn
                p2 = outer_points_start + (i+1)*n_circumference + jn
                p3 = outer_points_start + (i+1)*n_circumference + j
                all_faces.extend([3, p0, p1, p2])
                all_faces.extend([3, p0, p2, p3])

        # Внутренняя поверхность
        inner_points_start = len(all_points)
        for z in z_values:
            for t in theta:
                all_points.append([self.inner_radius*np.cos(t),
                                   self.inner_radius*np.sin(t), z])

        for i in range(n_length - 1):
            for j in range(n_circumference):
                jn = (j + 1) % n_circumference
                p0 = inner_points_start + i*n_circumference + j
                p1 = inner_points_start + i*n_circumference + jn
                p2 = inner_points_start + (i+1)*n_circumference + jn
                p3 = inner_points_start + (i+1)*n_circumference + j
                all_faces.extend([3, p0, p2, p1])
                all_faces.extend([3, p0, p3, p2])

        # Верхний торец
        top_outer_start = outer_points_start + (n_length - 1)*n_circumference
        top_inner_start = inner_points_start + (n_length - 1)*n_circumference
        for j in range(n_circumference):
            jn = (j + 1) % n_circumference
            p0 = top_outer_start + j
            p1 = top_outer_start + jn
            p2 = top_inner_start + jn
            p3 = top_inner_start + j
            all_faces.extend([3, p0, p1, p2])
            all_faces.extend([3, p0, p2, p3])

        # Нижний торец
        bottom_outer_start = outer_points_start
        bottom_inner_start = inner_points_start
        for j in range(n_circumference):
            jn = (j + 1) % n_circumference
            p0 = bottom_outer_start + j
            p1 = bottom_outer_start + jn
            p2 = bottom_inner_start + jn
            p3 = bottom_inner_start + j
            all_faces.extend([3, p0, p2, p1])
            all_faces.extend([3, p0, p3, p2])

        points_array = np.array(all_points, dtype=np.float32)
        faces_array = np.array(all_faces, dtype=np.int32)
        return pv.PolyData(points_array, faces_array)
    
    def random_orientation_angles(self):
        """Генерация случайных углов ориентации"""
        return (np.random.uniform(0, 360),
                np.random.uniform(0, 360),
                np.random.uniform(0, 360))

    def aligned_orientation_angles(self):
        """Генерация углов для выровненной ориентации с джиттером"""
        base = np.asarray(self.aligned_dir, dtype=float)
        base /= (np.linalg.norm(base) + 1e-12)
        jitter_rad = np.deg2rad(np.random.normal(0.0, self.aligned_jitter_deg/2))
        axis = np.random.normal(size=3)
        axis /= (np.linalg.norm(axis) + 1e-12)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(jitter_rad)*K + (1 - np.cos(jitter_rad))*(K@K)
        v = (R @ base).astype(float)
        yaw = np.degrees(np.arctan2(v[1], v[0]))
        pitch = np.degrees(np.arctan2(np.sqrt(v[0]**2 + v[1]**2), v[2])) - 90
        roll = 0.0
        return (pitch, 0.0, yaw + roll)
    
    def rotation_matrix_from_angles(self, angles):
        """Построение матрицы поворота из углов Эйлера"""
        ax, ay, az = np.radians(angles)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(ax), -np.sin(ax)],
                       [0, np.sin(ax),  np.cos(ax)]])
        Ry = np.array([[ np.cos(ay), 0, np.sin(ay)],
                       [0, 1, 0],
                       [-np.sin(ay), 0, np.cos(ay)]])
        Rz = np.array([[np.cos(az), -np.sin(az), 0],
                       [np.sin(az),  np.cos(az), 0],
                       [0, 0, 1]])
        return Rz @ Ry @ Rx

    def transform_object(self, obj, position, angles):
        """Трансформация объекта (поворот и перенос)"""
        points = obj.points.copy()
        R = self.rotation_matrix_from_angles(angles)
        transformed_points = (points @ R.T) + position
        transformed_obj = pv.PolyData(transformed_points.astype(np.float32), obj.faces)
        return transformed_obj, R

    def get_tube_axis(self, center, rotation_matrix):
        """Получение оси нанотрубки"""
        axis_direction = rotation_matrix @ np.array([0, 0, 1], dtype=float)
        axis_direction /= (np.linalg.norm(axis_direction) + 1e-12)
        half_length = self.tube_length / 2
        return center - axis_direction * half_length, center + axis_direction * half_length
    
    def get_particle_axis(self, center, rotation_matrix, length):
        """Получение оси частицы аэрогеля"""
        axis_direction = rotation_matrix @ np.array([0, 0, 1], dtype=float)
        axis_direction /= (np.linalg.norm(axis_direction) + 1e-12)
        half_length = length / 2
        return center - axis_direction * half_length, center + axis_direction * half_length

    def clip_object_by_bounds(self, obj):
        """Обрезка объекта по границам куба"""
        if not self.clip_protruding_parts:
            return obj
        
        clipped = obj.copy()
        
        clipped = clipped.clip('x', origin=(0, 0, 0), invert=False)
        clipped = clipped.clip('x', origin=(self.field_size, 0, 0), invert=True)
        
        clipped = clipped.clip('y', origin=(0, 0, 0), invert=False)
        clipped = clipped.clip('y', origin=(0, self.field_size, 0), invert=True)
        
        clipped = clipped.clip('z', origin=(0, 0, 0), invert=False)
        clipped = clipped.clip('z', origin=(0, 0, self.field_size), invert=True)
        
        return clipped

    def distance_between_line_segments(self, p1, p2, p3, p4):
        """Расстояние между двумя отрезками (Numba-ускоренная версия)"""
        p1 = np.asarray(p1, dtype=np.float64)
        p2 = np.asarray(p2, dtype=np.float64)
        p3 = np.asarray(p3, dtype=np.float64)
        p4 = np.asarray(p4, dtype=np.float64)
        
        return _distance_between_segments_numba(p1, p2, p3, p4)

    def check_collision_with_objects(self, axis, obj_type='tube', obj_params=None):
        """
        Проверка коллизий с существующими объектами.
        
        ИСПРАВЛЕНО: Правильная геометрия для всех комбинаций объектов:
        - tube-tube: distance - r_tube1 - r_tube2
        - tube-particle: distance - r_tube - r_particle
        - particle-particle: distance - r_particle1 - r_particle2
        
        Args:
            axis: Ось объекта (p1, p2)
            obj_type: 'tube' или 'particle'
            obj_params: Параметры объекта (для частиц - словарь с 'radius')
        """
        min_gap = self.min_gap if obj_type == 'tube' else self.min_gap_particles
        
        # Радиус текущего объекта
        if obj_type == 'tube':
            r_current = self.outer_radius
        else:
            # Для частицы используем реальный радиус из параметров
            if obj_params and 'radius' in obj_params:
                r_current = obj_params['radius']
            else:
                r_current = 15.0  # Fallback, не должно использоваться
        
        # Проверка с существующими трубками
        for existing_axis in self.tube_axes:
            distance = self.distance_between_line_segments(
                axis[0], axis[1], existing_axis[0], existing_axis[1]
            )
            
            # Правильный расчёт эффективного расстояния
            r_tube = self.outer_radius
            effective_distance = distance - r_current - r_tube
            
            if effective_distance < min_gap:
                return True
        
        # Проверка с существующими частицами аэрогеля
        if self.enable_particles:
            for i, particle_axis in enumerate(self.particle_axes):
                distance = self.distance_between_line_segments(
                    axis[0], axis[1], particle_axis[0], particle_axis[1]
                )
                
                # ИСПРАВЛЕНО: Используем реальный радиус существующей частицы
                particle_type = self.particle_types[i]
                r_existing_particle = self.particle_params[particle_type]['radius']
                
                # Правильный расчёт для всех комбинаций
                effective_distance = distance - r_current - r_existing_particle
                
                if effective_distance < min_gap:
                    return True
        
        return False

    def calculate_packing_density(self):
        """Расчёт плотности упаковки композита"""
        field_volume = self.field_size ** 3
        
        tube_volume_single = np.pi * (self.outer_radius**2 - self.inner_radius**2) * self.tube_length
        total_tube_volume = tube_volume_single * len(self.tubes)
        
        total_particle_volume = 0
        if self.enable_particles:
            for i in range(len(self.particle_types)):
                params = self.particle_params[self.particle_types[i]]
                particle_volume = np.pi * params['radius']**2 * params['length'] + \
                                 (4/3) * np.pi * params['radius']**3
                total_particle_volume += particle_volume
        
        total_volume = total_tube_volume + total_particle_volume
        return total_volume / field_volume
    
    def generate_all_objects(self):
        """Совместная генерация нанотрубок и частиц аэрогеля"""
        print("\n" + "="*60)
        print("НАЧАЛО ГЕНЕРАЦИИ")
        print("="*60)
        
        self.set_random_seed()
        
        total_tubes = self.num_tubes
        
        if self.enable_particles:
            total_particles = int(self.num_tubes * self.particle_to_total_ratio / (1 - self.particle_to_total_ratio))
        else:
            total_particles = 0
        
        total_objects = total_tubes + total_particles
        
        print(f"\n📊 План генерации:")
        print(f"   • Нанотрубки: {total_tubes}")
        if self.enable_particles:
            print(f"   • Частицы аэрогеля: {total_particles}")
        print(f"   • Всего объектов: {total_objects}")
        
        template_tube = self.create_hollow_tube_template()
        
        objects_to_generate = ['tube'] * total_tubes
        if self.enable_particles:
            objects_to_generate.extend(['particle'] * total_particles)
        
        np.random.shuffle(objects_to_generate)
        
        if self.max_attempts_custom is not None:
            max_attempts = int(self.max_attempts_custom)
        else:
            max_attempts = int(total_objects * self.max_attempts_multiplier)
            max_attempts = min(max_attempts, self.max_attempts_ceiling)
        
        print(f"\n⚙️ Максимальное количество попыток: {max_attempts:,}")
        print(f"   📏 Границы генерации: [0, {self.field_size:.1f}] нм")
        
        attempts = 0
        tubes_placed = 0
        particles_placed = 0
        
        print(f"\n⏳ Начало генерации...")
        start_time = time.time()
        
        for idx, obj_type in enumerate(objects_to_generate):
            placed = False
            object_attempts = 0
            max_object_attempts = 1000
            
            while not placed and object_attempts < max_object_attempts and attempts < max_attempts:
                attempts += 1
                object_attempts += 1
                
                if attempts % 5000 == 0:
                    elapsed = time.time() - start_time
                    print(f"   ⏳ Попыток: {attempts:,}/{max_attempts:,}, размещено: трубки {tubes_placed}/{total_tubes}, "
                          f"частицы {particles_placed}/{total_particles}, время: {elapsed:.1f}с")
                
                position = np.random.uniform(0.0, self.field_size, 3)
                
                if self.orientation_mode == 'aligned':
                    angles = self.aligned_orientation_angles()
                elif self.orientation_mode == 'random_angles':
                    angles = self.random_orientation_angles()
                else:
                    angles = self.aligned_orientation_angles()
                
                if obj_type == 'tube':
                    transformed, rotation_matrix = self.transform_object(template_tube, position, angles)
                    axis = self.get_tube_axis(position, rotation_matrix)
                    
                    if not self.check_collision_with_objects(axis, 'tube'):
                        if self.clip_protruding_parts:
                            transformed = self.clip_object_by_bounds(transformed)
                        
                        self.tubes.append(transformed)
                        self.tube_axes.append(axis)
                        tubes_placed += 1
                        placed = True
                
                else:
                    distribution = self.particle_type_distribution[:self.num_particle_types]
                    distribution = np.array(distribution)
                    distribution = distribution / distribution.sum()
                    
                    particle_type = np.random.choice(
                        range(self.num_particle_types),
                        p=distribution
                    )
                    params = self.particle_params[particle_type]
                    
                    template_particle = self.create_ellipsoid_particle(params['radius'], params['length'])
                    transformed, rotation_matrix = self.transform_object(template_particle, position, angles)
                    axis = self.get_particle_axis(position, rotation_matrix, params['length'])
                    
                    # ИСПРАВЛЕНО: Передаём параметры частицы для правильной проверки коллизий
                    if not self.check_collision_with_objects(axis, 'particle', params):
                        if self.clip_protruding_parts:
                            transformed = self.clip_object_by_bounds(transformed)
                        
                        self.particles.append(transformed)
                        self.particle_axes.append(axis)
                        self.particle_types.append(particle_type)
                        particles_placed += 1
                        placed = True
            
            if (idx + 1) % max(1, total_objects // 10) == 0:
                progress = (idx + 1) / total_objects * 100
                elapsed = time.time() - start_time
                print(f"   ✅ Прогресс: {progress:.0f}% ({tubes_placed + particles_placed}/{total_objects} объектов), время: {elapsed:.1f}с")
        
        elapsed = time.time() - start_time
        
        print(f"\n📊 Результаты генерации:")
        print(f"   • Размещено нанотрубок: {tubes_placed}/{total_tubes}")
        if self.enable_particles:
            print(f"   • Размещено частиц: {particles_placed}/{total_particles}")
            for i in range(self.num_particle_types):
                count = sum(1 for t in self.particle_types if t == i)
                percentage = (count / len(self.particles) * 100) if self.particles else 0
                params = self.particle_params[i]
                print(f"     - Тип {i+1} ({params['color']}, α={params['opacity']}): {count} ({percentage:.1f}%)")
        print(f"   • Всего попыток: {attempts:,}")
        print(f"   • Плотность упаковки: {self.calculate_packing_density()*100:.3f}%")
        print(f"   • Время генерации: {elapsed:.1f} секунд")
        print("="*60)

    def calculate_actual_volumes(self):
        """Расчёт реального объёма УНТ и аэрогеля с учётом обрезки на границах"""
        field_volume = self.field_size ** 3
        
        # ========== ОБЪЁМ УНТ ==========
        total_tube_volume = 0.0
        
        # ✅ ИСПРАВЛЕНО: Используем tube_axes, а не tubes!
        for axis in self.tube_axes:
            start, end = axis  # axis это кортеж (p1, p2)
            direction = end - start
            
            # Находим параметры t, где трубка пересекает границы
            t_min, t_max = 0.0, 1.0
            
            for axis_dim in range(3):
                if abs(direction[axis_dim]) > 1e-12:
                    # Пересечение с левой границей (axis = 0)
                    t1 = -start[axis_dim] / direction[axis_dim]
                    # Пересечение с правой границей (axis = field_size)
                    t2 = (self.field_size - start[axis_dim]) / direction[axis_dim]
                    
                    # Обновляем диапазон
                    t_min = max(t_min, min(t1, t2))
                    t_max = min(t_max, max(t1, t2))
            
            # Если трубка хоть частично в поле
            if t_max > t_min:
                # Длина внутри поля
                actual_length = np.linalg.norm(direction) * (t_max - t_min)
                
                # Объём полой трубки
                tube_volume = np.pi * (self.outer_radius**2 - self.inner_radius**2) * actual_length
                total_tube_volume += tube_volume
        
        # ========== ОБЪЁМ АЭРОГЕЛЯ ==========
        total_particle_volume = 0.0
        
        if self.enable_particles and self.particle_to_total_ratio > 0:
            # ✅ ИСПРАВЛЕНО: Используем particle_axes, а не particles!
            for i, axis in enumerate(self.particle_axes):
                start, end = axis  # axis это кортеж (p1, p2)
                
                # Радиус частицы из параметров
                type_idx = self.particle_types[i]
                radius = self.particle_params[type_idx]['radius']
                
                direction = end - start
                
                # Аналогично трубкам
                t_min, t_max = 0.0, 1.0
                
                for axis_dim in range(3):
                    if abs(direction[axis_dim]) > 1e-12:
                        t1 = -start[axis_dim] / direction[axis_dim]
                        t2 = (self.field_size - start[axis_dim]) / direction[axis_dim]
                        t_min = max(t_min, min(t1, t2))
                        t_max = min(t_max, max(t1, t2))
                
                if t_max > t_min:
                    actual_length = np.linalg.norm(direction) * (t_max - t_min)
                    # Частица - сплошной цилиндр
                    particle_volume = np.pi * radius**2 * actual_length
                    total_particle_volume += particle_volume
        
        return {
            'tube_volume_nm3': total_tube_volume,
            'particle_volume_nm3': total_particle_volume,
            'tube_fraction': total_tube_volume / field_volume,
            'particle_fraction': total_particle_volume / field_volume,
            'total_fraction': (total_tube_volume + total_particle_volume) / field_volume
        }


    
    def calculate_conductivity(self):
        """Расчёт электрической проводимости композита"""
        if not self.enable_conductivity:
            return
        
        if len(self.tubes) == 0:
            print("⚠️ Нет трубок для расчёта проводимости")
            return
        
        print("\n" + "="*60)
        print("РАСЧЁТ ЭЛЕКТРОПРОВОДНОСТИ")
        print("="*60)
        
        start_time = time.time()
        
        self.conductivity_calculator = ConductivityCalculator(self.field_size, self.voxel_size)
        self.conductivity_calculator.simulator = self
        
        # ИСПРАВЛЕНО: Передаём радиус трубки для расчёта истинного зазора
        self.conductivity_calculator.tube_radius = self.outer_radius
        
        print(f"\n🔄 Вокселизация {len(self.tubes)} нанотрубок...")
        total_voxels = 0
        for i, axis in enumerate(self.tube_axes):
            voxels = self.conductivity_calculator.voxelize_nanotube(axis, self.outer_radius, i)
            total_voxels += voxels
            if (i + 1) % 100 == 0:
                print(f"   ✓ Обработано {i+1}/{len(self.tube_axes)} нанотрубок")
        
        print(f"   ✓ Вокселизировано {total_voxels:,} вокселей от трубок")
        
        if self.enable_particles and len(self.particles) > 0:
            print(f"\n🔄 Вокселизация {len(self.particles)} частиц аэрогеля...")
            particle_voxels = 0
            for i, axis in enumerate(self.particle_axes):
                particle_type = self.particle_types[i]
                params = self.particle_params[particle_type]
                voxels = self.conductivity_calculator.voxelize_particle(
                    axis, params['radius'], params['length'], particle_type
                )
                particle_voxels += voxels
                if (i + 1) % 100 == 0:
                    print(f"   ✓ Обработано {i+1}/{len(self.particle_axes)} частиц")
            
            print(f"   ✓ Вокселизировано {particle_voxels:,} вокселей от частиц")
        
        print(f"\n✓ Всего заполнено вокселей: {len(self.conductivity_calculator.voxels):,}")
        
        self.conductivity_calculator.calculate_local_conductivities()
        
        percolates, cluster_map, percolating_clusters = self.conductivity_calculator.find_percolating_clusters()
        
        self.last_cluster_map = cluster_map
        self.last_percolating_clusters = percolating_clusters
        
        cluster_sizes = defaultdict(int)
        for cid in cluster_map.values():
            cluster_sizes[cid] += 1
        
        num_clusters = len(set(cluster_sizes.keys()))
        largest_cluster_size = max(cluster_sizes.values()) if cluster_sizes else 0
        percolating_cluster_size = max([cluster_sizes[cid] for cid in percolating_clusters]) if percolates else 0
        
        sigma_eff, results = self.conductivity_calculator.calculate_effective_conductivity_simple(
            percolates, percolating_clusters, cluster_map
        )
        
        self.conductivity_results = results
        self.conductivity_results['percolates'] = percolates
        self.conductivity_results['sigma_effective'] = sigma_eff
        self.conductivity_results['num_clusters'] = num_clusters
        self.conductivity_results['largest_cluster_size'] = largest_cluster_size
        self.conductivity_results['percolating_cluster_size'] = percolating_cluster_size
        self.conductivity_results['num_percolating'] = len(percolating_clusters)
        
        elapsed = time.time() - start_time
        
        # НОВОЕ: Всегда показываем объёмные доли
        volumes = self.calculate_actual_volumes()
        
        print("\n" + "="*60)
        print("РАСЧЁТ ЗАВЕРШЁН")
        print("="*60)
        print(f"⏱️  Время расчёта: {elapsed:.1f} секунд")
        
        print(f"\n📊 ОБЪЁМНЫЕ ДОЛИ:")
        print(f"   • УНТ (с обрезкой):   {volumes['tube_fraction']*100:.3f}%")
        if volumes['particle_fraction'] > 0:
            print(f"   • Аэрогель:           {volumes['particle_fraction']*100:.3f}%")
            print(f"   • Всего:              {volumes['total_fraction']*100:.3f}%")
            
            # Проверка на переупаковку
            if volumes['particle_fraction'] > 0.5:
                print(f"   ⚠️  ВНИМАНИЕ: Аэрогель занимает >{volumes['particle_fraction']*100:.0f}% объёма!")
                print(f"       Это не аэрогель, а плотный материал!")
                print(f"       Рекомендуется: φ_аэрогель < 10%")
        
        if percolates:
            print(f"\n✅ СИСТЕМА ПРОВОДЯЩАЯ")
            print(f"   Эффективная проводимость: {sigma_eff:.2e} См/м")
            print(f"   Количество перколяционных кластеров: {len(percolating_clusters)}")
            print(f"   Размер перколяционного кластера: {percolating_cluster_size:,} вокселей")
        else:
            print(f"\n❌ СИСТЕМА НЕ ПРОВОДЯЩАЯ")
            print(f"   Всего кластеров: {num_clusters}")
            print(f"   Размер наибольшего кластера: {largest_cluster_size:,} вокселей")
    
    def print_percolation_analysis(self):
        """Детальный анализ перколяции"""
        if not hasattr(self, 'conductivity_results') or self.conductivity_results is None:
            return
        
        results = self.conductivity_results
        
        print("\n" + "="*70)
        print("🔬 ДЕТАЛЬНЫЙ АНАЛИЗ ПЕРКОЛЯЦИИ")
        print("="*70)
        print(f"   • Объёмная доля ОУНТ: {results.get('volume_fraction', 0)*100:.3f}%")
        print(f"   • Доля проводящих вокселей: {results.get('percolating_fraction', 0)*100:.2f}%")
        print(f"   • Количество кластеров: {results.get('num_clusters', 0)}")
        print(f"   • Извилистость пути: {results.get('tortuosity', 0):.2f}")
        print(f"   • Контактный фактор: {results.get('contact_resistance_factor', 0):.3f}")
        
        if results.get('percolates', False):
            print(f"\n✅ РЕЗУЛЬТАТ: ПЕРКОЛЯЦИЯ ОБНАРУЖЕНА")
            print(f"   • Метод расчёта: {results.get('method', 'Неизвестно')}")
            print(f"   • Эффективная проводимость: {results.get('sigma_effective', 0):.2e} См/м")
        else:
            print(f"\n❌ РЕЗУЛЬТАТ: ПЕРКОЛЯЦИИ НЕТ")
        
        print("="*70)
    
    def visualize(self):
        """Визуализация композита"""
        print("\n" + "="*60)
        print("ВИЗУАЛИЗАЦИЯ")
        print("="*60)
        
        self.plotter = pv.Plotter()
        self.plotter.set_background(self.background_color)
        
        print("🎨 Добавление нанотрубок...")
        for tube in self.tubes:
            self.plotter.add_mesh(tube, color=self.tube_color, opacity=self.tube_opacity)
        
        if self.enable_particles and self.particles:
            print(f"🎨 Добавление {len(self.particles)} частиц аэрогеля...")
            for i, particle_mesh in enumerate(self.particles):
                type_idx = self.particle_types[i]
                params = self.particle_params[type_idx]
                self.plotter.add_mesh(
                    particle_mesh,
                    color=params['color'],
                    opacity=params['opacity']
                )
        
        if self.show_conductive_paths:
            if (self.conductivity_results and 
                self.conductivity_results.get('percolates', False) and 
                self.last_cluster_map and 
                self.last_percolating_clusters and
                self.conductivity_calculator):
                
                print("🎨 Генерация подсветки проводящих путей...")
                
                if self.cluster_highlight_mode == 'tubes':
                    print("   📍 Режим: подсветка проводящих трубок (красным)")
                    
                    percolating_tube_ids = set()
                    calc = self.conductivity_calculator
                    
                    for idx, cluster_id in self.last_cluster_map.items():
                        if cluster_id in self.last_percolating_clusters:
                            voxel_data = calc.voxels.get(idx)
                            if voxel_data and voxel_data['type'] == 1 and 'tube_ids' in voxel_data:
                                percolating_tube_ids.update(voxel_data['tube_ids'])
                    
                    for tube_id in percolating_tube_ids:
                        if tube_id < len(self.tubes):
                            self.plotter.add_mesh(
                                self.tubes[tube_id],
                                color='red',
                                opacity=0.8,
                                label='Percolating Tubes'
                            )
                    
                    print(f"   ✓ Подсвечено {len(percolating_tube_ids)} проводящих трубок")
                
                else:
                    print("   📍 Режим: подсветка проводящих вокселей (жёлтым)")
                    
                    points = []
                    calc = self.conductivity_calculator
                    
                    for idx, cluster_id in self.last_cluster_map.items():
                        if cluster_id in self.last_percolating_clusters:
                            if calc.voxels[idx]['type'] == 1:
                                points.append(calc.idx_to_coord(*idx))
                    
                    if points:
                        cloud = pv.PolyData(np.array(points))
                        cube = pv.Cube(
                            x_length=self.voxel_size*0.9,
                            y_length=self.voxel_size*0.9,
                            z_length=self.voxel_size*0.9
                        )
                        voxels = cloud.glyph(geom=cube, scale=False, orient=False)
                        self.plotter.add_mesh(
                            voxels,
                            color='yellow',
                            opacity=0.3,
                            label='Conductive Path'
                        )
                        
                        print(f"   ✓ Подсвечено {len(points)} проводящих вокселей")
            else:
                print("   ⚠️ Подсветка кластеров недоступна (нет перколяции или не выполнен расчёт)")
        
        bounds = [0, self.field_size, 0, self.field_size, 0, self.field_size]
        cube_edges = pv.Box(bounds=bounds).extract_feature_edges()
        self.plotter.add_mesh(cube_edges, color='black', line_width=2)
        
        print("✓ Сцена готова, запуск визуализации...")
        self.plotter.show()


# ==========================================
# GUI КЛАСС
# ==========================================
class LauncherGUI:
    def __init__(self, simulator):
        self.sim = simulator
        self.root = tk.Tk()
        self.root.title("🔬 Симулятор композитных материалов")
        self.root.geometry("650x750")
        
        style = ttk.Style()
        style.theme_use('clam')
        
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.vars = {}
        
        # ВКЛАДКА 1: ОБЩИЕ
        tab_general = ttk.Frame(self.notebook)
        self.notebook.add(tab_general, text='⚙️ Общие')
        
        self.create_group(tab_general, "Параметры поля", [
            ('field_size', 'Размер поля (нм):', 'float'),
            ('random_seed', 'Сид (пусто = случайный):', 'int_none'),
            ('max_attempts_custom', 'Макс. попыток (пусто = авто):', 'int_none'),
        ])
        
        self.create_group(tab_general, "Обработка геометрии", [
            ('clip_protruding_parts', 'Обрезать выступающие части', 'bool')
        ])
        
        # ВКЛАДКА 2: НАНОТРУБКИ
        tab_tubes = ttk.Frame(self.notebook)
        self.notebook.add(tab_tubes, text='⚛️ Нанотрубки')
        
        self.create_group(tab_tubes, "Геометрия", [
            ('num_tubes', 'Количество трубок:', 'int'),
            ('tube_length', 'Длина трубки (нм):', 'float'),
            ('outer_radius', 'Внешний радиус (нм):', 'float'),
            ('inner_radius', 'Внутренний радиус (нм):', 'float'),
            ('min_gap', 'Мин. зазор между трубками (нм):', 'float')
        ])
        
        # Ориентация
        frame_orient = ttk.LabelFrame(tab_tubes, text="Ориентация", padding="10")
        frame_orient.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(frame_orient, text="Режим:").grid(row=0, column=0, sticky='w', pady=2)
        orient_modes = ['random_angles', 'aligned']
        var_orient = tk.StringVar(value=self.sim.orientation_mode)
        cb = ttk.Combobox(frame_orient, textvariable=var_orient, values=orient_modes, state="readonly")
        cb.grid(row=0, column=1, sticky='ew', padx=5)
        self.vars['orientation_mode'] = {'var': var_orient, 'type': 'str'}
        
        # ИСПРАВЛЕНО: Джиттер должен показывать 4.0 по умолчанию
        ttk.Label(frame_orient, text='Джиттер (для aligned) °:').grid(row=1, column=0, sticky='w', pady=2)
        var_jitter = tk.StringVar(value='4.0')  # ФИКС: было пусто, теперь 4.0
        widget_jitter = ttk.Entry(frame_orient, textvariable=var_jitter)
        widget_jitter.grid(row=1, column=1, sticky='ew', padx=5, pady=2)
        self.vars['aligned_jitter_deg'] = {'var': var_jitter, 'type': 'float'}
        
        self.add_field_to_frame(frame_orient, 2, 'aligned_dir_str', 'Вектор aligned (x,y,z):', 'str', 
                                default=f"{self.sim.aligned_dir[0]},{self.sim.aligned_dir[1]},{self.sim.aligned_dir[2]}")
        
        frame_orient.columnconfigure(1, weight=1)
        
        # ВКЛАДКА 3: АЭРОГЕЛЬ
        tab_particles = ttk.Frame(self.notebook)
        self.notebook.add(tab_particles, text='💨 Аэрогель')
        
        self.create_group(tab_particles, "Настройки частиц", [
            ('enable_particles', 'Включить генерацию частиц', 'bool'),
            ('particle_to_total_ratio', 'Доля частиц (0.0-1.0):', 'float'),
            ('min_gap_particles', 'Мин. зазор частиц (нм):', 'float')
        ])
        
        # Параметры типов частиц
        frame_p_types = ttk.LabelFrame(tab_particles, text="Параметры типов частиц", padding="10")
        frame_p_types.pack(fill=tk.X, padx=5, pady=5)
        
        # Тип 1
        ttk.Label(frame_p_types, text="🔵 Тип 1:", font=('TkDefaultFont', 10, 'bold')).grid(row=0, column=0, columnspan=2, sticky='w', pady=5)
        self.add_field_to_frame(frame_p_types, 1, 'p1_radius', '  Радиус (нм):', 'float', default=self.sim.particle_params[0]['radius'])
        self.add_field_to_frame(frame_p_types, 2, 'p1_length', '  Длина (нм):', 'float', default=self.sim.particle_params[0]['length'])
        self.add_field_to_frame(frame_p_types, 3, 'p1_ratio', '  Доля в смеси:', 'float', default=self.sim.particle_type_distribution[0])
        self.add_field_to_frame(frame_p_types, 4, 'p1_color', '  Цвет:', 'str', default=self.sim.particle_params[0]['color'])
        self.add_field_to_frame(frame_p_types, 5, 'p1_opacity', '  Прозрачность:', 'float', default=self.sim.particle_params[0]['opacity'])
        
        # Тип 2
        ttk.Label(frame_p_types, text="🔴 Тип 2:", font=('TkDefaultFont', 10, 'bold')).grid(row=6, column=0, columnspan=2, sticky='w', pady=5)
        self.add_field_to_frame(frame_p_types, 7, 'p2_radius', '  Радиус (нм):', 'float', default=self.sim.particle_params[1]['radius'])
        self.add_field_to_frame(frame_p_types, 8, 'p2_length', '  Длина (нм):', 'float', default=self.sim.particle_params[1]['length'])
        self.add_field_to_frame(frame_p_types, 9, 'p2_color', '  Цвет:', 'str', default=self.sim.particle_params[1]['color'])
        self.add_field_to_frame(frame_p_types, 10, 'p2_opacity', '  Прозрачность:', 'float', default=self.sim.particle_params[1]['opacity'])
        
        frame_p_types.columnconfigure(1, weight=1)
        
        # ВКЛАДКА 4: ПРОВОДИМОСТЬ
        tab_cond = ttk.Frame(self.notebook)
        self.notebook.add(tab_cond, text='⚡ Проводимость')
        
        self.create_group(tab_cond, "Расчёт", [
            ('enable_conductivity', 'Включить расчёт проводимости', 'bool'),
            ('voxel_size', 'Размер воксела (нм):', 'float'),
            ('show_percolation_analysis', 'Детальный анализ перколяции', 'bool')
        ])
        
        # ВКЛАДКА 5: ВИЗУАЛИЗАЦИЯ
        tab_viz = ttk.Frame(self.notebook)
        self.notebook.add(tab_viz, text='🎨 Визуализация')
        
        # Подсветка кластеров
        frame_clusters = ttk.LabelFrame(tab_viz, text="Подсветка проводящих кластеров", padding="10")
        frame_clusters.pack(fill=tk.X, padx=5, pady=5)
        
        # ИСПРАВЛЕНО: Подсветка должна быть включена по умолчанию
        ttk.Label(frame_clusters, text='Включить подсветку').grid(row=0, column=0, sticky='w', pady=2)
        var_paths = tk.BooleanVar(value=True)  # ФИКС: было False через default, теперь True явно
        widget_paths = ttk.Checkbutton(frame_clusters, variable=var_paths)
        widget_paths.grid(row=0, column=1, sticky='w', pady=2)
        self.vars['show_conductive_paths'] = {'var': var_paths, 'type': 'bool'}
        
        ttk.Label(frame_clusters, text="Режим подсветки:").grid(row=1, column=0, sticky='w', pady=2)
        cluster_modes = ['tubes', 'voxels']
        var_cluster_mode = tk.StringVar(value=self.sim.cluster_highlight_mode)
        cb_cluster = ttk.Combobox(frame_clusters, textvariable=var_cluster_mode, values=cluster_modes, state="readonly", width=15)
        cb_cluster.grid(row=1, column=1, sticky='w', padx=5)
        self.vars['cluster_highlight_mode'] = {'var': var_cluster_mode, 'type': 'str'}
        
        ttk.Label(frame_clusters, text="  • tubes = подсветка трубок красным").grid(row=2, column=0, columnspan=2, sticky='w', padx=20, pady=2)
        ttk.Label(frame_clusters, text="  • voxels = подсветка вокселей жёлтым").grid(row=3, column=0, columnspan=2, sticky='w', padx=20, pady=2)
        
        frame_clusters.columnconfigure(1, weight=1)
        
        self.create_group(tab_viz, "Цвета", [
            ('tube_color', 'Цвет трубок:', 'str'),
            ('tube_opacity', 'Прозрачность трубок (0.0-1.0):', 'float'),
            ('background_color', 'Цвет фона:', 'str')
        ])
        
        # Кнопка запуска
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        btn_run = ttk.Button(btn_frame, text="✅ ЗАПУСТИТЬ СИМУЛЯЦИЮ", command=self.on_run)
        btn_run.pack(fill=tk.X, ipady=10)
    
    def create_group(self, parent, title, fields):
        """Создание группы полей"""
        frame = ttk.LabelFrame(parent, text=title, padding="10")
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        for idx, item in enumerate(fields):
            attr_name, label_text, data_type = item
            default_val = getattr(self.sim, attr_name)
            self.add_field_to_frame(frame, idx, attr_name, label_text, data_type, default_val)
        
        frame.columnconfigure(1, weight=1)
    
    def add_field_to_frame(self, frame, row, attr_name, label_text, data_type, default=None):
        """Добавление поля в форму"""
        ttk.Label(frame, text=label_text).grid(row=row, column=0, sticky='w', pady=2)
        
        if data_type == 'bool':
            var = tk.BooleanVar(value=default if default is not None else False)
            widget = ttk.Checkbutton(frame, variable=var)
            widget.grid(row=row, column=1, sticky='w', pady=2)
        else:
            val_str = str(default) if default is not None else ""
            var = tk.StringVar(value=val_str)
            widget = ttk.Entry(frame, textvariable=var)
            widget.grid(row=row, column=1, sticky='ew', padx=5, pady=2)
        
        self.vars[attr_name] = {'var': var, 'type': data_type}
    
    def on_run(self):
        """Обработчик запуска симуляции"""
        try:
            for attr, data in self.vars.items():
                val = data['var'].get()
                dtype = data['type']
                
                if attr == 'aligned_dir_str':
                    try:
                        parts = [float(x.strip()) for x in val.split(',')]
                        if len(parts) == 3:
                            self.sim.aligned_dir = tuple(parts)
                    except:
                        print("⚠️ Ошибка парсинга вектора aligned_dir")
                    continue
                
                if attr.startswith('p1_') or attr.startswith('p2_'):
                    continue
                
                if dtype == 'int':
                    setattr(self.sim, attr, int(val))
                elif dtype == 'float':
                    setattr(self.sim, attr, float(val))
                elif dtype == 'bool':
                    setattr(self.sim, attr, data['var'].get())
                elif dtype == 'int_none':
                    if val.strip() == "":
                        setattr(self.sim, attr, None)
                    else:
                        setattr(self.sim, attr, int(val))
                else:
                    setattr(self.sim, attr, val)
            
            # Обновление параметров аэрогеля
            try:
                p1_rad = float(self.vars['p1_radius']['var'].get())
                p1_len = float(self.vars['p1_length']['var'].get())
                p1_ratio = float(self.vars['p1_ratio']['var'].get())
                p1_color = self.vars['p1_color']['var'].get()
                p1_opacity = float(self.vars['p1_opacity']['var'].get())
                
                self.sim.particle_params[0] = {
                    'radius': p1_rad,
                    'length': p1_len,
                    'color': p1_color,
                    'opacity': p1_opacity
                }
                
                p2_rad = float(self.vars['p2_radius']['var'].get())
                p2_len = float(self.vars['p2_length']['var'].get())
                p2_color = self.vars['p2_color']['var'].get()
                p2_opacity = float(self.vars['p2_opacity']['var'].get())
                
                self.sim.particle_params[1] = {
                    'radius': p2_rad,
                    'length': p2_len,
                    'color': p2_color,
                    'opacity': p2_opacity
                }
                
                self.sim.particle_type_distribution = [p1_ratio, 1.0 - p1_ratio]
            
            except Exception as e:
                print(f"⚠️ Ошибка при обновлении параметров частиц: {e}")
            
            self.root.destroy()
            
            print("\n" + "="*60)
            print("🚀 ЗАПУСК СИМУЛЯЦИИ")
            print("="*60)
            
            self.sim.generate_all_objects()
            
            if self.sim.enable_conductivity:
                self.sim.calculate_conductivity()
                
                if self.sim.show_percolation_analysis:
                    if self.sim.conductivity_results:
                        self.sim.print_percolation_analysis()
            
            print("\n🎨 Запуск визуализации...")
            self.sim.visualize()
            
        except ValueError as e:
            messagebox.showerror("Ошибка ввода", f"Проверьте правильность введённых данных:\n\n{e}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка:\n\n{e}")


# ==========================================
# ЗАПУСК
# ==========================================
if __name__ == "__main__":
    print("="*60)
    print("🔬 UNIFIED NANOTUBE SIMULATOR")
    print("="*60)
    print("")
    print("Возможности:")
    print("  ✅ Генерация углеродных нанотрубок")
    print("  ✅ Генерация частиц аэрогеля")
    print("  ✅ CPU-ускорение (Numba JIT)")
    print("  ✅ Расчёты Кирхгоффа (SciPy)")
    print("  ✅ Анализ перколяции")
    print("  ✅ Визуализация проводящих путей")
    print("")
    print("="*60)
    
    simulator = EnhancedNanotubeSimulator()
    
    app = LauncherGUI(simulator)
    app.root.mainloop()
