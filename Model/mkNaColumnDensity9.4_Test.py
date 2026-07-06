# -*- coding: utf-8 -*-
"""
==============================================================================
プロジェクト: 水星ナトリウム外気圏 3次元モンテカルロシミュレーション
              (Mercury Sodium Exosphere 3D Monte-Carlo Simulation)

概要:
    水星表面からのナトリウム放出と、外気圏における粒子の運動を計算する。

更新内容:
    - Budget Analysis (生成・消滅の内訳集計) 機能を追加
      TAAごとの生成(PSD/TD/SWS/MMV)と消滅(Stuck/Ionized/Escaped)をCSV出力。
    - [New] Multi-Bin Binding Energy Model (マルチビン束縛エネルギーモデル) の導入
      表面のナトリウムを複数の束縛エネルギー(U)のビンとして管理し、それぞれで
      独立した放出率(PSD/TD)を計算。
    - [New] 物理的な滞在時間(tau_TD)に基づく自然なマルチバウンド(即時脱離)モデルへ移行。

作成者: Koki Masaki (Rikkyo Univ.)
日付: 2026/05/29 (Updated: Multi-Bin Model Integration)
==============================================================================
"""

import numpy as np
import sys
from numba import njit
import os
from multiprocessing import Pool, cpu_count
import time
from typing import Dict, Tuple, List, Optional, Any
import csv 
from numba import njit, prange
import os

# 物理コア数を取得 (ハイパースレッディングの論理コアではなく物理コアを使うのが理想)
import multiprocessing
logical_cores = multiprocessing.cpu_count()
# 安全のために1コアだけOSに空けておく
use_cores = str(max(1, logical_cores - 1))

# Numbaのスレッド層を明示的に指定 (omp または tbb が高速)
os.environ['NUMBA_THREADING_LAYER'] = 'omp'  # エラーが出る場合は 'workqueue' か 'tbb' に変更

# Numbaが使用するスレッド数を強制
os.environ['NUMBA_NUM_THREADS'] = use_cores

# NumPy (OpenBLAS/MKL) の内部並列をオフにする (Numbaとの競合・デッドロックを防ぐ)
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
from numba import njit, prange
import numba


# ==============================================================================
# 0. シミュレーション設定・物理定数
# ==============================================================================

# 物理定数
PHYSICAL_CONSTANTS = {
    'PI': np.pi,
    'AU': 1.496e11,  # 1天文単位 [m]
    'MASS_NA': 3.8175e-26,  # ナトリウム原子質量 [kg]
    'K_BOLTZMANN': 1.380649e-23,  # ボルツマン定数 [J/K]
    'GM_MERCURY': 2.2032e13,  # 水星重力定数 (G * M_Mercury) [m^3/s^2]
    'RM': 2.440e6,  # 水星半径 [m]
    'C': 299792458.0,  # 光速 [m/s]
    'H': 6.62607015e-34,  # プランク定数 [J s]
    'E_CHARGE': 1.602e-19,  # 素電荷 [C]
    'ME': 9.109e-31,  # 電子質量 [kg]
    'EPSILON_0': 8.854e-12,  # 真空の誘電率 [F/m]
    'G': 6.6743e-11,  # 万有引力定数 [m^3 kg^-1 s^-2]
    'MASS_SUN': 1.989e30,  # 太陽質量 [kg]
    'EV_TO_JOULE': 1.602e-19,  # eV -> Joule 変換係数
    'ROTATION_PERIOD': 58.6462 * 86400,  # 自転周期 [s]
    'ORBITAL_PERIOD': 87.969 * 86400,  # 公転周期 [s]

    'MERCURY_SEMI_MAJOR_AXIS_AU': 0.387098,
    'MERCURY_ECCENTRICITY': 0.205630,
}


# 統合シミュレーション設定
SIMULATION_SETTINGS = {
    # --- 時間ステップ設定 ---
    'DT_MOVE': 100.0,  # 粒子の位置更新ステップ [s]
    'DT_RATE_UPDATE': 100.0,  # 表面放出率の再計算ステップ [s]
    'DT_INTEGRATION': 100.0,  # 粒子軌跡計算の内部積分ステップ

    # --- 温度モデル設定 (Leblanc et al.) ---
    'TEMP_BASE': 100.0,
    'TEMP_AMP': 600.0,
    'TEMP_NIGHT': 100.0,

    # --- グリッド・領域設定 ---
    'N_LON': 72,
    'N_LAT': 36,
    'GRID_RESOLUTION': 101,
    'GRID_MAX_RM': 5.0,
    'GRID_RADIUS_RM': 6.0,

    # --- 物理フラグ ---
    'BETA': 1.0,
    #'T1AU': 134000.0,
    # Fulle et al. (2007) の極小期モデルを再現する場合
    'T1AU': 190000.0,
    'USE_SOLAR_GRAVITY': True,
    'USE_CORIOLIS_FORCES': True,

    # --- 計算モード ---
    'USE_EQUILIBRIUM_MODE': False,
    'USE_AREA_WEIGHTED_FLUX': False,
    'USE_SUBGRID_SMOOTHING': False,

    # --- [拡散モデル設定] ---
    'USE_STD_DIFFUSION': True,  # 標準拡散 (距離に応じてフル変動)
    'USE_CLAMPED_DIFFUSION': False,  # 近日点ピークカット拡散 (現在は基本使用していない)

    # 吸着モデルの選択 ('empirical' または 'baule')
    'STICKING_MODEL': 'empirical', #bauleは水星表面に適応できるか微妙なところ。そして理論もまだ不十分であり、改善の余地しかない。
    
    'USE_PROBABILISTIC_STICKING': True, # 確率的吸着か閾値(300K)吸着かの切り替え

    # ==========================================================================
    # マルチビン束縛エネルギー設定 (Multi-Bin Binding Energy)
    # ==========================================================================
    # 束縛エネルギーモデルの選択: 'fixed', 'dynamic', 'gaussian_random' のいずれかを指定
    # fixedは固定値モデル、dynamicは吸着の活性化エネルギーを考慮したモデル、gaussianは最もスタンダードなモデル
    'U_MODEL_TYPE': 'gaussian_random', 

    # --- [A] 固定値モデル('fixed')用設定 ---
    'U_BINS_FIXED': np.array([1.85, 1.85]),
    #'Q_PSD_BINS_FIXED': np.array([2.7e-21 / (100 ** 2), 2.7e-21 / (100 ** 2)]),
    'Q_PSD_BINS_FIXED': np.array([2.0e-20 / (100 ** 2), 2.0e-20 / (100 ** 2)]),

    # --- [B/C] 共通のビン分割設定 ('dynamic', 'gaussian_random'用) ---
    'N_U_BINS': 10,              # ビンの分割数（任意に変更可能）
    'U_MIN': 1.4,                # 最小束縛エネルギー [eV]
    'U_MAX': 2.7,                # 最大束縛エネルギー [eV]

    # --- [C] ガウス分布ランダムモデル('gaussian_random')用設定 ---
    'U_GAUSSIAN_MU': 1.85,       # 分布の中心 [eV] 1.85def
    'U_GAUSSIAN_SIGMA': 0.25,    # 分布の広がり

    # --- [B] 動的モデル('dynamic')用設定 ---
    'V_U_MODE': 'uniform',       # 'uniform' または 'gaussian'

    # 拡散供給時の束縛エネルギー設定
    'DIFFUSION_U_MODE': 'gaussian',  # 'single'(単一ビン) または 'gaussian'(ガウス分布)
    'DIFFUSION_U_TARGET': 1.85,     # singleの場合：どの深さに供給するか [eV]
    'DIFFUSION_U_MU': 1.85,         # gaussianの場合：分布の中心 [eV]
    'DIFFUSION_U_SIGMA': 0.35,     # gaussianの場合：分布の広がり
    'DIFFUSION_U_MIN': 1.4,        # gaussianの場合：供給する最小U [eV]
    'DIFFUSION_U_MAX': 2.7,        # gaussianの場合：供給する最大U [eV]

    # --- 表面拡散によるトラップ機構 (Inward Diffusion) ---
    'USE_SURFACE_DIFFUSION_TRAPPING': False, # オンオフ機構 (Trueで深いビンへ移動)
    'E_DIFF_EV': 0.6,                       # 表面拡散の活性化エネルギー [eV]
    'A_DIFF_RATE': 4.0e3,                   # 移行の頻度因子 [1/s] (400Kで数時間スケール)
}

# サイトのインデックス定義
IDX_SHALLOW = 0
IDX_DEEP = 1
#N_BINS = len(SIMULATION_SETTINGS['U_BINS'])

# 定数計算用
KB_EV_CONST = 8.617e-5  # ボルツマン定数 [eV/K]

# ==============================================================================
# [A] Diffusion Model Parameters
# ==============================================================================
#DIFF_REF_FLUX = 5.0e6 * (100.0 ** 2)
DIFF_REF_FLUX = 2.0e7 * (100.0 ** 2)
DIFF_REF_TEMP = 700.0  # 基準温度 [K]
DIFF_E_A_EV = 0.5  # 活性化エネルギー [eV]
Target_Grain_Radius = 100.0e-6  # [m]

# 頻度因子 A (J0) の事前計算
DIFF_PRE_FACTOR = DIFF_REF_FLUX / np.exp(-DIFF_E_A_EV / (KB_EV_CONST * DIFF_REF_TEMP))

# ==============================================================================
# [B] Clamped (Peak-Cut) Diffusion Settings
# ==============================================================================
TAA_CLAMP_START = 70.0  # これより小さいTAA (0~70) は 70度の距離に固定
TAA_CLAMP_END = 290.0  # これより大きいTAA (290~360) は 290度の距離に固定


def calculate_au_at_taa(taa_deg: float) -> float:
    a = PHYSICAL_CONSTANTS['MERCURY_SEMI_MAJOR_AXIS_AU']
    e = PHYSICAL_CONSTANTS['MERCURY_ECCENTRICITY']
    rad = np.deg2rad(taa_deg)
    r = a * (1 - e ** 2) / (1 + e * np.cos(rad))
    return r


AU_AT_CUTOFF = calculate_au_at_taa(TAA_CLAMP_START)
FORCED_INJECTION_EVENTS = []


# ==============================================================================
# 1. 物理モデル・ヘルパー関数群
# ==============================================================================

@njit
def set_numba_seed(seed_value):
    """Numba内部の乱数生成器のシードを固定する"""
    np.random.seed(seed_value)

def setup_binding_energy_bins(settings: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    設定されたモデル(fixed, dynamic, gaussian_random)に応じて
    束縛エネルギーのビン(U_bins)と、サイトの確率重み(V_weights)を生成する。
    """
    mode = settings.get('U_MODEL_TYPE', 'fixed')

    # [A] 固定値モデル (従来の浅い/深い2ビン等)
    if mode == 'fixed':
        U_bins = settings.get('U_BINS_FIXED', np.array([1.95, 2.7]))
        V_weights = np.ones(len(U_bins))
        V_weights /= np.sum(V_weights)
        return U_bins, V_weights

    # [B/C] 共通のビン生成 (指定された最小〜最大をN分割)
    n_bins = settings.get('N_U_BINS', 5)
    U_bins = np.linspace(settings['U_MIN'], settings['U_MAX'], n_bins)

    # [C] 新規: ガウス分布ランダムモデル
    if mode == 'gaussian_random':
        mu = settings.get('U_GAUSSIAN_MU', 1.85)
        sigma = settings.get('U_GAUSSIAN_SIGMA', 0.35)
        
        # ガウス分布に基づく確率の重みを計算
        V_weights = np.exp(-0.5 * ((U_bins - mu) / sigma)**2)
        
        # 確率の規格化 (合計を1.0にする)
        weight_sum = np.sum(V_weights)
        if weight_sum > 1e-30:
            V_weights /= weight_sum
        else:
            V_weights = np.ones(n_bins) / n_bins # 安全対策

        return U_bins, V_weights

    # [B] 従来の動的エネルギーモデル
    elif mode == 'dynamic':
        v_mode = settings.get('V_U_MODE', 'uniform')
        if v_mode == 'gaussian':
            mu = settings.get('U_GAUSSIAN_MU', 1.85)
            sigma = settings.get('U_GAUSSIAN_SIGMA', 0.35)
            V_weights = np.exp(-0.5 * ((U_bins - mu) / sigma)**2)
        else:
            V_weights = np.ones_like(U_bins)
        
        V_weights /= np.max(V_weights) # dynamicは規格化ではなく最大値1で扱う
        return U_bins, V_weights

    # フェイルセーフ
    return settings.get('U_BINS_FIXED', np.array([1.95, 2.7])), np.array([1.0, 0.0])


def assign_sticking_bin(E_in_eV: float, temp_impact_K: float, U_bins: np.ndarray, V_weights: np.ndarray, settings: Dict) -> int:
    """
    粒子の吸着サイト(ビン)を決定する関数。
    """
    mode = settings.get('U_MODEL_TYPE', 'fixed')

    # [A] 固定値モデルの場合は常に浅いサイト(インデックス0)を返す
    if mode == 'fixed':
        return IDX_SHALLOW

    # [C] 新規: ガウス分布ランダムモデル
    # 入射エネルギーや温度に依存せず、表面のサイト分布(V_weights)のみで決定
    if mode == 'gaussian_random':
        return np.random.choice(len(U_bins), p=V_weights)

    # [B] 従来の動的エネルギーモデル (エネルギーと温度に依存)
    if mode == 'dynamic':
        kBT_eV = PHYSICAL_CONSTANTS['K_BOLTZMANN'] * temp_impact_K / PHYSICAL_CONSTANTS['EV_TO_JOULE']
        alpha = (settings['U_MAX'] - settings['U_MIN']) / 0.5
        U_target = settings['U_MIN'] + alpha * E_in_eV
        gamma = 0.5
        
        energy_factor = np.exp(- ((U_bins - U_target)**2) / (kBT_eV * gamma))
        probabilities = V_weights * energy_factor

        P_total = np.sum(probabilities)
        if P_total > 1e-100:
            probabilities /= P_total
        else:
            closest_idx = np.argmin(np.abs(U_bins - U_target))
            probabilities = np.zeros(len(U_bins))
            probabilities[closest_idx] = 1.0

        return np.random.choice(len(U_bins), p=probabilities)
        
    return IDX_SHALLOW

def get_diffusion_supply_distribution(total_supply: float, U_bins: np.ndarray, settings: Dict) -> np.ndarray:
    """
    内部拡散や強制注入によってもたらされた総供給量を、設定に応じて各ビンに分配する。
    """
    distribution = np.zeros(len(U_bins))
    if total_supply <= 0:
        return distribution

    mode = settings.get('DIFFUSION_U_MODE', 'single')

    if mode == 'single':
        # 指定された単一の深さ(DIFFUSION_U_TARGET)に一番近いビンを探す
        target_U = settings.get('DIFFUSION_U_TARGET', 2.7)
        closest_idx = np.argmin(np.abs(U_bins - target_U))
        distribution[closest_idx] = total_supply

    elif mode == 'gaussian':
        mu = settings.get('DIFFUSION_U_MU', 2.5)
        sigma = settings.get('DIFFUSION_U_SIGMA', 0.15)
        u_min = settings.get('DIFFUSION_U_MIN', 2.0)
        u_max = settings.get('DIFFUSION_U_MAX', 2.7)
        
        # 🌟 最小値〜最大値の範囲内(mask)にあるビンだけ計算する
        mask = (U_bins >= u_min) & (U_bins <= u_max)
        weights = np.zeros(len(U_bins))
        
        if np.any(mask):
            weights[mask] = np.exp(-0.5 * ((U_bins[mask] - mu) / sigma)**2)
        
        weights_sum = np.sum(weights)
        if weights_sum > 1e-30:
            weights /= weights_sum
            distribution = total_supply * weights
        else:
            # 範囲設定が厳しすぎて当てはまるビンが無い場合は一番深いビンに入れる
            distribution[-1] = total_supply
            
    else:
        distribution[-1] = total_supply

    return distribution

# ==============================================================================
# Python側 補助関数 (軌道計算・MMV生成用)
# ==============================================================================
def sample_speed_from_flux_distribution(mass_kg: float, temp_k: float) -> float:
    kT = PHYSICAL_CONSTANTS['K_BOLTZMANN'] * temp_k
    E = np.random.gamma(2.0, kT)
    return np.sqrt(2.0 * E / mass_kg)

def sample_lambertian_direction_local() -> np.ndarray:
    u1, u2 = np.random.random(2)
    phi = 2 * PHYSICAL_CONSTANTS['PI'] * u1
    cos_theta = np.sqrt(1 - u2)
    sin_theta = np.sqrt(u2)
    return np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])

def transform_local_to_world(local_vec: np.ndarray, normal_vector: np.ndarray) -> np.ndarray:
    local_z = normal_vector / np.linalg.norm(normal_vector)
    world_up = np.array([0., 0., 1.])
    if np.abs(np.dot(local_z, world_up)) > 0.99:
        world_up = np.array([0., 1., 0.])
    local_x = np.cross(world_up, local_z)
    local_x /= np.linalg.norm(local_x)
    local_y = np.cross(local_z, local_x)
    return local_vec[0] * local_x + local_vec[1] * local_y + local_vec[2] * local_z

def get_orbital_params_linear(time_sec: float, orbit_data: np.ndarray, t_perihelion_file: float) -> Tuple[float, float, float, float, float]:
    time_col_original = orbit_data[:, 2]
    t_lookup = np.clip(time_sec, time_col_original[0], time_col_original[-1])
    taa_deg = np.interp(t_lookup, time_col_original, orbit_data[:, 0])
    au = np.interp(t_lookup, time_col_original, orbit_data[:, 1])
    v_rad = np.interp(t_lookup, time_col_original, orbit_data[:, 3])
    v_tan = np.interp(t_lookup, time_col_original, orbit_data[:, 4])
    sub_lon_deg = np.interp(t_lookup, time_col_original, orbit_data[:, 5])
    return taa_deg, au, v_rad, v_tan, np.deg2rad(sub_lon_deg)

def lonlat_to_xyz(lon_rad: float, lat_rad: float, radius: float) -> np.ndarray:
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return np.array([x, y, z])    

def calculate_mmv_flux(AU: float) -> float:
    TOTAL_FLUX_AT_PERI = 5e23
    PERIHELION_AU = 0.307
    AREA = 4 * PHYSICAL_CONSTANTS['PI'] * (PHYSICAL_CONSTANTS['RM'] ** 2)
    avg_flux_peri = TOTAL_FLUX_AT_PERI / AREA
    C = avg_flux_peri * (PERIHELION_AU ** 1.9)
    return C * (AU ** (-1.9))

# Numba用 物理定数
MASS_NA_NB = 3.8175e-26
C_NB = 299792458.0
H_NB = 6.62607015e-34
KB_NB = 1.380649e-23
EV_TO_JOULE_NB = 1.602e-19
GM_MERCURY_NB = 2.2032e13 
RM_NB = 2.440e6
GM_SUN_NB = 6.6743e-11 * 1.989e30
SIGMA0_1_NB = (1.602e-19**2) / (4 * 9.109e-31 * 299792458.0 * 8.854e-12) * 0.320
SIGMA0_2_NB = (1.602e-19**2) / (4 * 9.109e-31 * 299792458.0 * 8.854e-12) * 0.641

#@njit(fastmath=True)
@njit()
def sample_lambertian_direction_numba():
    """Numba内で動作するローカル座標系でのランバート放出方向サンプリング"""
    u1 = np.random.random()
    u2 = np.random.random()
    phi = 2.0 * np.pi * u1
    cos_theta = np.sqrt(1.0 - u2)
    sin_theta = np.sqrt(u2)
    return sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta

#@njit(fastmath=True)
@njit()
def transform_local_to_world_numba(lx, ly, lz, nx, ny, nz):
    """Numba内でオブジェクト生成を排除したスカラー展開版ローカル→ワールド座標変換"""
    n_mag = np.sqrt(nx**2 + ny**2 + nz**2)
    zx = nx / n_mag
    zy = ny / n_mag
    zz = nz / n_mag
    
    wx, wy, wz = 0.0, 0.0, 1.0
    if np.abs(zx * wx + zy * wy + zz * wz) > 0.99:
        wx, wy, wz = 0.0, 1.0, 0.0
        
    xx = wy * zz - wz * zy
    xy = wz * zx - wx * zz
    xz = wx * zy - wy * zx
    x_mag = np.sqrt(xx**2 + xy**2 + xz**2)
    xx /= x_mag
    xy /= x_mag
    xz /= x_mag
    
    yx = zy * xz - zz * xy
    yy = zz * xx - zx * xz
    yz = zx * xy - zy * xx
    
    wx_out = lx * xx + ly * yx + lz * zx
    wy_out = lx * xy + ly * yy + lz * zy
    wz_out = lx * xz + ly * yz + lz * zz
    return wx_out, wy_out, wz_out

#@njit(fastmath=True)
@njit()
def sample_speed_from_flux_numba(temp_k):
    """マクスウェル・ボルツマン流束分布からの速度サンプリング(Numba版)"""
    u1 = np.random.random()
    u2 = np.random.random()
    E = -np.log(u1) - np.log(u2)
    E_joule = E * KB_NB * temp_k
    return np.sqrt(2.0 * E_joule / MASS_NA_NB)

#@njit(fastmath=True)
@njit()
def assign_sticking_bin_numba(E_in_eV, temp_impact_K, U_bins, V_weights, u_model_type, u_max, u_min):
    """Numba対応の吸着エネルギービン選択 (完全ロックフリー・スカラー最適化版)"""
    n_bins = U_bins.shape[0]
    
    if u_model_type == 0: # 'fixed'
        return 0

    elif u_model_type == 1: # 'gaussian_random'
        r = np.random.random()
        cum_p = 0.0
        for b in range(n_bins):
            cum_p += V_weights[b]
            if r <= cum_p:
                return b
        return n_bins - 1

    else: # 'dynamic'
        kBT_eV = KB_NB * temp_impact_K / EV_TO_JOULE_NB
        alpha = (u_max - u_min) / 0.5
        U_target = u_min + alpha * E_in_eV
        gamma_val = 0.5
        
        # 代わりにスカラー変数だけで「合計値の計算」と「確率の判定」を行う
        p_sum = 0.0
        for b in range(n_bins):
            p_sum += V_weights[b] * np.exp(- ((U_bins[b] - U_target)**2) / (kBT_eV * gamma_val))
            
        if p_sum > 1e-100:
            r = np.random.random() * p_sum
            cum_p = 0.0
            for b in range(n_bins):
                cum_p += V_weights[b] * np.exp(- ((U_bins[b] - U_target)**2) / (kBT_eV * gamma_val))
                if r <= cum_p:
                    return b
        else:
            # 最も近いビンを探索
            min_diff = 999.0
            closest_idx = 0
            for b in range(n_bins):
                diff = np.abs(U_bins[b] - U_target)
                if diff < min_diff:
                    min_diff = diff
                    closest_idx = b
            return closest_idx
            
    return 0

#@njit(fastmath=True)
@njit()
def calculate_acceleration_numba(px, py, pz, vx, vy, vz, V_rad, V_tan, r0, spec_wl, spec_gamma, JL_const, AU):
    r_sq = px**2 + py**2 + pz**2
    if r_sq > 0:
        r_cube_inv = 1.0 / (r_sq * np.sqrt(r_sq))
        accel_g_x = -GM_MERCURY_NB * px * r_cube_inv
        accel_g_y = -GM_MERCURY_NB * py * r_cube_inv
        accel_g_z = -GM_MERCURY_NB * pz * r_cube_inv
    else:
        accel_g_x, accel_g_y, accel_g_z = 0.0, 0.0, 0.0

    if r0 > 0:
        omega_val = V_tan / r0
        omega_sq = omega_val ** 2
        accel_cen_x = omega_sq * (px - r0)
        accel_cen_y = omega_sq * py
        accel_cen_z = 0.0
        
        two_omega = 2.0 * omega_val
        accel_cor_x = two_omega * vy
        accel_cor_y = -two_omega * vx
        accel_cor_z = 0.0
    else:
        accel_cen_x, accel_cen_y, accel_cen_z = 0.0, 0.0, 0.0
        accel_cor_x, accel_cor_y, accel_cor_z = 0.0, 0.0, 0.0

    velocity_for_doppler = vx - V_rad
    w_na_d2 = 589.1582e-9 * (1.0 + velocity_for_doppler / C_NB)
    w_na_d1 = 589.7558e-9 * (1.0 + velocity_for_doppler / C_NB)
    
    b = 0.0
    if (spec_wl[0] * 1e-9 <= w_na_d2 < spec_wl[-1] * 1e-9) and (spec_wl[0] * 1e-9 <= w_na_d1 < spec_wl[-1] * 1e-9):
        gamma2 = np.interp(w_na_d2 * 1e9, spec_wl, spec_gamma)
        gamma1 = np.interp(w_na_d1 * 1e9, spec_wl, spec_gamma)
        F_at_Merc = (JL_const * 1e13) / (AU ** 2)

        term_d1 = (H_NB / w_na_d1) * SIGMA0_1_NB * (F_at_Merc * gamma1 * w_na_d1 ** 2 / C_NB)
        term_d2 = (H_NB / w_na_d2) * SIGMA0_2_NB * (F_at_Merc * gamma2 * w_na_d2 ** 2 / C_NB)
        b = (term_d1 + term_d2) / MASS_NA_NB

    if px < 0 and np.sqrt(py ** 2 + pz ** 2) < RM_NB:
        b = 0.0
    accel_srp_x, accel_srp_y, accel_srp_z = -b, 0.0, 0.0 

    # =========================================================
    # ▼ここから追加：太陽重力の計算▼
    # =========================================================
    accel_sun_x, accel_sun_y, accel_sun_z = 0.0, 0.0, 0.0
    if r0 > 0.0:
        # 太陽は水星中心から +x 方向（距離 r0）にある
        r_ps_x = r0 - px
        r_ps_y = -py
        r_ps_z = -pz
        r_ps_sq = r_ps_x**2 + r_ps_y**2 + r_ps_z**2
        
        if r_ps_sq > 0.0:
            r_ps_cube_inv = 1.0 / (r_ps_sq * np.sqrt(r_ps_sq))
            accel_sun_x = GM_SUN_NB * r_ps_x * r_ps_cube_inv
            accel_sun_y = GM_SUN_NB * r_ps_y * r_ps_cube_inv
            accel_sun_z = GM_SUN_NB * r_ps_z * r_ps_cube_inv
    # =========================================================

    # 全加速度の合算（accel_sun を追加）
    ax = accel_g_x + accel_cen_x + accel_cor_x + accel_srp_x + accel_sun_x
    ay = accel_g_y + accel_cen_y + accel_cor_y + accel_srp_y + accel_sun_y
    az = accel_g_z + accel_cen_z + accel_cor_z + accel_srp_z + accel_sun_z

    return ax, ay, az

#@njit(fastmath=True)
@njit()
def calculate_surface_temperature_leblanc_nb(lon_rad, lat_rad, AU, subsolar_lon_rad, t_base, t_amp, t_night):
    scaling = np.sqrt(0.306 / AU)
    cos_theta = np.cos(lat_rad) * np.cos(lon_rad - subsolar_lon_rad)
    if cos_theta <= 0: 
        return t_night
    return t_base + t_amp * (cos_theta ** 0.25) * scaling

#@njit(fastmath=True)
@njit()
def calculate_sticking_probability_nb(surface_temp_2):
    if surface_temp_2 <= 0: return 1.0
    p_stick = 0.0804 * np.exp(458.0 / surface_temp_2)
    p_stick_eff = p_stick / (1.0 - (1.0 - p_stick) * 0.8)
    if p_stick_eff > 1.0: return 1.0
    return p_stick_eff

#@njit(fastmath=True, parallel=True, nogil=True)
@njit()
def update_particles_numba(pos, vel, status_mask, result_codes, particle_bins, dt_step, V_rad, V_tan, r0, R_MAX, tau_ion, 
                           spec_wl, spec_gamma, JL_const, AU, subsolar_lon, U_bins, V_weights, 
                           u_model_type, sticking_model, use_prob_sticking, temp_base, temp_amp, temp_night, u_max, u_min):
    n_particles = pos.shape[0]
    HOP_TAU_THRESHOLD = 30.0
    
    #for i in prange(n_particles):
    for i in range(n_particles):
        if not status_mask[i]:
            continue

        px, py, pz = pos[i, 0], pos[i, 1], pos[i, 2]
        vx, vy, vz = vel[i, 0], vel[i, 1], vel[i, 2]
        t_rem = dt_step

        while t_rem > 1e-6:
            dt = min(t_rem, dt_step)
            
            # 1. 光電離判定
            if px > 0 and np.random.random() < (1.0 - np.exp(-dt / tau_ion)):
                r_mag = np.sqrt(px**2 + py**2 + pz**2)
                altitude = r_mag - RM_NB
                if altitude < 10000.0:
                    ln = np.arctan2(py, px)
                    lt = np.arcsin(pz / r_mag)
                    ln_fix = (ln + subsolar_lon + np.pi) % (2 * np.pi) - np.pi
                    t_impact = calculate_surface_temperature_leblanc_nb(ln_fix, lt, AU, subsolar_lon, temp_base, temp_amp, temp_night)
                    v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
                    E_eV = (0.5 * MASS_NA_NB * (v_mag**2)) / EV_TO_JOULE_NB
                    
                    particle_bins[i] = assign_sticking_bin_numba(E_eV, t_impact, U_bins, V_weights, u_model_type, u_max, u_min)
                    result_codes[i] = 1 # stuck
                    px, py, pz = px * (RM_NB / r_mag), py * (RM_NB / r_mag), pz * (RM_NB / r_mag)
                else:
                    result_codes[i] = 2 # ionized
                status_mask[i] = False
                break
                
            # -----------------------------------------------------
            # 【復元】2. 衝突予測 (Parabolic Analytical Check)
            # -----------------------------------------------------
            r_mag_now = np.sqrt(px**2 + py**2 + pz**2)
            nx_n, ny_n, nz_n = px / r_mag_now, py / r_mag_now, pz / r_mag_now
            g_mag = GM_MERCURY_NB / (r_mag_now**2)
            v_rad_local = vx * nx_n + vy * ny_n + vz * nz_n
            
            t_hit_est = 999999.0
            if v_rad_local > 0.0:
                t_hit_est = 2.0 * v_rad_local / g_mag
            elif r_mag_now > RM_NB:
                val_c = r_mag_now - RM_NB
                if val_c < 1000.0:  # 地表付近(1000m以内)のみ予測を適用
                    term_sq = v_rad_local**2 + 2.0 * g_mag * val_c
                    if term_sq >= 0.0:
                        t_hit_est = (np.abs(v_rad_local) + np.sqrt(term_sq)) / g_mag

            is_hit = False
            
            if t_hit_est < dt:
                # === 近似予測での衝突 ===
                t_flight = t_hit_est
                acc_approx_x, acc_approx_y, acc_approx_z = -nx_n * g_mag, -ny_n * g_mag, -nz_n * g_mag
                
                px_hit = px + vx * t_flight + 0.5 * acc_approx_x * (t_flight**2)
                py_hit = py + vy * t_flight + 0.5 * acc_approx_y * (t_flight**2)
                pz_hit = pz + vz * t_flight + 0.5 * acc_approx_z * (t_flight**2)
                
                # 地表（RM_NB）に正規化してめり込みを防止
                hit_mag = np.sqrt(px_hit**2 + py_hit**2 + pz_hit**2)
                px_hit = px_hit * (RM_NB / hit_mag)
                py_hit = py_hit * (RM_NB / hit_mag)
                pz_hit = pz_hit * (RM_NB / hit_mag)
                
                vx_next = vx + acc_approx_x * t_flight
                vy_next = vy + acc_approx_y * t_flight
                vz_next = vz + acc_approx_z * t_flight
                
                t_rem -= t_flight
                is_hit = True
            
                
                
            else:
                # --- 3. 軌道積分 (完全版 RK4) ---
                k1_vx, k1_vy, k1_vz = calculate_acceleration_numba(px, py, pz, vx, vy, vz, V_rad, V_tan, r0, spec_wl, spec_gamma, JL_const, AU)
                k1_px, k1_py, k1_pz = vx, vy, vz
                
                px2, py2, pz2 = px + 0.5 * dt * k1_px, py + 0.5 * dt * k1_py, pz + 0.5 * dt * k1_pz
                vx2, vy2, vz2 = vx + 0.5 * dt * k1_vx, vy + 0.5 * dt * k1_vy, vz + 0.5 * dt * k1_vz
                k2_vx, k2_vy, k2_vz = calculate_acceleration_numba(px2, py2, pz2, vx2, vy2, vz2, V_rad, V_tan, r0, spec_wl, spec_gamma, JL_const, AU)
                k2_px, k2_py, k2_pz = vx2, vy2, vz2
                
                px3, py3, pz3 = px + 0.5 * dt * k2_px, py + 0.5 * dt * k2_py, pz + 0.5 * dt * k2_pz
                vx3, vy3, vz3 = vx + 0.5 * dt * k2_vx, vy + 0.5 * dt * k2_vy, vz + 0.5 * dt * k2_vz
                k3_vx, k3_vy, k3_vz = calculate_acceleration_numba(px3, py3, pz3, vx3, vy3, vz3, V_rad, V_tan, r0, spec_wl, spec_gamma, JL_const, AU)
                k3_px, k3_py, k3_pz = vx3, vy3, vz3
                
                px4, py4, pz4 = px + dt * k3_px, py + dt * k3_py, pz + dt * k3_pz
                vx4, vy4, vz4 = vx + dt * k3_vx, vy + dt * k3_vy, vz + dt * k3_vz
                k4_vx, k4_vy, k4_vz = calculate_acceleration_numba(px4, py4, pz4, vx4, vy4, vz4, V_rad, V_tan, r0, spec_wl, spec_gamma, JL_const, AU)
                k4_px, k4_py, k4_pz = vx4, vy4, vz4
                
                dt_6 = dt / 6.0
                px_next = px + dt_6 * (k1_px + 2.0 * k2_px + 2.0 * k3_px + k4_px)
                py_next = py + dt_6 * (k1_py + 2.0 * k2_py + 2.0 * k3_py + k4_py)
                pz_next = pz + dt_6 * (k1_pz + 2.0 * k2_pz + 2.0 * k3_pz + k4_pz)
                vx_next = vx + dt_6 * (k1_vx + 2.0 * k2_vx + 2.0 * k3_vx + k4_vx)
                vy_next = vy + dt_6 * (k1_vy + 2.0 * k2_vy + 2.0 * k3_vy + k4_vy)
                vz_next = vz + dt_6 * (k1_vz + 2.0 * k2_vz + 2.0 * k3_vz + k4_vz)
                
                r_next_sq = px_next**2 + py_next**2 + pz_next**2
                
                # 境界脱離判定
                if r_next_sq > R_MAX**2:
                    result_codes[i] = 3 # escaped
                    status_mask[i] = False
                    break
                    
                # 移動後のめり込み判定
                if r_next_sq <= RM_NB**2:
                    t_rem -= dt
                    norm_mag = np.sqrt(r_next_sq)
                    px_hit = px_next * (RM_NB / norm_mag)
                    py_hit = py_next * (RM_NB / norm_mag)
                    pz_hit = pz_next * (RM_NB / norm_mag)
                    is_hit = True
                else:
                    px, py, pz = px_next, py_next, pz_next
                    vx, vy, vz = vx_next, vy_next, vz_next
                    t_rem -= dt

            # --- 4. 衝突判定・バウンド処理 ---
            if is_hit:
                ln = np.arctan2(py_hit, px_hit)
                lt = np.arcsin(pz_hit / RM_NB)
                ln_fix = (ln + subsolar_lon + np.pi) % (2 * np.pi) - np.pi
                temp_impact = calculate_surface_temperature_leblanc_nb(ln_fix, lt, AU, subsolar_lon, temp_base, temp_amp, temp_night)
                
                v_hit_mag = np.sqrt(vx_next**2 + vy_next**2 + vz_next**2)
                E_hit_eV = (0.5 * MASS_NA_NB * (v_hit_mag**2)) / EV_TO_JOULE_NB
                bin_idx = assign_sticking_bin_numba(E_hit_eV, temp_impact, U_bins, V_weights, u_model_type, u_max, u_min)
                U_assigned = U_bins[bin_idx]
                
                is_trapped = False
                v_rb_x, v_rb_y, v_rb_z = 0.0, 0.0, 0.0
                
                if sticking_model == 1: # 'baule'
                    mu = 22.99 / 16.0
                    transfer_coeff = (4.0 * mu) / ((1.0 + mu)**2)
                    v_n_scalar = (vx_next * px_hit + vy_next * py_hit + vz_next * pz_hit) / RM_NB
                    vnx, vny, vnz = v_n_scalar * (px_hit / RM_NB), v_n_scalar * (py_hit / RM_NB), v_n_scalar * (pz_hit / RM_NB)
                    vtx, vty, vtz = vx_next - vnx, vy_next - vny, vz_next - vnz
                    
                    E_n_eV = (0.5 * MASS_NA_NB * (v_n_scalar**2)) / EV_TO_JOULE_NB
                    delta_E = transfer_coeff * (E_n_eV + U_assigned)
                    
                    if delta_E >= E_n_eV:
                        is_trapped = True
                    else:
                        E_rem_n = E_n_eV - delta_E
                        v_rem_mag = np.sqrt((2.0 * E_rem_n * EV_TO_JOULE_NB) / MASS_NA_NB)
                        v_rb_x = v_rem_mag * (px_hit / RM_NB) + vtx
                        v_rb_y = v_rem_mag * (py_hit / RM_NB) + vty
                        v_rb_z = v_rem_mag * (pz_hit / RM_NB) + vtz
                else: # 'empirical'
                    if use_prob_sticking:
                        if np.random.random() < calculate_sticking_probability_nb(temp_impact):
                            is_trapped = True
                    else:
                        if temp_impact < 300.0:
                            is_trapped = True
                            
                    if not is_trapped:
                        rb_spd = sample_speed_from_flux_numba(temp_impact)
                        lx, ly, lz = sample_lambertian_direction_numba()
                        v_rb_x, v_rb_y, v_rb_z = transform_local_to_world_numba(lx, ly, lz, px_hit, py_hit, pz_hit)
                        v_rb_x *= rb_spd
                        v_rb_y *= rb_spd
                        v_rb_z *= rb_spd
                        
                # トラップされなかった場合の即時バウンド
                if not is_trapped:
                    px, py, pz = px_hit + (px_hit / RM_NB), py_hit + (py_hit / RM_NB), pz_hit + (pz_hit / RM_NB)
                    vx, vy, vz = v_rb_x, v_rb_y, v_rb_z
                    continue # t_rem はすでに上で引かれているので引かない！
                    
                # トラップされた場合の滞在時間判定
                exponent = - (U_assigned * EV_TO_JOULE_NB) / (KB_NB * temp_impact)
                td_rate = 1e13 * np.exp(exponent) if exponent >= -700 else 0.0
                tau_td = 1.0 / td_rate if td_rate > 1e-30 else 999999.0
                
                if tau_td <= HOP_TAU_THRESHOLD and t_rem > tau_td:
                    t_rem -= tau_td
                    rb_spd = sample_speed_from_flux_numba(temp_impact)
                    lx, ly, lz = sample_lambertian_direction_numba()
                    v_rb_x, v_rb_y, v_rb_z = transform_local_to_world_numba(lx, ly, lz, px_hit, py_hit, pz_hit)
                    px, py, pz = px_hit + (px_hit / RM_NB), py_hit + (py_hit / RM_NB), pz_hit + (pz_hit / RM_NB)
                    vx, vy, vz = v_rb_x * rb_spd, v_rb_y * rb_spd, v_rb_z * rb_spd
                    continue
                else:
                    particle_bins[i] = bin_idx
                    result_codes[i] = 1
                    status_mask[i] = False
                    px, py, pz = px_hit, py_hit, pz_hit
                    break
                    
        pos[i, 0], pos[i, 1], pos[i, 2] = px, py, pz
        vel[i, 0], vel[i, 1], vel[i, 2] = vx, vy, vz

#@njit(fastmath=True)
@njit()
def sample_thompson_sigmund_energy_numba(U_eV, E_max_eV):
    f_max = (U_eV / 2.0) / (U_eV / 2.0 + U_eV) ** 3
    while True:
        E_try = np.random.uniform(0.0, E_max_eV)
        f_val = E_try / (E_try + U_eV) ** 3
        if np.random.uniform(0.0, f_max) <= f_val:
            return E_try



#@njit(fastmath=True)
@njit()
def generate_particles_grid_numba(
    pos_array, vel_array, weight_array, status_mask, empty_idx,
    total_loss_step, frac_psd, frac_td, frac_sws,
    w_psd, w_td, w_sws, temp_psd, temp_night, temp_day_map, illum_frac_map,
    lon_edges, lat_edges, sub_lon, RM
):
    ptr = 0
    n_empty = empty_idx.shape[0]
    n_lon = total_loss_step.shape[0]
    n_lat = total_loss_step.shape[1]
    n_bins = total_loss_step.shape[2]
    
    # Dawn/Duskの集計用配列: [psd_dawn, psd_dusk, td_dawn, td_dusk, sws_dawn, sws_dusk]
    stats = np.zeros(6, dtype=np.float64) 
    
    for i in range(n_lon):
        lon_f = 0.5 * (lon_edges[i] + lon_edges[i+1])
        for j in range(n_lat):
            lat_f = 0.5 * (lat_edges[j] + lat_edges[j+1])
            
            diff_rad = (lon_f - sub_lon + np.pi) % (2 * np.pi) - np.pi
            is_dawn = (diff_rad < 0)
            
            nx = np.cos(lat_f) * np.cos(lon_f - sub_lon)
            ny = np.cos(lat_f) * np.sin(lon_f - sub_lon)
            nz = np.sin(lat_f)
            
            temp_day = temp_day_map[i, j]
            illum = illum_frac_map[i, j]
            temp_eff_td = temp_day if illum > 0.5 else temp_night
            
            for b in range(n_bins):
                n_lost = total_loss_step[i, j, b]
                if n_lost <= 0.0: continue
                
                # === PSD ===
                n_psd = n_lost * frac_psd[i, j, b]
                if n_psd > 0.0:
                    num_f = n_psd / w_psd
                    num = int(num_f)
                    if np.random.random() < (num_f - num): num += 1
                    for _ in range(num):
                        if ptr >= n_empty: break
                        idx = empty_idx[ptr]
                        ptr += 1
                        
                        lx, ly, lz = sample_lambertian_direction_numba()
                        spd = sample_speed_from_flux_numba(temp_psd)
                        vx, vy, vz = transform_local_to_world_numba(lx, ly, lz, nx, ny, nz)
                        
                        pos_array[idx, 0], pos_array[idx, 1], pos_array[idx, 2] = nx * RM, ny * RM, nz * RM
                        vel_array[idx, 0], vel_array[idx, 1], vel_array[idx, 2] = vx * spd, vy * spd, vz * spd
                        weight_array[idx] = w_psd
                        status_mask[idx] = True
                        
                        if is_dawn: stats[0] += w_psd
                        else:       stats[1] += w_psd
                        
                # === TD ===
                n_td = n_lost * frac_td[i, j, b]
                if n_td > 0.0:
                    num_f = n_td / w_td
                    num = int(num_f)
                    if np.random.random() < (num_f - num): num += 1
                    for _ in range(num):
                        if ptr >= n_empty: break
                        idx = empty_idx[ptr]
                        ptr += 1
                        
                        lx, ly, lz = sample_lambertian_direction_numba()
                        spd = sample_speed_from_flux_numba(temp_eff_td)
                        vx, vy, vz = transform_local_to_world_numba(lx, ly, lz, nx, ny, nz)
                        
                        pos_array[idx, 0], pos_array[idx, 1], pos_array[idx, 2] = nx * RM, ny * RM, nz * RM
                        vel_array[idx, 0], vel_array[idx, 1], vel_array[idx, 2] = vx * spd, vy * spd, vz * spd
                        weight_array[idx] = w_td
                        status_mask[idx] = True
                        
                        if is_dawn: stats[2] += w_td
                        else:       stats[3] += w_td

                # === SWS ===
                n_sws = n_lost * frac_sws[i, j, b]
                if n_sws > 0.0:
                    num_f = n_sws / w_sws
                    num = int(num_f)
                    if np.random.random() < (num_f - num): num += 1
                    for _ in range(num):
                        if ptr >= n_empty: break
                        idx = empty_idx[ptr]
                        ptr += 1
                        
                        lx, ly, lz = sample_lambertian_direction_numba()
                        E_eV = sample_thompson_sigmund_energy_numba(0.27, 5.0)
                        spd = np.sqrt(2.0 * E_eV * EV_TO_JOULE_NB / MASS_NA_NB)
                        vx, vy, vz = transform_local_to_world_numba(lx, ly, lz, nx, ny, nz)
                        
                        pos_array[idx, 0], pos_array[idx, 1], pos_array[idx, 2] = nx * RM, ny * RM, nz * RM
                        vel_array[idx, 0], vel_array[idx, 1], vel_array[idx, 2] = vx * spd, vy * spd, vz * spd
                        weight_array[idx] = w_sws
                        status_mask[idx] = True
                        
                        
                        if is_dawn: stats[4] += w_sws
                        else:       stats[5] += w_sws

    return stats

#@njit(fastmath=True)
@njit()
def fast_3d_histogram_numba(pos, weights, status_mask, bins, gmin, gmax):
    grid = np.zeros((bins, bins, bins), dtype=np.float32)
    inv_dx = bins / (gmax - gmin)
    
    n_particles = pos.shape[0]
    for i in range(n_particles):
        if status_mask[i]:
            # 座標をビンインデックスに変換
            ix = int((pos[i, 0] - gmin) * inv_dx)
            iy = int((pos[i, 1] - gmin) * inv_dx)
            iz = int((pos[i, 2] - gmin) * inv_dx)
            
            # グリッドの範囲内かチェックして加算
            if 0 <= ix < bins and 0 <= iy < bins and 0 <= iz < bins:
                grid[ix, iy, iz] += weights[i]
                
    return grid

#@njit(fastmath=True)
@njit()
def process_dead_particles_numba(pos, weight, result_codes, particle_bins,
                                 gained_grid, subsolar_lon,
                                 n_lon, n_lat, n_bins):
    """
    死んだ粒子を処理する統合関数。
    Stuckした粒子の還元、全死因の集計、およびインデックスのリセットを行う。
    """
    dlon = 2.0 * np.pi / n_lon
    dlat = np.pi / n_lat
    
    # 統計用配列: [Stuck, Ionized, Escaped]
    stats = np.zeros(3, dtype=np.float64)

    for i in range(pos.shape[0]):
        code = result_codes[i]
        if code == 0:
            continue

        w = weight[i]
        
        # 1: Stuck (地表に吸着)
        if code == 1:
            stats[0] += w
            px, py, pz = pos[i, 0], pos[i, 1], pos[i, 2]
            r = np.sqrt(px * px + py * py + pz * pz)
            if r > 0.0:
                ln = np.arctan2(py, px)
                lt = np.arcsin(max(-1.0, min(1.0, pz / r)))
                ln_fix = (ln + subsolar_lon + np.pi) % (2.0 * np.pi) - np.pi

                ix = int((ln_fix + np.pi) / dlon)
                iy = int((lt + 0.5 * np.pi) / dlat)

                if ix < 0: ix = 0
                elif ix >= n_lon: ix = n_lon - 1

                if iy < 0: iy = 0
                elif iy >= n_lat: iy = n_lat - 1

                b = particle_bins[i]
                if 0 <= b < n_bins:
                    gained_grid[ix, iy, b] += w

        # 2: Ionized (光電離)
        elif code == 2:
            stats[1] += w
            
        # 3: Escaped (宇宙空間へ脱出)
        elif code == 3:
            stats[2] += w

        result_codes[i] = 0

    return stats


#@njit(fastmath=True, parallel=True)
@njit()
def update_surface_maps_numba(
    surface_density, accumulated_gained_grid,
    cached_rate_psd, cached_rate_td, cached_rate_sws, cached_loss_rate_grid,
    temp_day_map, illum_frac_map,
    lon_centers, lat_centers, cell_areas,
    dt_accumulated, AU, sub_lon, f_uv, sw_flux,
    u_bins, v_weights, q_psd_bins, diffusion_ratio,
    temp_base, temp_amp, temp_night, use_area_weighted, use_std_diff,
    diff_pre_factor, diff_e_a_ev, kb_ev_const,
    sws_yield, sws_ref_dens, sws_lon_min, sws_lon_max,
    sws_lat_n_min, sws_lat_n_max, sws_lat_s_min, sws_lat_s_max,
    use_surface_diffusion_trapping, a_diff_rate, e_diff_ev,
    ev_to_joule, k_boltzmann
):
    n_lon = surface_density.shape[0]
    n_lat = surface_density.shape[1]
    n_bins = surface_density.shape[2]

    scaling = np.sqrt(0.306 / AU)
    dlon = lon_centers[1] - lon_centers[0]
    sin_half_width = np.sin(dlon / 2.0)

    stats_out = np.zeros((n_lon, n_lat, 2), dtype=np.float64)

    #for i in prange(n_lon):
    for i in range(n_lon):
        lon = lon_centers[i]
        lon_sun = (lon - sub_lon + np.pi) % (2.0 * np.pi) - np.pi
        mask_sws_lon = (sws_lon_min <= lon_sun) and (lon_sun <= sws_lon_max)

        for j in range(n_lat):
            lat = lat_centers[j]
            area = cell_areas[j]

            mask_sws_lat = ((sws_lat_n_min <= lat <= sws_lat_n_max) or
                            (sws_lat_s_min <= lat <= sws_lat_s_max))
            mask_sws = mask_sws_lon and mask_sws_lat

            cos_z = np.cos(lat) * np.cos(lon - sub_lon)
            eff_cos = max(0.0, cos_z)

            if cos_z > sin_half_width:
                illum = 1.0
            elif cos_z < -sin_half_width:
                illum = 0.0
            else:
                illum = (cos_z + sin_half_width) / (2.0 * sin_half_width)

            if not use_area_weighted:
                illum = 1.0 if cos_z > 0.0 else 0.0

            t_day = temp_base + temp_amp * (eff_cos ** 0.25) * scaling
            
            temp_day_map[i, j] = t_day
            illum_frac_map[i, j] = illum

            supply_std = 0.0
            if use_std_diff and t_day > 100.0:
                supply_std = diff_pre_factor * np.exp(-diff_e_a_ev / (kb_ev_const * t_day)) * dt_accumulated

            stats_out[i, j, 0] = supply_std * area
            trans_inward_sum = 0.0

            for b in range(n_bins):
                supply_b = supply_std * diffusion_ratio[b]
                r_psd = f_uv * q_psd_bins[b] * eff_cos * illum
                
                u_j = u_bins[b] * ev_to_joule
                exp_day = -u_j / (k_boltzmann * t_day)
                rate_day = 1e13 * np.exp(exp_day) if (exp_day >= -700.0 and t_day >= 10.0) else 0.0

                exp_night = -u_j / (k_boltzmann * temp_night)
                rate_night = 1e13 * np.exp(exp_night) if exp_night >= -700.0 else 0.0

                if use_area_weighted:
                    r_td = rate_day * illum + rate_night * (1.0 - illum)
                else:
                    r_td = rate_day if illum > 0.5 else rate_night

                r_sws = (sw_flux * sws_yield) / sws_ref_dens if mask_sws else 0.0
                rate_total = r_psd + r_td + r_sws

                current_dens = surface_density[i, j, b]
                gain_dens = accumulated_gained_grid[i, j, b] / area
                total_input = gain_dens + supply_b

                # 吸着分は消費したためゼロクリア（Python側のバグも修正）
                accumulated_gained_grid[i, j, b] = 0.0

                transfer_r = 0.0
                if use_surface_diffusion_trapping and b < n_bins - 1 and t_day > 10.0:
                    transfer_r = a_diff_rate * np.exp(-e_diff_ev / (kb_ev_const * t_day))

                eff_total_rate = rate_total + transfer_r
                decay_factor = np.exp(-eff_total_rate * dt_accumulated) if eff_total_rate > 1e-30 else 1.0
                total_loss_dens = current_dens * (1.0 - decay_factor)

                act_loss = 0.0
                trans_dens = 0.0
                if eff_total_rate > 1e-30:
                    act_loss = total_loss_dens * (rate_total / eff_total_rate)
                    trans_dens = total_loss_dens * (transfer_r / eff_total_rate)

                surface_density[i, j, b] = max(0.0, current_dens - total_loss_dens + total_input)
                loss_per_sec = act_loss * area / dt_accumulated

                if b < n_bins - 1:
                    max_dens = 1.0e19 * v_weights[b+1]
                    avail = max(0.0, max_dens - surface_density[i, j, b+1])
                    act_trans = min(trans_dens, avail)
                    surface_density[i, j, b] += (trans_dens - act_trans)
                    accumulated_gained_grid[i, j, b+1] += act_trans * area
                    trans_inward_sum += act_trans * area

                cached_rate_psd[i, j, b] = r_psd
                cached_rate_td[i, j, b] = r_td
                cached_rate_sws[i, j, b] = r_sws
                cached_loss_rate_grid[i, j, b] = loss_per_sec

            stats_out[i, j, 1] = trans_inward_sum

    return stats_out


# ==============================================================================
# 3. メインルーチン
# ==============================================================================
def main_snapshot_simulation():
    np.random.seed(42)
    set_numba_seed(42)
    start_time = time.time()
    # 🌟 Windowsでの複数回出力を防ぐため、ここに移動
    print(f"Diffusion Parameters: Ea={DIFF_E_A_EV}eV, RefFlux={DIFF_REF_FLUX:.1e} at {DIFF_REF_TEMP}K")

    # --- 設定読み込み ---
    DT_MOVE = SIMULATION_SETTINGS['DT_MOVE']
    DT_RATE_UPDATE = SIMULATION_SETTINGS['DT_RATE_UPDATE']
    N_LON_FIXED = SIMULATION_SETTINGS['N_LON']
    N_LAT = SIMULATION_SETTINGS['N_LAT']
    GRID_RESOLUTION = SIMULATION_SETTINGS['GRID_RESOLUTION']
    GRID_MAX_RM = SIMULATION_SETTINGS['GRID_MAX_RM']

    OUTPUT_DIRECTORY = r"./SimulationResult_202606"

    # 実行パラメータ
    INIT_SURF_DENS = 7.5e14 * (100 ** 2) * 0.0053
    SPIN_UP_YEARS = 2.0
    TOTAL_SIM_YEARS = 1.0
    TARGET_TAA = np.arange(0, 360, 1)

    # スーパーパーティクル数
    TARGET_SPS = {'TD': 100, 'PSD': 100, 'SWS': 100, 'MMV': 100}

    # ソースプロセス物理定数
    F_UV_1AU = 1.5e14 * (100 ** 2)
    TEMP_PSD = 1500.0
    TEMP_MMV = 3000.0

    SWS_PARAMS = {
        'FLUX_1AU': 10.0 * 100 ** 3 * 400e3 * 4,
        'YIELD': 0.06,
        'U_eV': 0.27,
        'REF_DENS': 7.5e14 * 100 ** 2,
        'LON_RANGE': np.deg2rad([-40, 40]),
        'LAT_N_RANGE': np.deg2rad([20, 80]),
        'LAT_S_RANGE': np.deg2rad([-80, -20]),
    }

    # === 初期化処理 ===
    mode_str = "EqMode" if SIMULATION_SETTINGS['USE_EQUILIBRIUM_MODE'] else "NoEq"

    U_BINS_ARRAY, V_WEIGHTS_ARRAY = setup_binding_energy_bins(SIMULATION_SETTINGS)
    N_BINS_DYNAMIC = len(U_BINS_ARRAY)

    # メインループで参照されるように設定を上書き
    SIMULATION_SETTINGS['U_BINS'] = U_BINS_ARRAY 
    mode = SIMULATION_SETTINGS.get('U_MODEL_TYPE', 'fixed')
    if mode == 'fixed':
        # fixedの場合は、設定で定義した Q_PSD_BINS_FIXED をそのまま使う
        SIMULATION_SETTINGS['Q_PSD_BINS'] = SIMULATION_SETTINGS['Q_PSD_BINS_FIXED']
    else:
        # dynamic や gaussian_random の場合はビン数に合わせて一律で拡張
        base_q = 2.0e-20 / (100 ** 2)
        #base_q = 2.7e-21 / (100 ** 2)
        SIMULATION_SETTINGS['Q_PSD_BINS'] = np.full(N_BINS_DYNAMIC, base_q)

    # 表面密度マップの3次元化 (N_BINS を N_BINS_DYNAMIC に変更)
    surface_density = np.zeros((N_LON_FIXED, N_LAT, N_BINS_DYNAMIC), dtype=np.float64)
    # 初期値は内部拡散用に、一番深いサイト(インデックス -1)に与える
    #surface_density[:, :, -1] = INIT_SURF_DENS
    # V_WEIGHTS_ARRAY は既に合計1.0に規格化されている前提です
    #for b in range(N_BINS_DYNAMIC):
    #    surface_density[:, :, b] = INIT_SURF_DENS * V_WEIGHTS_ARRAY[b]


    # 初期状態は深い半分に分布させてスタート
    half_idx = N_BINS_DYNAMIC // 2
    for b in range(half_idx, N_BINS_DYNAMIC):
        surface_density[:, :, b] = INIT_SURF_DENS / (N_BINS_DYNAMIC - half_idx)

    # ファイル名 (最新版に_MultiBinを付与)
    #run_name = f"ParabolicHop_{N_LON_FIXED}x{N_LAT}_{mode_str}_DT{int(DT_MOVE)}_0510_Fixed_BD0.4_U1.85_Q2.0_Bouncetau30s_A2.0_LongLT(Fulle)"
    run_name = f"ParabolicHop_{N_LON_FIXED}x{N_LAT}_{mode_str}_DT{int(DT_MOVE)}_0621_Multi_BD0.5_UG_Q2.0_Bouncetau30s_A2.0_LongLT(Fulle)_Test_2"
    #run_name = f"ParabolicHop_{N_LON_FIXED}x{N_LAT}_{mode_str}_DT{int(DT_MOVE)}_0502_BD0.4_U1.85_Q0.27_Bouncetau30s_A0.5_LongLT_CHECK"
    target_output_dir = os.path.join(OUTPUT_DIRECTORY, run_name)
    os.makedirs(target_output_dir, exist_ok=True)
    print(f"Simulation Start. Results: {target_output_dir}")
    print(f"Settings: DT_MOVE={DT_MOVE}s")

    # 表面グリッド定義
    lon_edges = np.linspace(-np.pi, np.pi, N_LON_FIXED + 1)
    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, N_LAT + 1)
    dlon = lon_edges[1] - lon_edges[0]
    cell_areas = (PHYSICAL_CONSTANTS['RM'] ** 2) * dlon * (np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1]))

    # 表面密度マップの3次元化
    # surface_density = np.zeros((N_LON_FIXED, N_LAT, N_BINS), dtype=np.float64)
    # 初期値は内部拡散用の深いサイト(IDX_DEEP)に与える
    # surface_density[:, :, IDX_DEEP] = INIT_SURF_DENS

    # 外部データ読み込み
    try:
        spec_np = np.loadtxt('SolarSpectrum_Na0.txt', usecols=(0, 3))
        orbit_data = np.loadtxt('orbit2025_spice_unwrapped.txt')
        orbit_data[:, 0] = np.rad2deg(np.unwrap(np.deg2rad(orbit_data[:, 0])))
        orbit_data[:, 5] = np.rad2deg(np.unwrap(np.deg2rad(orbit_data[:, 5])))
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # スペクトルデータ準備
    wl, gamma = spec_np[:, 0], spec_np[:, 1]
    if wl[1] < wl[0]:
        idx = np.argsort(wl)
        wl, gamma = wl[idx], gamma[idx]

    const_sigma = PHYSICAL_CONSTANTS['E_CHARGE'] ** 2 / (
            4 * PHYSICAL_CONSTANTS['ME'] * PHYSICAL_CONSTANTS['C'] * PHYSICAL_CONSTANTS['EPSILON_0'])
    spec_dict = {
        'wl': wl, 'gamma': gamma,
        'sigma0_perdnu2': const_sigma * 0.641,
        'sigma0_perdnu1': const_sigma * 0.320,
        'JL': 5.18e14
    }

    # 時間管理変数
    MERCURY_YEAR = PHYSICAL_CONSTANTS['ORBITAL_PERIOD']
    t_file_start = orbit_data[0, 2]
    t_start_spinup = t_file_start
    t_start_run = t_start_spinup + SPIN_UP_YEARS * MERCURY_YEAR
    t_end_run = t_start_run + TOTAL_SIM_YEARS * MERCURY_YEAR
    t_curr = t_start_spinup
    t_peri_file = t_file_start


    MAX_PARTICLES = 1000000 
    pos_array = np.zeros((MAX_PARTICLES, 3), dtype=np.float64)
    vel_array = np.zeros((MAX_PARTICLES, 3), dtype=np.float64)
    weight_array = np.zeros(MAX_PARTICLES, dtype=np.float64)
    status_mask = np.zeros(MAX_PARTICLES, dtype=np.bool_)
    result_codes = np.zeros(MAX_PARTICLES, dtype=np.int32)
    particle_bins = np.zeros(MAX_PARTICLES, dtype=np.int32)

    active_particles = []
    prev_taa = -999

    # マップの3次元化
    cached_rate_psd = np.zeros_like(surface_density)
    cached_rate_td = np.zeros_like(surface_density)
    cached_rate_sws = np.zeros_like(surface_density)
    cached_loss_rate_grid = np.zeros_like(surface_density)
    accumulated_gained_grid = np.zeros_like(surface_density)

    temp_day_map = np.zeros((N_LON_FIXED, N_LAT), dtype=np.float64)
    illum_frac_map = np.zeros((N_LON_FIXED, N_LAT), dtype=np.float64)

    time_since_last_update = DT_RATE_UPDATE * 2.0
    total_steps = int((t_end_run - t_start_spinup) / DT_MOVE)
    step_count = 0

    half_grid_width_rad = dlon / 2.0
    sin_half_width = np.sin(half_grid_width_rad)

    stats_data = {}
    for deg in range(360):
        stats_data[deg] = {
            'Gen_PSD': 0.0, 'Gen_TD': 0.0, 'Gen_SWS': 0.0, 'Gen_MMV': 0.0,
            'Gen_PSD_Dawn': 0.0, 'Gen_PSD_Dusk': 0.0,
            'Gen_TD_Dawn': 0.0, 'Gen_TD_Dusk': 0.0,
            'Gen_SWS_Dawn': 0.0, 'Gen_SWS_Dusk': 0.0,
            'Gen_MMV_Dawn': 0.0, 'Gen_MMV_Dusk': 0.0,
            'Loss_Stuck': 0.0, 'Loss_Ionized': 0.0, 'Loss_Escaped': 0.0,
            'Supply_Internal': 0.0,
            'Trans_Inward': 0.0,
            'Step_Count': 0
        }

    timeseries_log = []

    # === メインループ ===
    try:  # 🌟 Ctrl+Cをキャッチするための開始地点
        # 🌟 ループに入る前にPoolを1回だけ作成する
        #with Pool(cpu_count() - 1) as pool:
            
            while t_curr < t_end_run:  # 🌟 ここから時間ステップのループ
                step_count += 1

                TAA_raw, AU, V_rad, V_tan, sub_lon = get_orbital_params_linear(t_curr, orbit_data, t_peri_file)
                TAA = TAA_raw % 360.0
                time_since_last_update += DT_MOVE

                is_recording_phase = (t_curr >= t_start_run)
                current_taa_bin = int(TAA) % 360
                step_stats = {k: 0.0 for k in stats_data[0].keys()}

                # ----------------------------------------------------------------------
                # A. 表面放出率マップの更新 (Numba一括計算版)
                # ----------------------------------------------------------------------
                if time_since_last_update >= DT_RATE_UPDATE:
                    dt_accumulated = time_since_last_update
                    f_uv = F_UV_1AU / (AU ** 2)
                    sw_flux = SWS_PARAMS['FLUX_1AU'] / (AU ** 2)

                    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2.0
                    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2.0

                    diffusion_ratio = get_diffusion_supply_distribution(1.0, SIMULATION_SETTINGS['U_BINS'], SIMULATION_SETTINGS)

                    # Numbaエンジンの呼び出し
                    stats_out = update_surface_maps_numba(
                        surface_density, accumulated_gained_grid,
                        cached_rate_psd, cached_rate_td, cached_rate_sws, cached_loss_rate_grid,
                        temp_day_map, illum_frac_map,
                        lon_centers, lat_centers, cell_areas,
                        dt_accumulated, AU, sub_lon, f_uv, sw_flux,
                        SIMULATION_SETTINGS['U_BINS'], V_WEIGHTS_ARRAY, SIMULATION_SETTINGS['Q_PSD_BINS'], diffusion_ratio,
                        SIMULATION_SETTINGS['TEMP_BASE'], SIMULATION_SETTINGS['TEMP_AMP'], SIMULATION_SETTINGS['TEMP_NIGHT'],
                        SIMULATION_SETTINGS['USE_AREA_WEIGHTED_FLUX'], SIMULATION_SETTINGS['USE_STD_DIFFUSION'],
                        DIFF_PRE_FACTOR, DIFF_E_A_EV, KB_EV_CONST,
                        SWS_PARAMS['YIELD'], SWS_PARAMS['REF_DENS'],
                        SWS_PARAMS['LON_RANGE'][0], SWS_PARAMS['LON_RANGE'][1],
                        SWS_PARAMS['LAT_N_RANGE'][0], SWS_PARAMS['LAT_N_RANGE'][1],
                        SWS_PARAMS['LAT_S_RANGE'][0], SWS_PARAMS['LAT_S_RANGE'][1],
                        SIMULATION_SETTINGS.get('USE_SURFACE_DIFFUSION_TRAPPING', False),
                        SIMULATION_SETTINGS.get('A_DIFF_RATE', 0.0), SIMULATION_SETTINGS.get('E_DIFF_EV', 0.0),
                        PHYSICAL_CONSTANTS['EV_TO_JOULE'], PHYSICAL_CONSTANTS['K_BOLTZMANN']
                    )

                    step_stats['Supply_Internal'] += np.sum(stats_out[:, :, 0])
                    step_stats['Trans_Inward'] += np.sum(stats_out[:, :, 1])

                    time_since_last_update = 0.0

                # ----------------------------------------------------------------------
                # B. 粒子の生成 (Numba化版)
                # ----------------------------------------------------------------------
                
                # 🌟 毎ステップ1回だけ、空き箱のリストを取得する！（超重要）
                empty_idx = np.where(~status_mask)[0]
                empty_ptr = 0  # 取得した空き箱を使うためのポインタ
                n_empty = len(empty_idx)

                # MMVの生成 (量は少ないのでPythonでサクッと処理)
                mmv_flux = calculate_mmv_flux(AU)
                n_mmv = mmv_flux * 4 * np.pi * PHYSICAL_CONSTANTS['RM'] ** 2 * DT_MOVE
                w_mmv = max(1.0, n_mmv / (TARGET_SPS['MMV'] * (DT_MOVE / DT_RATE_UPDATE)))

                if n_mmv > 0:
                    num_p = int(n_mmv / w_mmv)
                    if np.random.random() < (n_mmv / w_mmv - num_p): num_p += 1

                    for _ in range(num_p):
                        # 🌟 ループ内での np.where をやめ、ポインタを進めるだけに。
                        if empty_ptr >= n_empty:
                            break
                        
                        idx = empty_idx[empty_ptr]
                        empty_ptr += 1

                        lr = np.random.uniform(-np.pi, np.pi)
                        latr = np.arcsin(np.random.uniform(-1, 1))

                        diff_rad_mmv = (lr - sub_lon + np.pi) % (2 * np.pi) - np.pi
                        is_dawn_mmv = (diff_rad_mmv < 0)

                        pos = lonlat_to_xyz(lr, latr, PHYSICAL_CONSTANTS['RM'])
                        norm = pos / PHYSICAL_CONSTANTS['RM']
                        spd = sample_speed_from_flux_distribution(PHYSICAL_CONSTANTS['MASS_NA'], TEMP_MMV)
                        vel = spd * transform_local_to_world(sample_lambertian_direction_local(), norm)
                        
                        pos_array[idx] = pos
                        vel_array[idx] = vel
                        weight_array[idx] = w_mmv
                        status_mask[idx] = True

                        step_stats['Gen_MMV'] += w_mmv
                        if is_dawn_mmv: step_stats['Gen_MMV_Dawn'] += w_mmv
                        else:           step_stats['Gen_MMV_Dusk'] += w_mmv

                # グリッドからの放出 (PSD, TD, SWS をNumbaで超高速処理)
                total_loss_step = cached_loss_rate_grid * DT_MOVE
                with np.errstate(divide='ignore', invalid='ignore'):
                    rate_tot = cached_rate_psd + cached_rate_td + cached_rate_sws
                    frac_psd = np.where(rate_tot > 0, cached_rate_psd / rate_tot, 0)
                    frac_td = np.where(rate_tot > 0, cached_rate_td / rate_tot, 0)
                    frac_sws = np.where(rate_tot > 0, cached_rate_sws / rate_tot, 0)

                scale_factor = DT_MOVE / DT_RATE_UPDATE
                atoms_psd = np.sum(total_loss_step * frac_psd)
                atoms_td = np.sum(total_loss_step * frac_td)
                atoms_sws = np.sum(total_loss_step * frac_sws)

                w_psd = max(1.0, atoms_psd / (TARGET_SPS['PSD'] * scale_factor))
                w_td = max(1.0, atoms_td / (TARGET_SPS['TD'] * scale_factor))
                w_sws = max(1.0, atoms_sws / (TARGET_SPS['SWS'] * scale_factor))

                # 🌟 まだ空き箱が残っていれば生成関数を呼ぶ（残りの空き箱スライスを渡す）
                if empty_ptr < n_empty:
                    gen_stats = generate_particles_grid_numba(
                        pos_array, vel_array, weight_array, status_mask, empty_idx[empty_ptr:],
                        total_loss_step, frac_psd, frac_td, frac_sws,
                        w_psd, w_td, w_sws, TEMP_PSD, SIMULATION_SETTINGS['TEMP_NIGHT'], 
                        temp_day_map, illum_frac_map, lon_edges, lat_edges, sub_lon, PHYSICAL_CONSTANTS['RM']
                    )
                    
                    # 統計の反映
                    step_stats['Gen_PSD_Dawn'] += gen_stats[0]
                    step_stats['Gen_PSD_Dusk'] += gen_stats[1]
                    step_stats['Gen_TD_Dawn']  += gen_stats[2]
                    step_stats['Gen_TD_Dusk']  += gen_stats[3]
                    step_stats['Gen_SWS_Dawn'] += gen_stats[4]
                    step_stats['Gen_SWS_Dusk'] += gen_stats[5]
                    
                    step_stats['Gen_PSD'] += gen_stats[0] + gen_stats[1]
                    step_stats['Gen_TD']  += gen_stats[2] + gen_stats[3]
                    step_stats['Gen_SWS'] += gen_stats[4] + gen_stats[5]
                else:
                    print("WARNING: Particle array full! Dropping grid particles.")

                # ----------------------------------------------------------------------
                # C. 粒子の移動 (Numba 一括計算) & 消滅理由の集計
                # ----------------------------------------------------------------------
                r0 = AU * PHYSICAL_CONSTANTS['AU']
                tau_ion = SIMULATION_SETTINGS['T1AU'] * AU ** 2
                r_max_val = PHYSICAL_CONSTANTS['RM'] * SIMULATION_SETTINGS['GRID_RADIUS_RM']
                
                # Numba用に文字列設定を整数フラグに変換
                u_mode_flag = 0 if SIMULATION_SETTINGS['U_MODEL_TYPE'] == 'fixed' else (1 if SIMULATION_SETTINGS['U_MODEL_TYPE'] == 'gaussian_random' else 2)
                stick_model_flag = 1 if SIMULATION_SETTINGS['STICKING_MODEL'] == 'baule' else 0
                
                n_particles = pos_array.shape[0]
                
                update_particles_numba(
                    pos_array, vel_array, status_mask, result_codes, particle_bins,
                    DT_MOVE, V_rad, V_tan, r0, r_max_val, tau_ion, 
                    spec_dict['wl'], spec_dict['gamma'], spec_dict['JL'], AU, sub_lon,
                    U_BINS_ARRAY, V_WEIGHTS_ARRAY, u_mode_flag, stick_model_flag,
                    SIMULATION_SETTINGS['USE_PROBABILISTIC_STICKING'],
                    SIMULATION_SETTINGS['TEMP_BASE'], SIMULATION_SETTINGS['TEMP_AMP'], SIMULATION_SETTINGS['TEMP_NIGHT'],
                    SIMULATION_SETTINGS['U_MAX'], SIMULATION_SETTINGS['U_MIN'],
                )

                # 死亡粒子の処理と、消滅理由の集計
                loss_stats = process_dead_particles_numba(
                    pos_array, weight_array, result_codes, particle_bins,
                    accumulated_gained_grid, sub_lon,
                    N_LON_FIXED, N_LAT, N_BINS_DYNAMIC
                )
                
                step_stats['Loss_Stuck'] += loss_stats[0]
                step_stats['Loss_Ionized'] += loss_stats[1]
                step_stats['Loss_Escaped'] += loss_stats[2]
                
                # ----------------------------------------------------------------------
                # 統計データの蓄積 (スピンアップ後のみ)
                # ----------------------------------------------------------------------
                if is_recording_phase:
                    tgt = stats_data[current_taa_bin]
                    for key in step_stats:
                        if key != 'Step_Count':
                            tgt[key] += step_stats[key]
                    tgt['Step_Count'] += 1

                # ----------------------------------------------------------------------
                # D. データ保存
                # ----------------------------------------------------------------------
                if prev_taa != -999:
                    passed = False
                    for tgt in TARGET_TAA:
                        if (prev_taa < tgt <= TAA) or (prev_taa > 350 and TAA < 10 and tgt == 0):
                            passed = True
                            break

                    if passed:
                        rel_h = (t_curr - t_start_spinup) / 3600.0
                        print(f"[SAVE] TAA={TAA:.1f}, Time={rel_h:.1f}h, Particles={np.sum(status_mask)}")

                        timeseries_log.append({
                            'Time_hours': rel_h,
                            'TAA': TAA,
                            'Gen_Total': step_stats['Gen_PSD'] + step_stats['Gen_TD'] + step_stats['Gen_SWS'] + step_stats['Gen_MMV'],
                            # ... (中略) ...
                            'Trans_Inward': step_stats['Trans_Inward']
                        })

                        dgrid = np.zeros((GRID_RESOLUTION, GRID_RESOLUTION, GRID_RESOLUTION), dtype=np.float32)
                        gmin, gmax = -GRID_MAX_RM * PHYSICAL_CONSTANTS['RM'], GRID_MAX_RM * PHYSICAL_CONSTANTS['RM']
                        cvol = ((gmax - gmin) / GRID_RESOLUTION) ** 3

                        # 🌟 配列のコピー（抽出）処理を削除し、直接Numba関数へ渡す
                        H = fast_3d_histogram_numba(pos_array, weight_array, status_mask, GRID_RESOLUTION, gmin, gmax)
                        dgrid = H / cvol

                        fname_d = f"density_grid_t{int(rel_h):05d}_taa{int(round(TAA)):03d}.npy"
                        np.save(os.path.join(target_output_dir, fname_d), dgrid)
                        np.save(os.path.join(target_output_dir, f"surface_density_t{int(rel_h):05d}.npy"), surface_density)

                if step_count % 100 == 0:
                    elapsed = time.time() - start_time
                    progress_pct = (step_count / total_steps) * 100
                    print(f"Step {step_count}/{total_steps} ({progress_pct:.1f}%) | TAA={TAA:.2f} | Particles={np.sum(status_mask)} | Elapsed={elapsed:.1f}s")

                prev_taa = TAA
                t_curr += DT_MOVE

    # 🌟 Ctrl+C が押されたらここに来る（whileループと同階層）
    except KeyboardInterrupt:
        print("\n[中断] Ctrl+Cが押されました。すべての子プロセスを安全に終了します...")
        #pool.terminate()  # 実行中のプロセスを強制終了
        #pool.join()       # ゾンビ化を防ぐための終了待ち
        sys.exit(0)       # 完全にプログラムを終了する

    # === ループ終了後：CSVへの書き出し ===
    print("Saving TAA-binned statistics...")
    csv_filename = os.path.join(target_output_dir, "budget_statistics_per_taa.csv")

    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        headers = ['TAA_Bin',
                   'Gen_Total', 'Gen_PSD', 'Gen_TD', 'Gen_SWS', 'Gen_MMV',
                   'Gen_PSD_Dawn', 'Gen_PSD_Dusk', 'Gen_TD_Dawn', 'Gen_TD_Dusk',
                   'Gen_SWS_Dawn', 'Gen_SWS_Dusk', 'Gen_MMV_Dawn', 'Gen_MMV_Dusk',
                   'Pct_PSD', 'Pct_TD', 'Pct_SWS', 'Pct_MMV',
                   'Loss_Total', 'Loss_Stuck', 'Loss_Ionized', 'Loss_Escaped',
                   'Supply_Internal', 'Trans_Inward',
                   'Pct_Stuck', 'Pct_Ionized', 'Pct_Escaped']
        writer.writerow(headers)

        for deg in range(360):
            d = stats_data[deg]
            gen_total = d['Gen_PSD'] + d['Gen_TD'] + d['Gen_SWS'] + d['Gen_MMV']
            loss_total = d['Loss_Stuck'] + d['Loss_Ionized'] + d['Loss_Escaped']

            def safe_pct(val, total):
                return (val / total * 100.0) if total > 0 else 0.0

            row = [
                deg,
                f"{gen_total:.4e}",
                f"{d['Gen_PSD']:.4e}", f"{d['Gen_TD']:.4e}", f"{d['Gen_SWS']:.4e}", f"{d['Gen_MMV']:.4e}",
                f"{d['Gen_PSD_Dawn']:.4e}", f"{d['Gen_PSD_Dusk']:.4e}",
                f"{d['Gen_TD_Dawn']:.4e}", f"{d['Gen_TD_Dusk']:.4e}",
                f"{d['Gen_SWS_Dawn']:.4e}", f"{d['Gen_SWS_Dusk']:.4e}",
                f"{d['Gen_MMV_Dawn']:.4e}", f"{d['Gen_MMV_Dusk']:.4e}",
                f"{safe_pct(d['Gen_PSD'], gen_total):.1f}",
                f"{safe_pct(d['Gen_TD'], gen_total):.1f}",
                f"{safe_pct(d['Gen_SWS'], gen_total):.1f}",
                f"{safe_pct(d['Gen_MMV'], gen_total):.1f}",
                f"{loss_total:.4e}",
                f"{d['Loss_Stuck']:.4e}", f"{d['Loss_Ionized']:.4e}", f"{d['Loss_Escaped']:.4e}",
                
                f"{d['Supply_Internal']:.4e}", 
                f"{d['Trans_Inward']:.4e}",
                
                f"{safe_pct(d['Loss_Stuck'], loss_total):.1f}",
                f"{safe_pct(d['Loss_Ionized'], loss_total):.1f}",
                f"{safe_pct(d['Loss_Escaped'], loss_total):.1f}"
            ]
            writer.writerow(row)

    print(f"Statistics saved to {csv_filename}")

    import pandas as pd
    ts_csv_filename = os.path.join(target_output_dir, "budget_timeseries.csv")
    pd.DataFrame(timeseries_log).to_csv(ts_csv_filename, index=False)
    print(f"Timeseries log saved to {ts_csv_filename}")

    print("Done. Simulation Completed.")


if __name__ == '__main__':
    sys.modules['__main__'].__spec__ = None
    main_snapshot_simulation()
    