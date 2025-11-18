# -*- coding: utf-8 -*-
"""
==============================================================================
概要
==============================================================================
シミュレーションコード (main_snapshot_simulation) が生成した
表面密度グリッド (.npy ファイル) を読み込み、
【太陽固定回転座標系 (MSO)】の地図としてプロットします。

実行前に、以下の【設定】セクションを、
実行したシミュレーションコードに合わせて変更してください。

★ 更新 (2025/11/15) ★
- 惑星固定座標系 (Fixed) ではなく、太陽固定回転座標系 (MSO) で
  プロットするように変更しました。
  (太陽直下点が常に経度0度に来るようにデータを回転させます)
- このため、'orbit2025_v5.txt' ファイルの読み込みが必須になりました。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  # 対数スケール用
import glob
import os
import re
import sys  # sys.exit のために追加

# ==============================================================================
# 【設定】
# 実行したシミュレーションコードの該当箇所に合わせて変更してください。
# ==============================================================================

# 1. シミュレーションコードで設定したグリッド解像度
N_LON, N_LAT = 72, 36

# 2. シミュレーションコードで設定した出力ディレクトリ
#    (シミュレーションコード内の `OUTPUT_DIRECTORY`)
BASE_OUTPUT_DIRECTORY = r"./SimulationResult_202510"

# 3. シミュレーションコードで自動生成される `run_name`
#    (例: f"DynamicGrid{N_LON_FIXED}x{N_LAT}_2.0")
RUN_NAME = f"DynamicGrid{N_LON}x{N_LAT}_3.0"

# 4. プロット対象のファイル
#    'latest'   : 見つかったファイルのうち、時間が最新（TAAが最後）のもの
#    'earliest' : 見つかったファイルのうち、時間が最初（TAAが最初）のもの
#    数値 (例: 90): 指定したTAA [度] に該当する最新のファイル
#    ファイル名 (例: 'surface_density_t00123_taa090.npy'): 特定のファイル名
FILE_TO_PLOT = 180  # (例: 90, 180, 'latest' など)

# 5. カラースケール
#    True : 対数スケール (LogNorm) でプロット (密度の変化範囲が広い場合に推奨)
#    False: 線形スケール (Linear) でプロット
USE_LOG_SCALE = True

# 6. 軌道データファイル名 (シミュレーションで使用したもの)
ORBIT_FILE_PATH = 'orbit2025_v5.txt'

# ==============================================================================
# ★ MSO座標系変換のためのヘルパー関数 (シミュレーションコードから抜粋)
# ==============================================================================

# 水星の自転周期 [s]
ROTATION_PERIOD_SEC = 58.646 * 24 * 3600


def get_simulation_time_details(orbit_file_path):
    """
    軌道ファイルを読み込み、シミュレーションの基準時刻 t_start_run を計算する
    (t_start_run = 軌道ファイル上の TAA=0 の時刻)
    """
    try:
        orbit_data = np.loadtxt(orbit_file_path)
        taa_col = orbit_data[:, 0]
        time_col = orbit_data[:, 2]
        # TAA=0 (近日点) のインデックスを探す
        idx_perihelion = np.argmin(np.abs(taa_col))
        t_start_run = time_col[idx_perihelion]  # TAA=0 を RUN の開始時刻とする
        print(f"軌道ファイル読み込み成功: 基準時刻 (TAA=0) t_start_run = {t_start_run:.1f} s")
        return t_start_run
    except FileNotFoundError:
        print(f"エラー: 軌道ファイル '{orbit_file_path}' が見つかりません。")
        print("シミュレーションコードと同じディレクトリに配置してください。")
        return None
    except Exception as e:
        print(f"エラー: 軌道ファイルの読み込み中にエラーが発生しました。 {e}")
        return None


def calculate_subsolar_lon_rad(time_h, t_start_run):
    """
    ファイル名の時間(hour)から、その時点での
    太陽直下点経度(惑星固定座標系) [rad] を計算する
    """
    # ファイル名の時間(hour) -> シミュレーションRUN開始からの秒数
    relative_time_sec = float(time_h) * 3600.0
    # シミュレーション絶対時刻 (スピンアップ開始からの時刻)
    t_sec = relative_time_sec + t_start_run

    # 太陽直下点経度を計算
    subsolar_lon_rad = (2 * np.pi * t_sec / ROTATION_PERIOD_SEC) % (2 * np.pi)

    # 経度を -pi から +pi (-180° から +180°) の範囲に調整
    subsolar_lon_rad = (subsolar_lon_rad + np.pi) % (2 * np.pi) - np.pi

    return subsolar_lon_rad


# ==============================================================================
# ファイル検索関数 (TAA指定対応版)
# ==============================================================================

def find_target_file(target_dir, preference):
    """
    指定されたディレクトリからプロット対象のファイルを探し、
    (ファイルパス, 時間[h]) のタプルを返す。
    """

    search_path = os.path.join(target_dir, "surface_density_*.npy")
    files = glob.glob(search_path)

    if not files:
        print(f"エラー: ディレクトリ '{target_dir}' に 'surface_density_*.npy' ファイルが見つかりません。")
        print("設定 (BASE_OUTPUT_DIRECTORY, RUN_NAME) が正しいか確認してください。")
        return None

    # (時間[h], TAA[deg], ファイルパス) のタプルのリストを作成
    file_info_list = []
    for f in files:
        # ファイル名から時間(t)とTAAを抽出 (r'...' のクォートが必須)
        match = re.search(r'_t(\d+)_taa(\d+)\.npy$', os.path.basename(f))
        if match:
            time_h = int(match.group(1))  # 時間 (int)
            taa = int(match.group(2))  # TAA (int)
            file_info_list.append((time_h, taa, f))

    if not file_info_list:
        print(f"エラー: 'surface_density_*.npy' ファイルは見つかりましたが、命名規則 (t..._taa...) に一致しません。")
        return None

    # 時間(t)順にソート (これが基準)
    file_info_list.sort(key=lambda x: x[0])

    # --- 1. preference が 数値 (int or float) の場合 (TAA指定) ---
    # ★ このブロックを一番最初に移動し、優先的にチェックする
    if isinstance(preference, (int, float)):
        target_taa = int(preference)
        print(f"TAA={target_taa} に該当するファイルを探しています...")

        found_files_info = []
        for time_h, taa, f in file_info_list:
            if taa == target_taa:
                found_files_info.append((time_h, taa, f))

        if not found_files_info:
            print(f"エラー: TAA={target_taa} に該当するファイルが見つかりませんでした。")
            print(f"利用可能なTAAを確認するか、設定 (FILE_TO_PLOT) を 'latest' に変更してみてください。")
            return None

        # 該当ファイルが複数ある場合（例: 2周目）、
        # リストは既にtでソート済なので、最後（最新）のものを返す
        latest_match = found_files_info[-1]
        print(f"TAA={target_taa} のファイル (t={latest_match[0]}h) を使用します。")
        return latest_match[2], latest_match[0]  # (filepath, time_h)

    # --- 2. preference が 'latest' または 'earliest' (文字列) の場合 ---
    # ★ `elif` に変更
    elif isinstance(preference, str) and preference == 'latest':
        latest_info = file_info_list[-1]
        return latest_info[2], latest_info[0]  # (filepath, time_h)

    elif isinstance(preference, str) and preference == 'earliest':
        earliest_info = file_info_list[0]
        return earliest_info[2], earliest_info[0]  # (filepath, time_h)

    # --- 3. preference が その他の文字列 の場合 (ファイル名指定) ---
    # ★ `elif` に変更
    elif isinstance(preference, str):
        target_file = os.path.join(target_dir, preference)
        if os.path.exists(target_file):
            # ファイル名から time_h を抽出
            match = re.search(r'_t(\d+)_taa(\d+)\.npy$', os.path.basename(target_file))
            if match:
                time_h = int(match.group(1))
                return target_file, time_h  # (filepath, time_h)
            else:
                print(f"エラー: 指定されたファイル名 '{preference}' が命名規則に一致せず、時間を抽出できません。")
                return None
        else:
            print(f"エラー: 指定されたファイル '{target_file}' が見つかりません。")

            # ★ TAA指定と間違えている可能性を指摘するヒントを追加
            if preference.isdigit():
                print(
                    f"ヒント: TAA={preference} を指定したい場合は、FILE_TO_PLOT = {preference} のように引用符を付けずに数値で指定してください。")
            return None

    # --- 4. それ以外 (該当なし) ---
    # ★ `else` に変更
    else:
        print(f"エラー: FILE_TO_PLOT の設定 '{preference}' (型: {type(preference)}) が認識できません。")
        return None

# ==============================================================================
# プロット関数 (★ MSO座標系対応版)
# ==============================================================================

def plot_surface_grid(filepath, time_h, n_lon, n_lat, use_log, t_start_run):
    """
    表面密度グリッドを読み込み、MSO座標系（太陽固定）でプロットする。
    """
    try:
        # データを読み込み (形状は (N_LON, N_LAT))
        data_fixed = np.load(filepath)
    except Exception as e:
        print(f"エラー: ファイル '{filepath}' の読み込みに失敗しました。")
        print(e)
        return

    if data_fixed.shape != (n_lon, n_lat):
        print(f"エラー: データの形状 {data_fixed.shape} が設定 ({n_lon}, {n_lat}) と一致しません。")
        print("【設定】の N_LON, N_LAT を確認してください。")
        return

    # --- ★ 1. 太陽直下点経度を計算 ---
    # (このスナップショット時点での、惑星固定座標系における太陽直下点の経度)
    subsolar_lon_rad_fixed = calculate_subsolar_lon_rad(time_h, t_start_run)
    subsolar_lon_deg_fixed = np.rad2deg(subsolar_lon_rad_fixed)

    # --- ★ 2. データを MSO 座標系に回転（シフト） ---

    # 経度1グリッドあたりの角度 [deg]
    dlon_deg = 360.0 / n_lon

    # 固定座標での太陽直下点経度を、グリッドのインデックス数に換算
    # (経度は -180 ~ +180 で定義されているため)
    shift_indices = int(np.round(subsolar_lon_deg_fixed / dlon_deg))

    # np.roll(data, -shift_indices, axis=0)
    # これにより、固定座標でインデックスが `shift_indices` だったデータ
    # (つまり太陽直下点) が、インデックス 0 (MSO座標の経度 -180度地点)
    # ではなく、(インデックス 0 は -180度なので)、
    # インデックス 0 のデータ (固定座標の -180度) が、
    # MSO座標の (-180 - subsolar_lon) 度 の場所 (インデックス -shift_indices) に移動する。
    #
    # np.roll(data, shift) は data_rolled[i] = data[i - shift]
    # MSO座標のインデックス i_mso は、固定座標のインデックス i_fixed - shift_indices
    # data_mso[i_mso] = data_fixed[i_fixed]
    # data_mso[i_fixed - shift_indices] = data_fixed[i_fixed]
    # これは np.roll(data_fixed, -shift_indices, axis=0) と等価。
    data_mso = np.roll(data_fixed, shift=-shift_indices, axis=0)

    print(f"--- 座標変換情報 ---")
    print(f"ファイル時刻 (t): {time_h} h")
    print(f"基準時刻 (t_start_run): {t_start_run:.1f} s")
    print(f"太陽直下点経度 (固定座標): {subsolar_lon_deg_fixed:.2f} 度")
    print(f"データを {shift_indices} グリッド ({-shift_indices * dlon_deg:.2f} 度相当) シフトしました。")
    print(f"---")

    # --- 座標の準備 (pcolormesh用) ---
    # 経度: -180° から +180° まで (MSO座標系)
    lon_edges_deg = np.linspace(-180, 180, n_lon + 1)
    # 緯度: -90° から +90° まで
    lat_edges_deg = np.linspace(-90, 90, n_lat + 1)

    # --- プロット ---
    plt.figure(figsize=(12, 6))
    ax = plt.gca()

    # データを (N_LAT, N_LON) の形状に転置
    data_to_plot = data_mso.T

    # 最小値が0または負の場合、LogNormがエラーになるためクリップ
    vmin = None
    if use_log:
        min_positive_val = np.min(data_to_plot[data_to_plot > 0])
        if np.isnan(min_positive_val):
            min_positive_val = 1e-10  # 全て0の場合のフォールバック
        data_to_plot = np.clip(data_to_plot, min_positive_val, None)
        norm = LogNorm(vmin=min_positive_val, vmax=np.max(data_to_plot))
    else:
        norm = None  # 線形スケール

    # pcolormesh でグリッドを描画
    mesh = ax.pcolormesh(
        lon_edges_deg,
        lat_edges_deg,
        data_to_plot,
        shading='auto',
        cmap='viridis',
        norm=norm
    )

    # カラーバーを追加
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label('Surface Density [atoms/m^2]')

    # 軸とタイトルの設定 (★ MSO用に変更)
    ax.set_xlabel('Longitude (MSO / Sun-Fixed) [degrees]')
    ax.set_ylabel('Latitude [degrees]')
    ax.set_title(f'Surface Density Map (Sun-Fixed Coordinates)\n{os.path.basename(filepath)}')

    # 軸の目盛りを調整 (経度0度が太陽直下点)
    ax.set_xticks(np.linspace(-180, 180, 13))  # 30度ごと
    ax.set_yticks(np.linspace(-90, 90, 7))  # 30度ごと

    # 太陽直下点 (0, 0) と 夜側中心 (-180, 0) に縦線を引く
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Subsolar Point (Noon)')
    ax.axvline(-180, color='blue', linestyle='--', linewidth=1.0, label='Anti-solar Point (Midnight)')
    ax.axvline(180, color='blue', linestyle='--', linewidth=1.0)
    ax.legend(loc='upper right')

    ax.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()


# ==============================================================================
# メイン実行部
# ==============================================================================

if __name__ == "__main__":

    print(f"--- 表面密度プロッター (MSO座標系) ---")

    # --- 1. 軌道ファイルを読み込み、基準時刻を取得 ---
    t_start_run = get_simulation_time_details(ORBIT_FILE_PATH)

    if t_start_run is None:
        print("基準時刻の取得に失敗したため、終了します。")
        sys.exit(1)  # エラーで終了

    # --- 2. プロット対象のファイルパスと時間を取得 ---
    full_output_dir = os.path.join(BASE_OUTPUT_DIRECTORY, RUN_NAME)

    result = find_target_file(full_output_dir, FILE_TO_PLOT)

    if result:
        target_file, time_h = result
        print(f"プロット対象ファイル: {target_file}")

        # --- 3. プロット関数を実行 ---
        plot_surface_grid(target_file, time_h, N_LON, N_LAT, USE_LOG_SCALE, t_start_run)

    else:
        print("プロット対象ファイルが見つからなかったため、終了します。")