import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def run(base_dir, out_base, csv_dir, processed_dates, config):
    """
    パイプラインの最後に呼び出される、複数日の集計・PA補正・年次CSV更新・プロット実行関数
    """
    agg_conf = config.get("aggregation", {})
    correction_file = Path(agg_conf.get("pa_correction_file", ""))

    print("\n" + "=" * 60)
    print(f"--- 最終結果のCSV統合 (Yearly Smart Update) を開始します ---")
    print("=" * 60)

    if not processed_dates:
        print("  > エラー: 処理対象の日付データがありませんでした。")
        return

    # --- 1. 位相角(PA)補正ファイルの読み込み ---
    pa_correction_coeffs = None
    if correction_file.exists():
        try:
            pa_data, cf_data = np.loadtxt(correction_file, usecols=(0, 1), unpack=True)
            pa_correction_coeffs = np.polyfit(pa_data, cf_data, 2)
            print(f"  > 位相角補正ファイル ({correction_file.name}) を読み込み、2次フィットを実行しました。")
        except Exception as e:
            print(f"  > 警告: 位相角補正ファイルの読み込みに失敗しました ({e})。補正係数は 1.0 とします。")

    d2r = np.pi / 180.0

    # 年（YYYY）ごとにデータをまとめるための辞書
    yearly_results = {}
    taa_for_plot = {}
    cf_for_plot = {}

    # --- 2. 各日付のデータをループ処理して抽出 ---
    for day in processed_dates:
        # 日付文字列 (YYYYMMDD) の先頭4文字を「年」として扱う
        year = str(day)[:4]

        if year not in yearly_results:
            yearly_results[year] = []
            taa_for_plot[year] = []
            cf_for_plot[year] = []

        day_dir = out_base / day
        summary_file = day_dir / f'Final_Summary_{day}.txt'
        region_file = day_dir / 'Na_atoms_regions.dat'
        csv_file_path = csv_dir / f"mcparams{day}.csv"

        try:
            PA, TAA, avg_atoms, integrated_err = np.loadtxt(summary_file)

            # CSV読み込み
            if csv_file_path.exists():
                df = pd.read_csv(csv_file_path)
                # 水星データの行を取得
                obs_row = df[df['Type'].str.strip() == 'MERCURY'].iloc[0]

                Rms = obs_row['mercury_sun_distance_au']
                beta = obs_row['ecliptic_latitude_deg']
                elon = obs_row['ecliptic_longitude_deg']

                # ★修正点: CSVにある 'terminator_side' をそのまま使う
                if 'terminator_side' in obs_row:
                    obs_type = str(obs_row['terminator_side']).strip()
                else:
                    obs_type = "Unknown"
            else:
                # CSVがない場合のフォールバック（基本ここには来ないはず）
                Rms, beta, elon = np.nan, np.nan, np.nan
                obs_type = "Unknown"

        except Exception as e:
            print(f"  > 警告: {day} の基本データ読み込みに失敗しました ({e})。スキップします。")
            continue

        # 領域データの抽出
        reg_means = {'Region_Avg': np.nan, 'North': np.nan, 'South': np.nan, 'Equator': np.nan, 'Dawn': np.nan,
                     'SS': np.nan, 'Dusk': np.nan}
        if region_file.exists():
            try:
                rdata = np.loadtxt(region_file)
                if rdata.ndim == 1: rdata = rdata.reshape(1, -1)

                def get_valid_mean(col_idx):
                    if col_idx >= rdata.shape[1]: return np.nan
                    vals = rdata[:, col_idx]
                    valid_vals = vals[vals > 0]
                    return np.nanmean(valid_vals) if len(valid_vals) > 0 else np.nan

                reg_means['Region_Avg'] = get_valid_mean(1)
                reg_means['North'] = get_valid_mean(2)
                reg_means['South'] = get_valid_mean(3)
                reg_means['Equator'] = get_valid_mean(4)
                if rdata.shape[1] >= 8:
                    reg_means['Dawn'] = get_valid_mean(5)
                    reg_means['SS'] = get_valid_mean(6)
                    reg_means['Dusk'] = get_valid_mean(7)
            except Exception:
                pass

        # --- (D) 物理量の補正と計算 ---
        pa_correction_factor = np.polyval(pa_correction_coeffs, PA) if pa_correction_coeffs is not None else 1.0
        h = Rms * np.sin(beta * d2r)

        # 水星の半球の表面積 (cm^2)
        half_surface_area = 3.74e17

        # 1D由来: 位相角補正あり & なし の平均柱密度
        cd_1d_raw = avg_atoms / half_surface_area
        cd_1d_corrected = (avg_atoms / pa_correction_factor) / half_surface_area

        cd_err_corrected = (integrated_err / pa_correction_factor) / half_surface_area

        result_row = {
            'Date': str(day),
            'Observation_Type': obs_type,  # ここにCSVから読んだ Dawn/Dusk が入る
            'TAA_deg': TAA,
            'PhaseAngle_deg': PA,
            'Distance_Rms_au': Rms,
            'Lat_h': h,
            'Ecliptic_Lon': elon,
            'Ecliptic_Lat': beta,
            # --- 1D由来 (Total Atoms) ---
            'Total_Atoms_Raw': avg_atoms,
            'Disk_Avg_CD_Raw': cd_1d_raw,  # 1D由来(補正なしCD)
            'Disk_Avg_CD_Corrected': cd_1d_corrected,  # 1D由来補正ありCD
            'Disk_Avg_Error_Corrected': cd_err_corrected,
            # --- 2D由来 (Regions) ---
            'Disk_Avg_Region_Raw': reg_means['Region_Avg'],  # 2次元由来(補正なしCD)
            'North': reg_means['North'],
            'South': reg_means['South'],
            'Equator': reg_means['Equator'],
            'Dawn': reg_means['Dawn'],
            'SSP': reg_means['SS'],
            'Dusk': reg_means['Dusk'],
            'Correction_Factor': pa_correction_factor
        }
        yearly_results[year].append(result_row)
        taa_for_plot[year].append(TAA)
        cf_for_plot[year].append(pa_correction_factor)
        print(
            f"  > [{day}] {obs_type:<4}, TAA={TAA:.1f}°, 補正係数={pa_correction_factor:.3f} -> Disk_Avg_CD_Corrected={cd_1d_corrected:.3e}")

    # --- 3. 年ごとにCSVを読み込み・結合・上書き保存 ---
    for year, rows in yearly_results.items():
        if not rows: continue

        csv_filename = out_base / f"All_Results_Summary_{year}.csv"
        new_df = pd.DataFrame(rows)

        if csv_filename.exists() and csv_filename.stat().st_size > 0:
            try:
                # 既存のCSVがある場合は読み込んで結合する
                old_df = pd.read_csv(csv_filename)
                old_df['Date'] = old_df['Date'].astype(str)
                new_df['Date'] = new_df['Date'].astype(str)

                # 結合
                combined_df = pd.concat([old_df, new_df], ignore_index=True)
                combined_df.drop_duplicates(subset=['Date'], keep='last', inplace=True)
            except Exception as e:
                print(f"  > [警告] 既存のCSV読み込みでエラーが発生しました。新規ファイルとして上書きします: {e}")
                combined_df = new_df
        else:
            combined_df = new_df

        # 日付順に並べ替え
        combined_df.sort_values(by='Date', inplace=True)

        # 保存
        combined_df.to_csv(csv_filename, index=False, float_format="%.5e")
        print(
            f"\n  > 【更新完了】 {year}年の統合CSV: {csv_filename.name} (合計 {len(combined_df)} 日分のデータが収録されています)")

        # プロット作成
        plot_name = agg_conf.get("plot_filename_template", f"Correction_Factor_vs_TAA_{year}.png").replace("{month}",
                                                                                                           year)
        plot_filename = out_base / plot_name
        try:
            plt.figure(figsize=(10, 6))
            plt.scatter(taa_for_plot[year], cf_for_plot[year], c='blue', marker='o',
                        label='Calculated Correction Factor')
            plt.title(f'Correction Factor vs. TAA ({year})')
            plt.xlabel('TAA (deg)')
            plt.ylabel('PA Correction Factor')
            plt.grid(True)
            plt.legend()
            plt.savefig(plot_filename)
            plt.close()
            print(f"  > プロット画像を保存しました: {plot_filename.name}")
        except Exception as e:
            print(f"  > 警告: プロット作成中にエラーが発生しました: {e}")


if __name__ == '__main__':
    print("このスクリプトは main.py からモジュールとして呼び出してください。")