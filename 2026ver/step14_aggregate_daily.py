import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def run(base_dir, out_base, csv_dir, processed_dates, config):
    """
    パイプラインの最後に呼び出される、複数日の集計・PA補正・年次CSV更新・プロット実行関数 (1D専用)
    """
    agg_conf = config.get("aggregation", {})
    correction_file = Path(agg_conf.get("pa_correction_file", ""))

    print("\n" + "=" * 60)
    print(f"--- 1D最終結果のCSV統合 (Yearly Smart Update) を開始します ---")
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

    yearly_results = {}
    taa_for_plot = {}
    cf_for_plot = {}

    # --- 2. 各日付のデータをループ処理して抽出 ---
    for day in processed_dates:
        year = str(day)[:4]

        if year not in yearly_results:
            yearly_results[year] = []
            taa_for_plot[year] = []
            cf_for_plot[year] = []

        day_dir = out_base / day
        summary_file = day_dir / f'Final_Summary_{day}.txt'
        csv_file_path = csv_dir / f"mcparams{day}.csv"

        try:
            PA, TAA, avg_atoms, integrated_err = np.loadtxt(summary_file)

            if csv_file_path.exists():
                df = pd.read_csv(csv_file_path)
                obs_row = df[df['Type'].str.strip() == 'MERCURY'].iloc[0]

                Rms = obs_row['mercury_sun_distance_au']
                beta = obs_row['ecliptic_latitude_deg']
                elon = obs_row['ecliptic_longitude_deg']
                obs_type = str(obs_row.get('terminator_side', 'Unknown')).strip()
            else:
                Rms, beta, elon = np.nan, np.nan, np.nan
                obs_type = "Unknown"

        except Exception as e:
            print(f"  > 警告: {day} の基本データ読み込みに失敗しました ({e})。スキップします。")
            continue

        # --- (D) 物理量の補正と計算 ---
        pa_correction_factor = np.polyval(pa_correction_coeffs, PA) if pa_correction_coeffs is not None else 1.0
        h = Rms * np.sin(beta * d2r)
        half_surface_area = 3.74e17

        cd_1d_raw = avg_atoms / half_surface_area
        cd_1d_corrected = (avg_atoms / pa_correction_factor) / half_surface_area
        cd_err_corrected = (integrated_err / pa_correction_factor) / half_surface_area

        result_row = {
            'Date': str(day),
            'Observation_Type': obs_type,
            'TAA_deg': TAA,
            'PhaseAngle_deg': PA,
            'Distance_Rms_au': Rms,
            'Lat_h': h,
            'Ecliptic_Lon': elon,
            'Ecliptic_Lat': beta,
            'Total_Atoms_Raw': avg_atoms,
            'Disk_Avg_CD_Raw': cd_1d_raw,
            'Disk_Avg_CD_Corrected': cd_1d_corrected,
            'Disk_Avg_Error_Corrected': cd_err_corrected,
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

        csv_filename = out_base / f"All_Results_Summary_1D_{year}.csv"  # 名前を1Dと明記
        new_df = pd.DataFrame(rows)

        if csv_filename.exists() and csv_filename.stat().st_size > 0:
            try:
                old_df = pd.read_csv(csv_filename)
                old_df['Date'] = old_df['Date'].astype(str)
                new_df['Date'] = new_df['Date'].astype(str)
                combined_df = pd.concat([old_df, new_df], ignore_index=True)
                combined_df.drop_duplicates(subset=['Date'], keep='last', inplace=True)
            except Exception as e:
                print(f"  > [警告] 既存のCSV読み込みエラー。上書きします: {e}")
                combined_df = new_df
        else:
            combined_df = new_df

        combined_df.sort_values(by='Date', inplace=True)
        combined_df.to_csv(csv_filename, index=False, float_format="%.5e")
        print(f"\n  > 【更新完了】 {year}年の統合CSV: {csv_filename.name}")

        plot_name = agg_conf.get("plot_filename_template", f"Correction_Factor_vs_TAA_1D_{year}.png").replace("{month}",
                                                                                                              year)
        plot_filename = out_base / plot_name
        try:
            plt.figure(figsize=(10, 6))
            plt.scatter(taa_for_plot[year], cf_for_plot[year], c='blue', marker='o',
                        label='Calculated Correction Factor')
            plt.title(f'Correction Factor vs. TAA (1D) ({year})')
            plt.xlabel('TAA (deg)')
            plt.ylabel('PA Correction Factor')
            plt.grid(True)
            plt.legend()
            plt.savefig(plot_filename)
            plt.close()
            print(f"  > プロット画像を保存しました: {plot_filename.name}")
        except Exception as e:
            print(f"  > 警告: プロット作成エラー: {e}")


if __name__ == '__main__':
    print("このスクリプトは main.py からモジュールとして呼び出してください。")