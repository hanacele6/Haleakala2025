import yaml
from pathlib import Path
import sys
import shutil
import pandas as pd
import numpy as np
import re
import traceback
from datetime import datetime

# 全ステップのモジュールをインポート
import step00_make_csv
import step01_update_spice
import step02_remove_hotpixels
import step03_fiber_trace
import step04_extract_spectra
import step05_combine_spectra
import step06_wavelength_calibration
import step07_resample_and_flat
import step08_subtract_background
import step09_hapke_model
import step10_solar_subtraction
import step11_calculate_gfactor
import step12_calc_column_density
import step13_final_summary
import step14_aggregate_daily


def organize_files_only(run_info):
    """
    Step13終了後に呼び出される、ファイル整理専用関数。
    """
    output_dir = run_info["output_dir"]

    fits_dir = output_dir / "1_fits"
    spec_dir = output_dir / "2_spectra"
    plot_dir = output_dir / "3_plots"
    for d in [fits_dir, spec_dir, plot_dir]:
        d.mkdir(exist_ok=True)

    for sub in ["plots", "fiber_plots", "wcal_plots"]:
        src_dir = output_dir / sub
        if src_dir.exists() and src_dir.is_dir():
            for f in src_dir.rglob("*.png"):
                try:
                    shutil.move(str(f), str(plot_dir / f.name))
                except Exception:
                    pass
            shutil.rmtree(src_dir, ignore_errors=True)

    for f in output_dir.glob("*"):
        if f.is_file():
            ext = f.suffix.lower()
            name = f.name
            if "Summary" in name or "Na_atoms" in name or "Quality_Report" in name:
                continue
            if ext == ".fits":
                shutil.move(str(f), str(fits_dir / name))
            elif ext == ".dat" or ext == ".exos":
                shutil.move(str(f), str(spec_dir / name))
            elif ext == ".png":
                shutil.move(str(f), str(plot_dir / name))
            elif ext == ".txt":
                shutil.move(str(f), str(spec_dir / name))

    print(f"  > ファイル整理完了。")


# ====================================================================

def main():
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / "config.yaml"

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"エラー: 以下のパスに config.yaml が見つかりません。\n{config_path}")
        return

    dirs_conf = config.get("directories", {})
    data_base_dir = Path(dirs_conf.get("data_base_dir", "C:/Users/hanac/University/Senior/Mercury"))
    project_base_dir = Path(dirs_conf.get("project_base_dir", "C:/Users/hanac/University/Senior/Mercury/Haleakala2025"))

    csv_dir = project_base_dir / dirs_conf.get("csv_dir_name", "2026ver/csvs")
    out_base = project_base_dir / dirs_conf.get("output_dir_name", "output")
    out_base.mkdir(parents=True, exist_ok=True)

    # --- 実行対象の日付リストを動的に構築 ---
    target_run_conf = config.get("target_run", {})
    target_dates = []

    if "dates" in target_run_conf and target_run_conf["dates"]:
        # リストで直接指定されている場合 (優先)
        target_dates = [str(d) for d in target_run_conf["dates"]]
    elif "start_date" in target_run_conf and "end_date" in target_run_conf:
        # 期間指定の場合
        start_d = str(target_run_conf["start_date"])
        end_d = str(target_run_conf["end_date"])
        print(f"\n[期間検索モード] {start_d} 〜 {end_d} の生データを検索します...")

        # 生データがあるディレクトリを検索対象にする
        target_data_dir = data_base_dir / "data"
        if target_data_dir.exists():
            for item in target_data_dir.iterdir():
                # フォルダ名やファイル名から 8桁の数字(YYYYMMDD) を抽出する
                match = re.search(r'(\d{8})', item.name)
                if match:
                    date_str = match.group(1)
                    # 抽出した日付が期間内に収まっているか判定
                    if start_d <= date_str <= end_d:
                        if date_str not in target_dates:  # 重複登録を防ぐ
                            target_dates.append(date_str)
        else:
            print(f"  > [警告] データディレクトリが見つかりません: {target_data_dir}")

        target_dates.sort()
    else:
        # 古い config へのフォールバック
        target_dates = [str(target_run_conf.get("date", "20251021"))]

    if not target_dates:
        print("実行対象の日付が見つかりませんでした。データディレクトリやconfigの設定を確認してください。")
        return

    print(f"  > 実行予定日 (計 {len(target_dates)}日): {', '.join(target_dates)}")

    completed_dates = []
    execution_report = {}  # 実行結果レポート用辞書

    # 複数日の予約を順番に実行するループ
    for target_date in target_dates:
        print(f"\n{'=' * 60}")
        print(f"==== 水星観測データ解析開始: 観測日 {target_date} ====")
        print(f"{'=' * 60}")

        csv_path = csv_dir / f"mcparams{target_date}.csv"
        output_dir = out_base / target_date

        run_info = {
            "date": target_date,
            "csv_path": csv_path,
            "output_dir": output_dir
        }

        try:
            # ----------------------------------------------------------
            # Step 0 ～ 2 (Pre-processing) の制御
            # ----------------------------------------------------------
            # configから設定を読み込む（デフォルトは False = 実行する）
            do_skip_pre = config.get('pipeline', {}).get('skip_preprocessing', False)

            step00_make_csv.run(run_info, config)
            step01_update_spice.run(run_info, config)

            if do_skip_pre:
                print(f"  [Info] 'skip_preprocessing' is True. Skipping Step 2.")
            else:
                # スキップしない場合のみ実行
                step02_remove_hotpixels.run(run_info, config)

            step03_fiber_trace.run(run_info, config)
            step04_extract_spectra.run(run_info, config)
            # ==========================================================
            # nhpファイル（画像）の掃除
            # Step 4でスペクトル抽出が終われば、画像データ自体はもう不要です。
            # Rawデータは残っているので、必要ならStep 2からやり直せます。
            # ==========================================================
            print(f"  [Cleanup] Deleting used nhp images to save space...")
            for f in output_dir.glob("*_nhp_py.fits"):
                try:
                    f.unlink()  # 削除
                except Exception:
                    pass
            step05_combine_spectra.run(run_info, config)
            step06_wavelength_calibration.run(run_info, config)
            step07_resample_and_flat.run(run_info, config)
            step08_subtract_background.run(run_info, config)
            step09_hapke_model.run(run_info, config)
            step10_solar_subtraction.run(run_info, config)
            step11_calculate_gfactor.run(run_info, config)
            step12_calc_column_density.run(run_info, config)
            step13_final_summary.run(run_info, config)
            organize_files_only(run_info)

            # 成功した日付を記録
            completed_dates.append(target_date)
            execution_report[target_date] = "Success"

        except Exception as e:
            # エラー発生時はログに記録して次の日付へ
            print(f"\n[!] 処理中に致命的なエラーが発生しました ({target_date}): {e}")
            err_msg = traceback.format_exc().strip().split('\n')[-1]  # エラー内容の最後の行を抽出
            execution_report[target_date] = f"Error: {str(e)} / {err_msg}"
            continue

    # --- Step 14: 全スケジュール完了後の統合集計 ---
    if completed_dates:
        print(f"\n==== 予約完了: 正常終了した全 {len(completed_dates)} 日分のデータを集計 (Step14) します ====")
        step14_aggregate_daily.run(base_dir=project_base_dir, out_base=out_base, csv_dir=csv_dir,
                                   processed_dates=completed_dates, config=config)
    else:
        print("\n  > 警告: 正常に完了したデータが見つからなかったため、集計はスキップされました。")

    # --- 最終実行レポートの出力 ---
    now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = out_base / f"Pipeline_Execution_Report_{now_str}.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(" Mercury Pipeline Execution Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Executed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total processed: {len(target_dates)} days\n")
        f.write(f"Successful: {len(completed_dates)} days\n")
        f.write(f"Failed: {len(target_dates) - len(completed_dates)} days\n")
        f.write("-" * 60 + "\n\n")

        for d in target_dates:
            status = execution_report.get(d, "Unknown Status")
            # 成功と失敗で見やすくマークをつける
            mark = "[ OK ]" if status == "Success" else "[FAIL]"
            f.write(f"{mark} {d} : {status}\n")

    print(f"\n==== パイプラインの全工程が完了しました ====")
    print(f"  > 実行レポートを保存しました: {report_path.name}")


if __name__ == "__main__":
    main()