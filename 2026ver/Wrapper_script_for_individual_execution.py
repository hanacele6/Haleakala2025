import yaml
from pathlib import Path

# --- 必須の前処理モジュール (Step 0~2) ---
import step00_make_csv
import step01_update_spice
import step02_remove_hotpixels

# --- テストしたいモジュールをインポート ---
import step03_fiber_trace
# import step04_extract_spectra
# import step06_wavelength_calibration


def run_single_step():
    # 1. 自身のディレクトリを基準に config.yaml を読み込む (main.py と同じ挙動)
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / "config.yaml"
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"エラー: 以下のパスに config.yaml が見つかりません。\n{config_path}")
        return

    # 2. config の設定からパスを動的に構築する
    dirs_conf = config.get("directories", {})
    project_base_dir = Path(dirs_conf.get("project_base_dir", "C:/Users/hanac/univ/Mercury/Haleakala2025"))
    csv_dir_name = dirs_conf.get("csv_dir_name", "2026ver/csvs")
    output_dir_name = dirs_conf.get("output_dir_name", "output")
    
    # テストしたい日付を指定
    target_date = "20250811"  
    
    # パスの組み立て
    csv_path = project_base_dir / csv_dir_name / f"mcparams{target_date}.csv"
    output_dir = project_base_dir / output_dir_name / target_date

    run_info = {
        "date": target_date,
        "csv_path": csv_path,
        "output_dir": output_dir
    }

    print(f"--- {target_date} のデータで単独テストを開始 ---")

    # 3. 前処理 (Step 0~2) の自動実行・スキップ判定
    print("\n[準備フェーズ: Step 0~2 の確認]")
    step00_make_csv.run(run_info, config)
    step01_update_spice.run(run_info, config)
    
    if not config.get('pipeline', {}).get('skip_preprocessing', False):
        step02_remove_hotpixels.run(run_info, config)

    # 4. 個別ステップの実行
    print("\n[テストフェーズ: 指定ステップの実行]")
    # ↓↓↓ ここを書き換えて好きなステップを動かす ↓↓↓
    step03_fiber_trace.run(run_info, config)
    
    print("\n--- テスト完了 ---")

if __name__ == "__main__":
    run_single_step()