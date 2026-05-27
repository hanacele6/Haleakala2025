import yaml
from pathlib import Path

# 動かしたいステップをインポート（例: Step 03）
import step03_fiber_trace
# import step04_extract_spectra
# import step06_wavelength_calibration

def run_single_step():
    # 1. 疑似的な run_info を手動で定義する
    target_date = "20250811"
    project_base = Path("C:/Users/hanac/univ/Mercury/Haleakala2025")
    
    run_info = {
        "date": target_date,
        "csv_path": project_base / "2026ver/csvs" / f"mcparams{target_date}.csv",
        "output_dir": project_base / "output" / target_date
    }

    # 2. config.yaml を読み込む
    config_path = project_base / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 3. 指定したステップだけを単独実行！
    print(f"--- {target_date} のデータで単独テストを開始 ---")
    
    # ここを書き換えて好きなステップを動かす
    step03_fiber_trace.run(run_info, config)
    
    print("--- テスト完了 ---")

if __name__ == "__main__":
    run_single_step()