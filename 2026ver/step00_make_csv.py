import pandas as pd
from pathlib import Path
import numpy as np
import sys
import re


def get_obj_type(filename):
    """ファイル名からオブジェクトタイプを判定する関数"""
    name_lower = filename.lower()
    if "me" in name_lower or "mercury" in name_lower:
        return "MERCURY"
    elif "sky" in name_lower:
        return "SKY"
    elif "led" in name_lower:
        return "LED"
    elif "hlg" in name_lower or "halo" in name_lower:
        return "HLG"
    #elif "dk" in name_lower or "dark" in name_lower:
    #    return "DARK"
    #elif "ne" in name_lower:
    #    return "NE"
    #elif "ve" in name_lower:
    #    return "VENUS"
    #elif "io" in name_lower:
    #    return "IO"
    return "UNKNOWN"


def search_neighboring_dates(data_base_dir, target_date, missing_types):
    """
    指定されたタイプ(missing_types)を含むファイルを、近隣の日付フォルダから探す。
    """
    # dataディレクトリ内の全ての日付フォルダを取得 (YYYYMMDD形式と仮定)
    data_root = data_base_dir / "data"
    all_dates = []
    for d in data_root.iterdir():
        if d.is_dir() and re.match(r'^\d{8}$', d.name):
            all_dates.append(d.name)

    all_dates.sort()

    if target_date not in all_dates:
        # 万が一ターゲット自体がリストにない場合（レアケース）
        return {}

    target_idx = all_dates.index(target_date)
    found_files = {t: [] for t in missing_types}
    filled_types = set()

    # 探索範囲を広げながら探す (近い日付優先: -1日, +1日, -2日, +2日...)
    # 最大前後30日くらい探せば十分
    max_offset = len(all_dates)

    print(f"  > [補完検索] 不足データ {missing_types} を近隣日程から検索します...")

    for offset in range(1, max_offset):
        check_indices = []
        if target_idx - offset >= 0: check_indices.append(target_idx - offset)
        if target_idx + offset < len(all_dates): check_indices.append(target_idx + offset)

        for idx in check_indices:
            neighbor_date = all_dates[idx]
            neighbor_dir = data_root / neighbor_date

            # そのフォルダ内のfitsを探す
            candidates = list(neighbor_dir.glob("*.fits"))

            # まだ見つかっていないタイプを探す
            for t in missing_types:
                if t in filled_types: continue  # 既に確保済みならスキップ

                # そのタイプに合致するファイルがあるか？
                # (nhpなどは除外)
                matches = [f for f in candidates
                           if get_obj_type(f.name) == t
                           and not f.name.endswith(("_nhp_py.fits", ".wc.fits", "_tr.fits"))]

                if matches:
                    print(f"    -> 発見: {t} を {neighbor_date} から {len(matches)} ファイル借用します。")
                    found_files[t].extend(matches)
                    filled_types.add(t)

        # 全ての不足タイプが埋まったら終了
        if len(filled_types) == len(missing_types):
            break

    return found_files


def run(run_info, config):
    target_date = run_info["date"]
    csv_path = run_info["csv_path"]

    force_rerun = config.get("pipeline", {}).get("force_rerun_csv", False)

    if csv_path.exists() and not force_rerun:
        print(f"  > [Step0] 処理済みのためスキップ: {csv_path.name}")
        return True

    dirs_conf = config.get("directories", {})
    data_base_dir = Path(dirs_conf.get("data_base_dir", "C:/Users/hanac/University/Senior/Mercury"))
    raw_data_dir = data_base_dir / "data" / target_date

    print(f"\n--- {target_date} のCSVファイル自動生成 (Step0) を開始します ---")

    if not raw_data_dir.exists():
        print(f"  > エラー: 生データのフォルダが見つかりません: {raw_data_dir}")
        return False

    # 除外リスト
    exclude_suffix = ("_nhp_py.fits", ".wc.fits", "_tr.fits", ".solar_sub.fits", ".sub.fits")

    # 当日のファイルをリスト化
    fits_files = list(raw_data_dir.glob("*.fits"))
    fits_files = [f for f in fits_files if not f.name.endswith(exclude_suffix)]

    if not fits_files:
        print("  > エラー: 当日のフォルダに FITSファイルが見つかりませんでした。")
        # 当日のファイルがゼロでも、借りてくれば動くかもしれないので続行は可能だが、
        # 通常は水星データもないはずなのでここで止めるのが無難
        return False

    # 定義
    columns = [
        "fits", "Type", "DATE-OBS", "apparent_diameter_arcsec",
        "mercury_sun_distance_au", "mercury_sun_radial_velocity_km_s",
        "mercury_earth_radial_velocity_km_s", "phase_angle_deg",
        "true_anomaly_deg", "ecliptic_longitude_deg",
        "ecliptic_latitude_deg", "terminator_side", "terminator_lon_delta_deg"
    ]

    data_list = []

    # ---------------------------------------------------------
    # 1. 当日のファイルを処理
    # ---------------------------------------------------------
    present_types = set()
    unknown_count = 0

    for f in fits_files:
        obj_type = get_obj_type(f.name)

        if obj_type == "UNKNOWN":
            unknown_count += 1
            continue

        present_types.add(obj_type)

        abs_path = str(f.resolve()).replace('\\', '/')
        row_data = {col: np.nan for col in columns}
        row_data["fits"] = abs_path
        row_data["Type"] = obj_type
        data_list.append(row_data)

    # ---------------------------------------------------------
    # 2. 不足データの確認と補完 (Borrowing Logic)
    # ---------------------------------------------------------
    missing_types = []

    # SKYがない場合
    if "SKY" not in present_types:
        missing_types.append("SKY")

    # HLGもLEDもない場合 (FLAT用)
    if "HLG" not in present_types and "LED" not in present_types:
        # HLGを優先して探すが、なければLEDでもいいので、とりあえずHLGを探させる
        # (search_neighboring_datesを少し改良して両方探す手もあるが、ここではHLG優先)
        missing_types.append("HLG")

    if missing_types:
        borrowed_map = search_neighboring_dates(data_base_dir, target_date, missing_types)

        # HLGが見つからず、かつ元々LEDもなかった場合、LEDを探しに行く
        if "HLG" in missing_types and not borrowed_map.get("HLG"):
            print("    -> HLGが見つかりませんでした。代替として LED を探します。")
            borrowed_led = search_neighboring_dates(data_base_dir, target_date, ["LED"])
            borrowed_map.update(borrowed_led)

        # 借りてきたファイルをリストに追加
        for b_type, path_list in borrowed_map.items():
            for p in path_list:
                abs_path = str(p.resolve()).replace('\\', '/')
                row_data = {col: np.nan for col in columns}
                row_data["fits"] = abs_path
                row_data["Type"] = b_type  # HLG or LED or SKY
                data_list.append(row_data)

                print(f"    + 追加 (借用): {p.name} as {b_type}")

    # ---------------------------------------------------------
    # 3. 保存
    # ---------------------------------------------------------
    if not data_list:
        print("  > エラー: 有効なデータが見つかりませんでした。")
        return False

    df = pd.DataFrame(data_list, columns=columns)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    print(f"  > 成功: {len(df)} 件のファイルをCSVに書き込みました。")
    if unknown_count > 0:
        print(f"  > (除外された不明ファイル: {unknown_count} 件)")

    # 最終チェック
    final_types = set(df["Type"].unique())
    has_sky = "SKY" in final_types
    has_flat = ("HLG" in final_types) or ("LED" in final_types)

    if not has_sky:
        print("  [警告] SKYデータが当日分にも近隣日程にも見つかりませんでした。後のStepでエラーになる可能性があります。")
    if not has_flat:
        print("  [警告] Flatデータ(HLG/LED)が見つかりませんでした。後のStepでエラーになる可能性があります。")

    print(f"  > 保存先: {csv_path.name}")
    return True


if __name__ == "__main__":
    print("このスクリプトは main.py からモジュールとして呼び出してください。")