from skyfield.api import Topos, load
from datetime import datetime, timedelta, timezone

# JSTタイムゾーンオブジェクト
JST = timezone(timedelta(hours=9))
# UTCタイムゾーンオブジェクトを追加
UTC = timezone.utc


def post_process_to_segments(visible_minutes_map, step_minutes=1):
    """
    日付ごとに記録された観測可能な時刻のリストから、
    連続する時間帯の文字列リストを生成する。
    出力にはJSTとUTCの両方を含む。
    """
    segments_by_date = {}
    for date_str, times_jst in visible_minutes_map.items():
        if not times_jst:
            segments_by_date[date_str] = []
            continue

        times_jst.sort()

        date_segments = []
        current_segment_start = times_jst[0]
        current_segment_end = times_jst[0]

        for i in range(1, len(times_jst)):
            if (times_jst[i] - current_segment_end) == timedelta(minutes=step_minutes):
                current_segment_end = times_jst[i]
            else:
                # --- ▼▼▼ ここから表示形式の変更 ▼▼▼ ---
                # JSTの時刻をUTCに変換
                start_utc = current_segment_start.astimezone(UTC)
                end_utc = current_segment_end.astimezone(UTC)

                # JSTとUTCを併記した文字列を作成
                segment_str = (
                    f"{current_segment_start.strftime('%H:%M')} - {current_segment_end.strftime('%H:%M')} (JST)  |  "
                    f"{start_utc.strftime('%H:%M')} - {end_utc.strftime('%H:%M')} (UTC)")
                date_segments.append(segment_str)
                # --- ▲▲▲ ここまで表示形式の変更 ▲▲▲ ---

                current_segment_start = times_jst[i]
                current_segment_end = times_jst[i]

        # --- ▼▼▼ 最後の区間も同様に変更 ▼▼▼ ---
        start_utc = current_segment_start.astimezone(UTC)
        end_utc = current_segment_end.astimezone(UTC)
        segment_str = (f"{current_segment_start.strftime('%H:%M')} - {current_segment_end.strftime('%H:%M')} (JST)  |  "
                       f"{start_utc.strftime('%H:%M')} - {end_utc.strftime('%H:%M')} (UTC)")
        date_segments.append(segment_str)
        # --- ▲▲▲ 最後の区間も同様に変更 ▲▲▲ ---

        segments_by_date[date_str] = date_segments

    return segments_by_date


def calculate_mercury_visibility_detailed(start_date_str, end_date_str, observer_location, step_minutes=1):
    """
    指定された期間において、特定の条件下で水星が観測可能な具体的な時間帯を計算する。
    天体暦ファイル 'de442.bsp' をスクリプトと同じディレクトリから読み込むことを想定。
    (この関数のロジックは変更ありません)
    """
    try:
        start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
        if end_dt < start_dt:
            return "終了日は開始日以降である必要があります。"
    except ValueError:
        return "日付の形式が正しくありません。YYYY-MM-DD形式で入力してください。"

    try:
        eph = load('de442.bsp')
    except Exception as e:
        return (f"天体暦ファイル 'de442.bsp' のロードに失敗しました: {e}\n"
                "スクリプトと同じディレクトリに 'de442.bsp' ファイルが配置されているか、"
                "ファイルが破損していないか確認してください。")

    sun = eph['sun']
    mercury = eph['mercury']
    earth = eph['earth']
    ts = load.timescale()

    all_visible_minutes_by_date_jst = {}

    current_loop_time_utc = datetime(start_dt.year, start_dt.month, start_dt.day, 0, 0, 0, tzinfo=UTC)
    end_loop_time_utc = datetime(end_dt.year, end_dt.month, end_dt.day, 23, 59, 59, tzinfo=UTC)

    observer = earth + observer_location

    # __name__ チェックはテスト用だったので、ここでは print をそのまま表示します
    print("計算中 (詳細)... (期間が長い場合、数分～数十分かかることがあります)")
    processed_steps = 0
    total_steps = (end_loop_time_utc - current_loop_time_utc).total_seconds() / (step_minutes * 60)

    while current_loop_time_utc <= end_loop_time_utc:
        t = ts.utc(current_loop_time_utc)

        sun_apparent = observer.at(t).observe(sun).apparent()
        mercury_apparent = observer.at(t).observe(mercury).apparent()

        sun_alt_deg = sun_apparent.altaz()[0].degrees
        mercury_alt_deg = mercury_apparent.altaz()[0].degrees
        sot_deg = mercury_apparent.separation_from(sun_apparent).degrees

        is_visible_now = (sun_alt_deg <= -1.5 and
                          sot_deg >= 15 and
                          mercury_alt_deg >= 10)

        if is_visible_now:
            current_time_jst = current_loop_time_utc.astimezone(JST)
            date_str = current_time_jst.strftime("%Y-%m-%d")
            if date_str not in all_visible_minutes_by_date_jst:
                all_visible_minutes_by_date_jst[date_str] = []
            all_visible_minutes_by_date_jst[date_str].append(current_time_jst)

        processed_steps += 1
        if processed_steps % 1440 == 0:
            print(
                f"  進捗: 約 {processed_steps / total_steps * 100:.1f}% 完了 ({current_loop_time_utc.strftime('%Y-%m-%d %H:%M')} UTCまで処理)")

        current_loop_time_utc += timedelta(minutes=step_minutes)

    print("  観測可能な時刻のリストアップ完了。時間帯に整形します...")
    return post_process_to_segments(all_visible_minutes_by_date_jst, step_minutes)


if __name__ == '__main__':
    print("--- 水星観測可能時間帯チェッカー (詳細版) ---")

    default_lat, default_lon = 20.7083119, 203.742  # ハレアカラ
    try:
        lat_input = input(f"観測地の緯度 (デフォルトはハレアカラ {default_lat}): ")
        obs_lat = float(lat_input) if lat_input else default_lat
    except ValueError:
        print(f"緯度の入力が無効です。デフォルト値 {default_lat} を使用します。")
        obs_lat = default_lat

    try:
        lon_input = input(f"観測地の経度 (デフォルトはハレアカラ {default_lon}): ")
        obs_lon = float(lon_input) if lon_input else default_lon
    except ValueError:
        print(f"経度の入力が無効です。デフォルト値 {default_lon} を使用します。")
        obs_lon = default_lon

    observer_location = Topos(latitude_degrees=obs_lat, longitude_degrees=obs_lon)

    while True:
        start_input = input("観測開始日 (YYYY-MM-DD形式, 例: 2025-06-01): ")
        if not start_input:
            start_input = "2025-06-01"
            print(f"デフォルトの開始日 {start_input} を使用します。")
        try:
            datetime.strptime(start_input, "%Y-%m-%d")
            break
        except ValueError:
            print("日付の形式が正しくありません。再入力してください。")

    while True:
        end_input = input(f"観測終了日 (YYYY-MM-DD形式, {start_input} 以降, 例: 2025-06-10): ")
        if not end_input:
            try:
                default_end_dt = datetime.strptime(start_input, "%Y-%m-%d") + timedelta(days=9)
                end_input = default_end_dt.strftime("%Y-%m-%d")
                print(f"デフォルトの終了日 {end_input} を使用します。")
            except ValueError:
                end_input = "2025-06-10"
                print(f"デフォルトの終了日 {end_input} を使用します。")
        try:
            if datetime.strptime(end_input, "%Y-%m-%d") < datetime.strptime(start_input, "%Y-%m-%d"):
                print("終了日は開始日以降である必要があります。")
            else:
                break
        except ValueError:
            print("日付の形式が正しくありません。再入力してください。")

    print(f"\n観測地: 緯度={obs_lat:.4f}, 経度={obs_lon:.4f}")
    print(f"使用天体暦: de442.bsp (スクリプトと同じディレクトリにあることを想定)")
    print(f"期間: {start_input} から {end_input} まで")
    print("条件:")
    print("  1. 太陽の高度 <= -1.5 度")
    print("  2. 太陽と水星の離角 (SOT) >= 15 度")
    print("  3. 水星の高度 >= 10 度")

    results = calculate_mercury_visibility_detailed(start_input, end_input, observer_location, step_minutes=1)

    if isinstance(results, str):
        print(f"\nエラー: {results}")
    elif not results:
        print("\n指定された期間と条件で水星が観測可能な時間帯は見つかりませんでした。")
    else:
        # --- ▼▼▼ 表示ヘッダーを変更 ▼▼▼ ---
        print("\n--- 水星の観測可能な時間帯 (日付はJST) ---")
        sorted_dates = sorted(results.keys())
        found_any_slot = False
        for date_str in sorted_dates:
            segments = results[date_str]
            if segments:
                found_any_slot = True
                print(f"  {date_str}:")
                for segment in segments:
                    # segment文字列はすにJST/UTC併記になっているのでそのまま出力
                    print(f"    {segment}")
        if not found_any_slot:
            print("指定された期間と条件で水星が観測可能な時間帯は見つかりませんでした。")

    print("\n--- 計算終了 ---")