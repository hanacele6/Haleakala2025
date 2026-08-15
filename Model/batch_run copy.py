import subprocess
import requests
import time

WEBHOOK_URL = "https://discord.com/api/webhooks/1524291619505963163/6fNzGPjFmXULPn6K13GKqZ1F84XjzgvR1DcOLHTvoqhefxrrAQIWH_wmB1u0Nfpm_wXt"

# 実行するモデルスクリプト
MODEL_SCRIPT = "mkNaColumnDensity9_9_3.py"


def send_discord(msg):
    if WEBHOOK_URL:
        try:
            requests.post(WEBHOOK_URL, json={"content": msg})
        except Exception as e:
            print(f"Discord通知に失敗: {e}")


# ==============================================================================
# ジョブの書き方
# ==============================================================================
# 物理パラメータ:
#   u_model, u_mu, u_sigma, q_psd_base, diff_ea, t1au
#
# 数値実験 (省略すると 'on'):
#   srp / td / psd  … 'on' | 'off' | 'final'
#     'on'    : 常に有効
#     'off'   : 最初から最後まで無効。スピンアップから切るので新しい定常に達する。
#               → 定常同士で比較できる。ドリフトの心配がない。
#     'final' : 記録フェーズ(最終年)のみ無効。
#               → 定常なパターンに対する即時効果を見る。
#                 ただし系は定常でなくなるので、単調ドリフトが乗る点に注意。
#
# 記録範囲 (省略すると 'final'):
#   save_phase … 'final' | 'all'
#     'final' : 最終年のみスナップショットとホップ輸送を保存 (容量小)
#     'all'   : スピンアップ含め全期間を保存 (形成過程を追える。容量大)
#
# 年数 (省略すると spinup=14, total=1):
#   spinup_years, total_years
#
# ファイル名には条件が自動で入る:
#   srp='off'   → _noSRP
#   srp='final' → _noSRPFin
#   save='all'  → _SaveAll
# ==============================================================================

queue = [
    # ------------------------------------------------------------------
    # 基準となる標準run (Q強 / Q弱)
    # ------------------------------------------------------------------
    {"u_model": "gaussian_random", "u_mu": 1.85, "u_sigma": 0.25,
     "q_psd_base": 2.0, "diff_ea": 0.8, "t1au": 190000.0,
     "save_phase": "all"},

    {"u_model": "gaussian_random", "u_mu": 1.85, "u_sigma": 0.25,
     "q_psd_base": 0.27, "diff_ea": 0.8, "t1au": 190000.0,
     "save_phase": "all"},

    # ------------------------------------------------------------------
    # SRPを最初から切って定常に到達させる
    #   最終年だけ切った場合と違い、ドリフトが乗らないので
    #   標準runと定常同士で直接比較できる。
    # ------------------------------------------------------------------
    {"u_model": "gaussian_random", "u_mu": 1.85, "u_sigma": 0.25,
     "q_psd_base": 2.0, "diff_ea": 0.8, "t1au": 190000.0,
     "srp": "off"},

    {"u_model": "gaussian_random", "u_mu": 1.85, "u_sigma": 0.25,
     "q_psd_base": 0.27, "diff_ea": 0.8, "t1au": 190000.0,
     "srp": "off"},

    # ------------------------------------------------------------------
    # 以下は必要に応じてコメントを外す
    # ------------------------------------------------------------------

    # PSDを最初から切る (TD単独のピーク位置を定常状態で見る)
    # {"u_model": "gaussian_random", "u_mu": 1.85, "u_sigma": 0.25,
    #  "q_psd_base": 2.0, "diff_ea": 0.8, "t1au": 190000.0,
    #  "psd": "off"},

    # TDを最初から切る (PSD単独)
    # {"u_model": "gaussian_random", "u_mu": 1.85, "u_sigma": 0.25,
    #  "q_psd_base": 2.0, "diff_ea": 0.8, "t1au": 190000.0,
    #  "td": "off"},

    # 最終年だけSRPを切る (定常パターンに対する即時効果)
    # {"u_model": "gaussian_random", "u_mu": 1.85, "u_sigma": 0.25,
    #  "q_psd_base": 2.0, "diff_ea": 0.8, "t1au": 190000.0,
    #  "srp": "final"},

    # 全期間を記録する (高密度領域の形成過程を追う。容量が大きい)
    # {"u_model": "gaussian_random", "u_mu": 1.85, "u_sigma": 0.25,
    #  "q_psd_base": 2.0, "diff_ea": 0.8, "t1au": 190000.0,
    #  "save_phase": "all"},
]


# ==============================================================================
# 実行
# ==============================================================================
def build_cmd(params):
    """パラメータ辞書からコマンドを組み立てる。未指定の項目は既定値に任せる。"""
    cmd = ["python", MODEL_SCRIPT]
    # 物理パラメータ (必須)
    for key in ("u_model", "u_mu", "u_sigma", "q_psd_base", "diff_ea", "t1au"):
        if key in params:
            cmd += [f"--{key}", str(params[key])]
    # 数値実験と記録範囲 (任意)
    for key in ("srp", "td", "psd", "save_phase", "spinup_years", "total_years"):
        if key in params:
            cmd += [f"--{key}", str(params[key])]
    return cmd


def describe(params):
    """ログ用の短い説明"""
    base = (f"Mu={params.get('u_mu')}, Sig={params.get('u_sigma')}, "
            f"Q={params.get('q_psd_base')}e-20, Ea={params.get('diff_ea')}eV")
    exp = []
    for key in ("srp", "td", "psd"):
        v = params.get(key, "on")
        if v != "on":
            exp.append(f"{key.upper()}={v}")
    if params.get("save_phase", "final") != "final":
        exp.append(f"save={params['save_phase']}")
    return base + ("  [" + ", ".join(exp) + "]" if exp else "  [標準]")


send_discord(f"🚀 **バッチ処理を開始します！** (全 {len(queue)} ジョブ)")

for i, params in enumerate(queue):
    info = describe(params)
    send_discord(f"⏳ **[{i+1}/{len(queue)}] 実行開始:** `{info}`")
    print(f"\n{'='*70}")
    print(f"[{i+1}/{len(queue)}] 実行開始: {info}")
    cmd = build_cmd(params)
    print("  " + " ".join(cmd))
    print('='*70)

    start_time = time.time()
    result = subprocess.run(cmd)
    elapsed_hours = (time.time() - start_time) / 3600

    if result.returncode == 0:
        send_discord(f"✅ **[{i+1}/{len(queue)}] 完了！** (所要時間: {elapsed_hours:.2f}時間)")
    else:
        send_discord(f"❌ **[{i+1}/{len(queue)}] エラー終了しました...**\n`{info}`")
        break

send_discord("🎉 **すべての計算が完了しました！**")