import subprocess
import requests
import time

WEBHOOK_URL = "https://discord.com/api/webhooks/1524291619505963163/6fNzGPjFmXULPn6K13GKqZ1F84XjzgvR1DcOLHTvoqhefxrrAQIWH_wmB1u0Nfpm_wXt"

def send_discord(msg):
    if WEBHOOK_URL:
        requests.post(WEBHOOK_URL, json={"content": msg})

# 回したいパラメータのリスト（好きなだけ増やせます）
queue = [
    # パターン1: 
    {"u_model": "gaussian_random", "u_mu": 1.85, "u_sigma": 0.25, "q_psd_base": 0.27, "diff_ea": 0.8, "t1au": 190000.0},
    
    # パターン2: 
    #{"u_model": "gaussian_random", "u_mu": 1.85, "u_sigma": 0.25, "q_psd_base": 2.0, "diff_ea": 0.8, "t1au": 190000.0},
    
    # パターン3: 
    #{"u_model": "gaussian_random", "u_mu": 1.92, "u_sigma": 0.22, "q_psd_base": 2.0, "diff_ea": 0.8, "t1au": 190000.0},

    # パターン4: 
    #{"u_model": "gaussian_random", "u_mu": 2.05, "u_sigma": 0.25, "q_psd_base": 2.0, "diff_ea": 0.8, "t1au": 190000.0},

    #{"u_model": "gaussian_random", "u_mu": 1.85, "u_sigma": 0.25, "q_psd_base": 2.0, "diff_ea": 0.8, "t1au": 190000.0},
]

send_discord(f"🚀 **バッチ処理を開始します！** (全 {len(queue)} ジョブ)")

for i, params in enumerate(queue):
    job_info = f"Model={params['u_model']}, Mu={params['u_mu']}, Sig={params['u_sigma']}, Q={params['q_psd_base']}e-20, Ea={params['diff_ea']}eV"
    send_discord(f"⏳ **[{i+1}/{len(queue)}] 実行開始:** `{job_info}`")
    print(f"[{i+1}/{len(queue)}] 実行開始: {job_info}")

    cmd = [
        "python", "mkNaColumnDensity9.9_test3.py",
        "--u_model", str(params["u_model"]),
        "--u_mu", str(params["u_mu"]),
        "--u_sigma", str(params["u_sigma"]),
        "--q_psd_base", str(params["q_psd_base"]),
        "--diff_ea", str(params["diff_ea"]),
        "--t1au", str(params["t1au"])
    ]

    start_time = time.time()
    result = subprocess.run(cmd)
    elapsed_hours = (time.time() - start_time) / 3600

    if result.returncode == 0:
        send_discord(f"✅ **[{i+1}/{len(queue)}] 完了！** (所要時間: {elapsed_hours:.2f}時間)")
    else:
        send_discord(f"❌ **[{i+1}/{len(queue)}] エラー終了しました...**")
        break 

send_discord("🎉 **すべての計算が完了しました！** お疲れ様でした。")