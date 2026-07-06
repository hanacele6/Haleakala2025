import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import os
import pandas as pd

# ==========================================
# 共通の物理定数とパラメータ
# ==========================================
nu = 1e13        # 頻度因子 (s^-1)
k_B = 8.617e-5   # ボルツマン定数 (eV/K)
beta_fixed = 7.0 # 昇温速度 (K/s)

# ==========================================
# シミュレーション用の関数群
# ==========================================
def dtheta_dT_vec(theta_vec, T, beta, E_arr):
    return -(nu / beta) * theta_vec * np.exp(-E_arr / (k_B * T))

def simulate_tpd_curve(T_eval, E_mean, sigma, amplitude):
    # curve_fit の仕様上、シミュレーション関数がそのままモデル関数になります
    E_d_arr = np.linspace(1.2, 3.0, 200)
    sigma = max(sigma, 1e-5)
    
    weights = np.exp(-0.5 * ((E_d_arr - E_mean) / sigma)**2)
    weights /= np.sum(weights)

    T_start = min(300.0, min(T_eval))
    T_dense = np.linspace(T_start, max(T_eval), 1000)
    
    theta_sol = odeint(dtheta_dT_vec, weights, T_dense, args=(beta_fixed, E_d_arr), atol=1e-8, rtol=1e-6)

    rate_tot = np.zeros(len(T_dense))
    for i in range(len(T_dense)):
        rate_tot[i] = np.sum(nu * theta_sol[i, :] * np.exp(-E_d_arr / (k_B * T_dense[i])))

    rate_interp = np.interp(T_eval, T_dense, rate_tot)
    
    max_rate = np.max(rate_interp)
    if max_rate > 0:
        return (rate_interp / max_rate) * amplitude
    return rate_interp

# ==========================================
# メイン処理
# ==========================================
csv_filename = 'Yakshinskiy2000exp.csv'

if not os.path.exists(csv_filename):
    print(f"エラー: {csv_filename} が見つかりません。")
else:
    print("CSVデータを読み込み中...")
    df = pd.read_csv(csv_filename)
    
    if 'Line_ID' in df.columns:
        df = df[df['Line_ID'] == 0]

    df = df.sort_values(by='X_Value')
    T_exp = df['X_Value'].values
    Signal_exp = df['Y_Value'].values
    
    mask = (T_exp >= 400) & (T_exp <= 950)
    T_exp_fit = T_exp[mask]
    Signal_exp_fit = Signal_exp[mask]

    if len(T_exp_fit) == 0:
        print("エラー: データが存在しません。")
    else:
        # スケールを合わせるための規格化
        max_sig = np.max(Signal_exp_fit)
        Signal_exp_fit_norm = Signal_exp_fit / max_sig
        
        # 初期推測値: [E_mean, sigma, amplitude]
        p0 = [1.85, 0.2, 1.0]
        
        # 物理的な制約 (bounds) の設定: (下限のリスト, 上限のリスト)
        # [E_mean, sigma, amplitude] の順に設定
        lower_bounds = [1.2, 0.01, 0.0]
        upper_bounds = [3.0, 1.0, np.inf]
        
        print("\ncurve_fit（Levenberg-Marquardt / TRF法）でフィッティングを開始します...")
        print("しばらくお待ちください（数分かかる場合があります）\n")
        
        try:
            # scipy.optimize.curve_fit を実行
            # diff_step を大きめ（デフォルトの 1.5e-8 から 0.01 等へ）に設定することで、微小なノイズを跨ぐ
            popt, pcov = curve_fit(
                simulate_tpd_curve, 
                T_exp_fit, 
                Signal_exp_fit_norm, 
                p0=p0, 
                bounds=(lower_bounds, upper_bounds),
                diff_step=0.01  # ← ギザギザ対策の要
            )
            
            print("\n" + "="*40)
            print("🎉 フィッティング成功！")
            
            opt_E_mean = popt[0]
            opt_sigma = popt[1]
            opt_amp_norm = popt[2]
            opt_amp_real = opt_amp_norm * max_sig
            
            print(f" 最頻値 (E_mean): {opt_E_mean:.4f} eV")
            print(f" 標準偏差 (sigma): {opt_sigma:.4f} eV")
            print("="*40 + "\n")

            # 結果のプロット
            plt.figure(figsize=(10, 6))
            plt.plot(T_exp, Signal_exp, label='Experimental Data', color='black', alpha=0.5, linewidth=2)
            
            sim_fit = simulate_tpd_curve(T_exp_fit, opt_E_mean, opt_sigma, opt_amp_real)
            plt.plot(T_exp_fit, sim_fit, label=f'Fitted Model\n(Mean={opt_E_mean:.3f}eV, $\sigma$={opt_sigma:.3f}eV)', 
                     color='red', linewidth=3, linestyle='--')

            plt.xlabel('Temperature T (K)', fontsize=14)
            plt.ylabel('TPD Signal', fontsize=14)
            plt.title('Na TPD Curve Fitting', fontsize=16)
            plt.xlim(400, 1000)
            plt.legend(fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.show()

        except RuntimeError as e:
            print("\n" + "="*40)
            print("⚠️ フィッティングが収束しませんでした。")
            print(f"エラー詳細: {e}")
            print("="*40 + "\n")