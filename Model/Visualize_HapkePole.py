import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'MS Gothic'

# ==========================================
# 1. パラメータ設定
# ==========================================
g_deg = 30           # 位相角 g (度)
r = 1.0              # 円柱の半径 a_E
t_max = 6.0          # ポールの長さ
res_theta = 60       # 円周方向の解像度
res_t = 60           # 長さ方向の解像度

g_rad = np.radians(g_deg)

# ==========================================
# 2. 2本のポールの中心軸ベクトル（単位ベクトル）
# ==========================================
# 散乱面を X-Z 平面とします。Y軸が「奥行き」です。
v1 = np.array([np.sin(g_rad/2), 0, np.cos(g_rad/2)])   # ポール1 (入射)
v2 = np.array([-np.sin(g_rad/2), 0, np.cos(g_rad/2)])  # ポール2 (出射)

# それぞれの軸に直交する基底ベクトルを計算 (円柱の側面を作るため)
u1 = np.array([0, 1, 0])
w1 = np.array([-np.cos(g_rad/2), 0, np.sin(g_rad/2)])

u2 = np.array([0, 1, 0])
w2 = np.array([-np.cos(g_rad/2), 0, -np.sin(g_rad/2)])

# ==========================================
# 3. 円柱表面のメッシュデータを生成する関数
# ==========================================
def generate_cylinder(v, u, w, r, t_max, res_theta, res_t):
    theta = np.linspace(0, 2*np.pi, res_theta)
    t = np.linspace(0, t_max, res_t)
    THETA, T = np.meshgrid(theta, t)
    
    # 媒介変数表示で円柱表面の座標を計算
    X = T * v[0] + r * np.cos(THETA) * u[0] + r * np.sin(THETA) * w[0]
    Y = T * v[1] + r * np.cos(THETA) * u[1] + r * np.sin(THETA) * w[1]
    Z = T * v[2] + r * np.cos(THETA) * u[2] + r * np.sin(THETA) * w[2]
    return X, Y, Z

X1, Y1, Z1 = generate_cylinder(v1, u1, w1, r, t_max, res_theta, res_t)
X2, Y2, Z2 = generate_cylinder(v2, u2, w2, r, t_max, res_theta, res_t)

# ==========================================
# 4. 重なり部分（共通体積の表面）の抽出
# ==========================================
# 相手の円柱の軸からの距離が 半径r 以下になる「表面の点」を探します
def get_overlap_points(X, Y, Z, v_opp):
    # 相手の軸(v_opp)との外積の大きさ＝軸からの距離
    dist_sq = (Y*v_opp[2] - Z*v_opp[1])**2 + (Z*v_opp[0] - X*v_opp[2])**2 + (X*v_opp[1] - Y*v_opp[0])**2
    mask = (dist_sq <= r**2) &  (Z >= 0)
    return X[mask], Y[mask], Z[mask]

# ポール1の表面のうち、ポール2の内部にある点
X1_over, Y1_over, Z1_over = get_overlap_points(X1, Y1, Z1, v2)
# ポール2の表面のうち、ポール1の内部にある点
X2_over, Y2_over, Z2_over = get_overlap_points(X2, Y2, Z2, v1)

# ==========================================
# 5. 3D描画
# ==========================================
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# ポール1・2全体を半透明のメッシュで描画 (plot_surfaceなので超軽量)
ax.plot_surface(X1, Y1, Z1, color='blue', alpha=0.15, edgecolor='none')
ax.plot_surface(X2, Y2, Z2, color='red', alpha=0.15, edgecolor='none')

# 重なり部分（共通体積の殻）を濃い紫色の点群で描画
# 全空間の点群ではなく「表面の点だけ」なので、数千点しかなく非常に軽いです
ax.scatter(X1_over, Y1_over, Z1_over, color='purple', s=4, alpha=0.7, label='Overlap Volume')
ax.scatter(X2_over, Y2_over, Z2_over, color='purple', s=4, alpha=0.7)

# 視点を見やすい角度に初期設定
ax.view_init(elev=20, azim=-45)

# 軸ラベルと範囲の設定
ax.set_xlabel('X (Width / 散乱面に沿った横幅)')
ax.set_ylabel('Y (Depth / 散乱面に垂直な奥行き)')
ax.set_zlabel('Z (Height / 高さ)')
ax.set_title(f'3D Geometry of Overlapping Cylinders (g = {g_deg}°)')

# スケールを均等にする
max_range = t_max
ax.set_xlim(-max_range/2, max_range/2)
ax.set_ylim(-max_range/2, max_range/2)
ax.set_zlim(0, max_range)
ax.set_box_aspect((1, 1, 1))

plt.show()