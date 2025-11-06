import matplotlib.pyplot as plt
import numpy as np


def create_sws_plot_separate_surfaces():
    """
    SWS (CPS) の発生領域を、ベースの球体とは「別サーフェス」として描画します。
    SWS領域はベースの球体(r=1.0)よりわずかに外側(r=1.001)にプロットし、
    境界のガタつきと描画の重なりを防ぎます。
    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # --- 1. ベースとなる全球をプロット (グレー) ---
    res_global = 100  # 全球の解像度
    phi_g = np.linspace(0, 2 * np.pi, res_global)
    theta_g = np.linspace(0, np.pi, res_global)
    phi_g_m, theta_g_m = np.meshgrid(phi_g, theta_g)

    # 半径 1.0 でベースの球を描画
    x_g = 1.0 * np.sin(theta_g_m) * np.cos(phi_g_m)
    y_g = 1.0 * np.sin(theta_g_m) * np.sin(phi_g_m)
    z_g = 1.0 * np.cos(theta_g_m)

    ax.plot_surface(
        x_g, y_g, z_g,
        color=[0.8, 0.8, 0.8, 0.2],  # グレー (半透明)
        rstride=1, cstride=1,
        antialiased=False, shade=False, linewidth=0.0
    )

    # --- 2. SWS領域のパラメータ (度) ---
    # ★ 論文の値 (-70, 70) に戻すか、-60, 60 のままにするか選べます
    LON_MIN_DEG, LON_MAX_DEG = -40, 40
    LAT_N_MIN_DEG, LAT_N_MAX_DEG = 30, 60
    LAT_S_MIN_DEG, LAT_S_MAX_DEG = -60, -30

    res_sws = 50  # SWS領域専用の解像度 (細かすぎなくてもOK)

    # --- 3. SWS 北半球領域をプロット (赤) ---

    # 経度と緯度を「度」から「ラジアン」に変換してlinspaceを作成
    phi_sws = np.linspace(np.deg2rad(LON_MIN_DEG), np.deg2rad(LON_MAX_DEG), res_sws)

    # 緯度(lat)から天頂角(theta)に変換: theta = 90 - lat
    # (北半球: 90 - 60 = 30度, 90 - 20 = 70度)
    theta_sws_N = np.linspace(np.deg2rad(90 - LAT_N_MAX_DEG), np.deg2rad(90 - LAT_N_MIN_DEG), res_sws)
    phi_sws_N_m, theta_sws_N_m = np.meshgrid(phi_sws, theta_sws_N)

    # ★半径 1.001 に設定し、ベースの球体よりわずかに外側にする
    sws_radius = 1.01
    x_sws_N = sws_radius * np.sin(theta_sws_N_m) * np.cos(phi_sws_N_m)
    y_sws_N = sws_radius * np.sin(theta_sws_N_m) * np.sin(phi_sws_N_m)
    z_sws_N = sws_radius * np.cos(theta_sws_N_m)

    ax.plot_surface(x_sws_N, y_sws_N, z_sws_N, color='red', shade=False, antialiased=True)

    # --- 4. SWS 南半球領域をプロット (赤) ---

    # 緯度(lat)から天頂角(theta)に変換: theta = 90 - lat
    # (南半球: 90 - (-20) = 110度, 90 - (-60) = 150度)
    theta_sws_S = np.linspace(np.deg2rad(90 - LAT_S_MAX_DEG), np.deg2rad(90 - LAT_S_MIN_DEG), res_sws)
    phi_sws_S_m, theta_sws_S_m = np.meshgrid(phi_sws, theta_sws_S)

    # ★同様に半径 1.001 に設定
    x_sws_S = sws_radius * np.sin(theta_sws_S_m) * np.cos(phi_sws_S_m)
    y_sws_S = sws_radius * np.sin(theta_sws_S_m) * np.sin(phi_sws_S_m)
    z_sws_S = sws_radius * np.cos(theta_sws_S_m)

    ax.plot_surface(x_sws_S, y_sws_S, z_sws_S, color='red', shade=False, antialiased=True)

    # --- 5. 緯度・経度線 (元のコードと同じ) ---
    line_points = np.linspace(0, 2 * np.pi, 200)
    line_radius = 1.01  # SWS領域(1.001)より更に外側

    # 赤道 (z=0)
    x_eq = line_radius * np.cos(line_points)
    y_eq = line_radius * np.sin(line_points)
    z_eq = np.zeros_like(line_points)
    ax.plot(x_eq, y_eq, z_eq, color='black', linestyle='-', linewidth=1)

    # 経度0度/180度 (y=0)
    #x_mer0 = line_radius * np.cos(line_points)
    #y_mer0 = np.zeros_like(line_points)
    #z_mer0 = line_radius * np.sin(line_points)
    #ax.plot(x_mer0, y_mer0, z_mer0, color='black', linestyle='--', linewidth=1)  # 破線に変更

    # 経度90度/-90度 (x=0)
    x_mer90 = np.zeros_like(line_points)
    y_mer90 = line_radius * np.cos(line_points)
    z_mer90 = line_radius * np.sin(line_points)
    ax.plot(x_mer90, y_mer90, z_mer90, color='black', linestyle='-', linewidth=1)

    # --- 6. 軸ラベルと視点 (元のコードと同じ) ---
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('SWS_ryoiki')

    # 視点の設定 (太陽方向X軸が手前に来るように調整)
    ax.view_init(elev=20., azim=45)

    # アスペクト比を揃える
    ax.set_box_aspect([1, 1, 1])

    # (オプション) 見やすくするために背景や軸を消す
    # ax.set_axis_off()

    print("プロット用の Figure オブジェクトが定義されました。")
    return fig


# --- このコードを実行環境にコピーして使用する場合 ---
if __name__ == "__main__":
    # 関数を呼び出してプロットオブジェクトを作成
    my_figure = create_sws_plot_separate_surfaces()

    # ★ 表示する場合
    plt.show()

    # ★ または、ファイルに保存する場合
    # my_figure.savefig("sws_region_separate.png", dpi=300)

    print("スクリプトが完了しました。")