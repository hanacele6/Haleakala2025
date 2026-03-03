import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager

# ★重要: 日本語を表示するためにフォントを指定してください
# Windowsの例: r'C:\Windows\Fonts\msmincho.ttc' など
# Macの例: '/System/Library/Fonts/Hiragino Sans GB.ttc' など
# ここでは標準的な設定としていますが、文字化けする場合はパスを変更してください
plt.rcParams['font.family'] = 'sans-serif'


def draw_surface_density_japanese():
    fig, ax = plt.subplots(figsize=(10, 6))

    # 軸を消す
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)

    # タイトル
    ax.text(5, 5.5, "表面密度の更新プロセス", ha='center', va='center', fontsize=20, fontweight='bold')

    # グリッド（表面セル）の描画
    grid_x = 3
    grid_y = 1
    size = 4

    # セル本体
    rect = patches.Rectangle((grid_x, grid_y), size, size, linewidth=2, edgecolor='black', facecolor='#e6f2ff')
    ax.add_patch(rect)

    # セル内の文字
    ax.text(grid_x + size / 2, grid_y + size / 2, "表面グリッド\n(緯度, 経度)\n\n現在の密度 $N$",
            ha='center', va='center', fontsize=16)

    # --- 放出 (Loss) プロセス ---
    # 赤い矢印（外へ）
    ax.arrow(grid_x + size * 0.7, grid_y + size * 0.7, 1.5, 1.5,
             head_width=0.3, head_length=0.3, fc='red', ec='red', linewidth=3)
    # 粒子
    ax.scatter([grid_x + size * 0.85, grid_x + size * 1.0, grid_x + size * 1.15],
               [grid_y + size * 0.95, grid_y + size * 1.1, grid_y + size * 1.25],
               color='red', s=80)
    # ラベル
    ax.text(grid_x + size + 0.5, grid_y + size + 1.2,
            "放出 (Loss)\n(熱脱離・PSD・SWS)\n密度が減る (-)",
            color='red', fontsize=14, ha='left', fontweight='bold')

    # --- 吸着 (Gain) プロセス ---
    # 緑の矢印（内へ）
    ax.arrow(grid_x - 1.0, grid_y + size + 1.0, 1.5, -1.5,
             head_width=0.3, head_length=0.3, fc='green', ec='green', linewidth=3)
    # 粒子
    ax.scatter([grid_x - 0.8, grid_x - 0.5, grid_x - 0.2],
               [grid_y + size + 0.8, grid_y + size + 0.5, grid_y + size + 0.2],
               color='green', s=80)
    # ラベル
    ax.text(grid_x - 1.2, grid_y + size + 1.2,
            "再衝突・吸着 (Gain)\n密度が増える (+)",
            color='green', fontsize=14, ha='right', fontweight='bold')

    # --- 下部の数式 ---
    equation = r"$N_{new} = N_{old} + \mathbf{Gain} (吸着) - \mathbf{Loss} (放出)$"
    ax.text(5, 0.5, equation, ha='center', va='center', fontsize=18,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.tight_layout()
    plt.show()


# 実行
if __name__ == "__main__":
    draw_surface_density_japanese()