# -*- coding: utf-8 -*-
"""PCA 降维演示：3D -> 2D 前后对比图（用于 LoRA 低秩假设博文配图）

思路：构造"本质上只有 2 个自由度"的 3D 数据（平面内高斯分布 + 少量法向噪声），
再用 PCA 找出前两个主成分做投影。若降维后方差保留率接近 100%，
说明这批 3D 数据其实近似位于一个 2D 平面上——这正是"低秩"的几何直观。
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]  # 中文字体
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)

# ---------- 1) 构造近似低秩的 3D 数据 ----------
n = 600
latent = rng.normal(size=(n, 2)) * [3.0, 1.5]          # 2 维隐空间坐标
u1 = np.array([1.0, 0.6, 0.2]);  u1 /= np.linalg.norm(u1)   # 平面基向量1
u2 = np.array([-0.3, 0.8, 0.5]); u2 /= np.linalg.norm(u2)   # 平面基向量2
n3 = np.cross(u1, u2); n3 /= np.linalg.norm(n3)             # 平面法向
mean = np.array([2.0, -1.0, 3.0])
noise = rng.normal(scale=0.15, size=n)                      # 法向小噪声
X = mean + latent @ np.vstack([u1, u2]) + np.outer(noise, n3)

# ---------- 2) PCA（numpy SVD 实现，无需 sklearn） ----------
Xc = X - X.mean(axis=0)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
ratio = S**2 / (S**2).sum()          # 各主成分方差贡献率
P = Xc @ Vt[:2].T                    # 投影到前 2 个主成分
var2 = ratio[:2].sum()               # 前 2 维保留的方差比例

# ---------- 3) 绘图：左 3D 原始，右 2D 降维后，同色对应 ----------
theta = np.arctan2(latent[:, 1], latent[:, 0])              # 按隐空间角度着色
colors = plt.cm.hsv((theta + np.pi) / (2 * np.pi))

fig = plt.figure(figsize=(13, 6), dpi=150)

ax1 = fig.add_subplot(121, projection="3d")
# 画出 PCA 找到的最优 2D 平面（低秩子空间的可视化）
g = np.linspace(-7, 7, 2)
G1, G2 = np.meshgrid(g, g)
plane = X.mean(axis=0) + G1[..., None] * Vt[0] + G2[..., None] * Vt[1]
ax1.plot_surface(plane[..., 0], plane[..., 1], plane[..., 2],
                 alpha=0.12, color="gray")
ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors, s=12, depthshade=False)
ax1.set_title(f"原始 3D 数据（n={n}）")
ax1.set_xlabel("x1"); ax1.set_ylabel("x2"); ax1.set_zlabel("x3")

ax2 = fig.add_subplot(122)
ax2.scatter(P[:, 0], P[:, 1], c=colors, s=12)
ax2.set_title(f"PCA 降到 2D 后（保留方差 {var2:.1%}）")
ax2.set_xlabel("主成分 1"); ax2.set_ylabel("主成分 2")
ax2.axis("equal"); ax2.grid(alpha=0.3)

fig.suptitle("PCA 降维前后对比：3D 数据本质上近似位于一个 2D 平面（低秩子空间）上", y=0.98)
fig.tight_layout()

out = r"d:\大三上的资料\假期学习记录\lora-blog\pca_3d_to_2d.png"
fig.savefig(out, bbox_inches="tight")
print("saved:", out)
print("explained variance ratio:", np.round(ratio, 4))
