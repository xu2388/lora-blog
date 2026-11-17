# -*- coding: utf-8 -*-
"""过拟合与自由度演示：2 点回归直线 vs 3 点拟合多条候选曲线（博文配图）

左图：2 个点 —— 唯一一条直线完美拟合（自由度恰好）。
中图：3 个点 —— 一条二次曲线唯一穿过，但多条高次曲线同样"完美穿过"，
      展现参数冗余带来的多解性与采样机会。
右图：3 个点 + 多条高次候选曲线的"解空间"全景 —— 参数远超样本数时，
      能映射出更多特征表示（强表达力），但也埋下过拟合的种子。
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(7)

# 三个固定数据点（刻意选得像"来自一条简单曲线"）
pts_x = np.array([-1.0, 0.0, 1.0])
pts_y = np.array([0.4, -0.6, 0.2])

x_dense = np.linspace(-1.8, 1.8, 400)

# ---------- 候选曲线：过 3 点的多项式 ----------
# 二次（唯一确定）+ 多条四次/五次/六次（同过 3 点，但形状各异）
cands = []

# 二次多项式：3 个约束唯一确定
coef_q = np.polyfit(pts_x, pts_y, 2)
cands.append((np.poly1d(coef_q), "二次曲线（唯一）", "tab:green", 2.2, "-"))

# 高次曲线族：再加"隐式约束"（额外控制点）生成不同形状，且全部过 3 点
extra_ctrl = [
    [(0.5, 2.2), (-0.5, -2.0)],   # 剧烈上翘下探
    [(0.5, -1.8), (-0.5, 2.4)],   # 反向摆动
    [(0.3, 3.2), (-0.3, -1.2)],   # 大幅震荡
    [(0.6, 0.9), (-0.6, -0.9), (0.0, 2.6)],
]
for i, ctrl in enumerate(extra_ctrl):
    xs = np.concatenate([pts_x, np.array([c[0] for c in ctrl])])
    ys = np.concatenate([pts_y, np.array([c[1] for c in ctrl])])
    deg = len(ctrl) + 2  # 4/4/4/5 次
    p = np.poly1d(np.polyfit(xs, ys, deg))
    cands.append((p, f"{deg}次曲线（候选{i+1}）", None, 1.4, "--"))

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), dpi=150)

# ---------- 左图：2 点 -> 直线唯一 ----------
ax = axes[0]
two_x, two_y = pts_x[:2], pts_y[:2]
w = np.polyfit(two_x, two_y, 1)
ax.plot(x_dense, np.poly1d(w)(x_dense), "tab:blue", lw=2.4, label="唯一一条直线")
ax.scatter(two_x, two_y, s=110, zorder=5, c="crimson", edgecolors="k")
ax.set_title("2 个点：直线唯一确定\n（自由度 = 约束数，没有多余选择）")
ax.set_xlim(-2, 2); ax.set_ylim(-3, 3)
ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=9)

# ---------- 中图：3 点 -> 二次唯一 + 高次多解 ----------
ax = axes[1]
p2, name2, c2, lw2, ls2 = cands[0]
ax.plot(x_dense, p2(x_dense), color=c2, lw=lw2, ls=ls2, label=name2)
for p, name, _, lw, ls in cands[1:3]:
    ax.plot(x_dense, p(x_dense), lw=lw, ls=ls, alpha=0.85, label=name)
ax.scatter(pts_x, pts_y, s=110, zorder=5, c="crimson", edgecolors="k")
ax.set_title("3 个点：二次唯一，高次同样'完美穿过'\n（参数 > 约束 => 多条曲线都满足）")
ax.set_xlim(-2, 2); ax.set_ylim(-3, 3)
ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=9)

# ---------- 右图：解空间全景 ----------
ax = axes[2]
for i, (p, name, c, lw, ls) in enumerate(cands[1:]):
    ax.plot(x_dense, p(x_dense), lw=1.2, ls="--", alpha=0.6,
            color=plt.cm.viridis(0.15 + 0.7 * i / max(1, len(cands) - 2)))
ax.plot(x_dense, p2(x_dense), color="tab:green", lw=2.2, label="二次（最简解）")
ax.scatter(pts_x, pts_y, s=110, zorder=5, c="crimson", edgecolors="k")
ax.fill_between(x_dense, -3, 3, where=np.ones_like(x_dense, dtype=bool),
                color="none")
ax.set_title("参数冗余 = 更大的假设空间\n（更多采样机会/更强表达力，但有过拟合风险）")
ax.set_xlim(-2, 2); ax.set_ylim(-3, 3)
ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=9)

fig.suptitle("自由度与过拟合：从'两点定线'到'三点多解'——参数越多，能映射的特征表示越多", y=1.02)
fig.tight_layout()

out = r"d:\大三上的资料\假期学习记录\lora-blog\dof_overfit_demo.png"
fig.savefig(out, bbox_inches="tight")
print("saved:", out)
