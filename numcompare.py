
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# =========================
# Fig.5 风格：不同 UAV 数量下最优路径长度对比
# 说明：
# - 固定同一批设备点位，保证对比公平
# - 对 UAV 数量 = {1, 3, 5} 进行：KMeans 分区 + GA 轨迹优化
# - 输出：总最优路径长度(∑Li)、最大UAV路径长度(max Li)、运行时间
# - 可视化：收敛曲线对比 + 终值对比柱状图 + 运行时间柱状图
# =========================

# ---- 中文字体（如你环境没 SimHei，可删掉这两行）----
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def generate_points(num_points: int = 100, x_range=(1, 1000), y_range=(1, 1000)):
    points = []
    for _ in range(num_points):
        points.append([random.randint(*x_range), random.randint(*y_range)])
    return points


def calculate_distance(p1, p2) -> float:
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def calculate_total_distance(path, points) -> float:
    """路径为点索引序列；计算闭环总距离"""
    n = len(path)
    if n <= 1:
        return 0.0
    dist = 0.0
    for i in range(n - 1):
        dist += calculate_distance(points[path[i]], points[path[i + 1]])
    dist += calculate_distance(points[path[-1]], points[path[0]])
    return dist


def kmeans_cluster(points, n_clusters: int, random_state: int = 42):
    """按 UAV 数量分区（KMeans 聚类）"""
    pts = np.asarray(points, dtype=float)
    # 兼容不同 sklearn 版本：不使用 n_init="auto"
    model = KMeans(n_clusters=n_clusters, max_iter=2000, random_state=random_state, n_init=10)
    labels = model.fit_predict(pts)

    cluster_dic = {i: [] for i in range(n_clusters)}
    for idx, lb in enumerate(labels):
        cluster_dic[int(lb)].append(points[idx])
    return cluster_dic, model


class ImprovedGA:
    """用于每个聚类内的 TSP 闭环优化"""

    def __init__(self, points, pop_size=60, elite_size=10, mutation_rate=0.06, generations=200, verbose=False):
        self.points = points
        self.pop_size = int(pop_size)
        self.elite_size = int(elite_size)
        self.mutation_rate = float(mutation_rate)
        self.generations = int(generations)
        self.verbose = bool(verbose)

        self.num_points = len(points)

    def create_individual(self):
        ind = list(range(self.num_points))
        random.shuffle(ind)
        return ind

    def initial_population(self):
        return [self.create_individual() for _ in range(self.pop_size)]

    def rank_population(self, population):
        scored = [(ind, calculate_total_distance(ind, self.points)) for ind in population]
        scored.sort(key=lambda x: x[1])
        return scored

    def tournament_selection(self, ranked_pop, tournament_size=5):
        selected = []
        # 精英保留
        selected.extend([ranked_pop[i][0] for i in range(min(self.elite_size, len(ranked_pop)))])
        # 锦标赛
        while len(selected) < self.pop_size:
            tournament = random.sample(ranked_pop, k=min(tournament_size, len(ranked_pop)))
            winner = min(tournament, key=lambda x: x[1])[0]
            selected.append(winner)
        return selected

    def ordered_crossover(self, p1, p2):
        size = len(p1)
        child = [-1] * size
        a, b = sorted(random.sample(range(size), 2))
        child[a:b + 1] = p1[a:b + 1]
        pos = (b + 1) % size
        for gene in p2:
            if gene not in child:
                while child[pos] != -1:
                    pos = (pos + 1) % size
                child[pos] = gene
        return child

    def swap_mutation_once(self, ind):
        """更温和：每个个体最多做一次 swap（比逐基因 swap 更平滑）"""
        out = ind.copy()
        if random.random() < self.mutation_rate and len(out) >= 2:
            i, j = random.sample(range(len(out)), 2)
            out[i], out[j] = out[j], out[i]
        return out

    def inversion_mutation(self, ind):
        out = ind.copy()
        if random.random() < self.mutation_rate and len(out) >= 4:
            a, b = sorted(random.sample(range(len(out)), 2))
            out[a:b + 1] = reversed(out[a:b + 1])
        return out

    def evolve(self):
        population = self.initial_population()
        best_ind, best_fit = None, float('inf')
        progress = []

        for gen in range(self.generations):
            ranked = self.rank_population(population)
            cur_ind, cur_fit = ranked[0]
            if cur_fit < best_fit:
                best_fit = cur_fit
                best_ind = cur_ind.copy()
            progress.append(best_fit)

            if self.verbose and gen % 50 == 0:
                avg_fit = float(np.mean([f for _, f in ranked]))
                print(f"    gen={gen:>4d} best={best_fit:,.2f} avg={avg_fit:,.2f}")

            selected = self.tournament_selection(ranked)
            next_pop = selected[:self.elite_size]

            while len(next_pop) < self.pop_size:
                p1, p2 = random.sample(selected, 2)
                child = self.ordered_crossover(p1, p2)
                child = self.swap_mutation_once(child)
                child = self.inversion_mutation(child)
                next_pop.append(child)

            population = next_pop[:self.pop_size]

        return best_ind, best_fit, progress


def solve_one_uav_setting(points, n_uav: int, ga_params: dict, seed: int = 42, verbose=False):
    """给定 UAV 数量，输出：每个UAV最优路径长度、总长度、最大长度、总收敛曲线（逐代总best）"""
    set_seed(seed)

    # 1) 分区
    cluster_dic, _ = kmeans_cluster(points, n_clusters=n_uav, random_state=42)

    # 2) 每个分区跑 GA
    per_uav_best = []
    per_uav_progress = []
    t0 = time.time()

    for cid in sorted(cluster_dic.keys()):
        pts = cluster_dic[cid]
        if len(pts) < 2:
            per_uav_best.append(0.0)
            per_uav_progress.append([0.0] * ga_params["generations"])
            continue

        ga = ImprovedGA(points=pts, verbose=verbose, **ga_params)
        _, best_fit, prog = ga.evolve()
        per_uav_best.append(float(best_fit))
        per_uav_progress.append([float(x) for x in prog])

    t1 = time.time()

    total_best = float(np.sum(per_uav_best))
    max_best = float(np.max(per_uav_best)) if per_uav_best else 0.0
    runtime = float(t1 - t0)

    # 3) 逐代“总best”收敛曲线：把每个簇的 best-so-far 叠加
    #    由于每个簇 GA 迭代次数相同，这里可以直接逐代求和。
    total_progress = list(np.sum(np.asarray(per_uav_progress, dtype=float), axis=0))

    return {
        "UAV数量": n_uav,
        "总最优路径长度": total_best,
        "最大UAV路径长度": max_best,
        "运行时间(s)": runtime,
        "total_progress": total_progress
    }


def plot_fig5_like(results, save_prefix="fig5_uav_compare"):
    """
    results: list[dict] from solve_one_uav_setting
    输出三张图：
      1) 总收敛曲线对比（类似 Fig.5 的“多UAV更快收敛”表达）
      2) 终值对比（总长度与最大长度）
      3) 运行时间对比
    """
    # 1) 收敛曲线对比
    plt.figure(figsize=(11, 7))
    for r in results:
        prog = r["total_progress"]
        plt.plot(prog, linewidth=2, label=f'UAV={r["UAV数量"]}')
    plt.title("不同UAV数量下：总最优路径长度收敛曲线对比", fontsize=15, fontweight="bold")
    plt.xlabel("迭代次数")
    plt.ylabel("总路径长度（逐代最优）")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_convergence.png", dpi=200)
    plt.show()

    # 2) 终值对比：总长度 vs 最大长度
    labels = [str(r["UAV数量"]) for r in results]
    x = np.arange(len(labels))
    total_vals = [r["总最优路径长度"] for r in results]
    max_vals = [r["最大UAV路径长度"] for r in results]

    width = 0.36
    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(x - width/2, total_vals, width=width, label="总最优路径长度(∑Li)")
    bars2 = plt.bar(x + width/2, max_vals, width=width, label="最大UAV路径长度(max Li)")
    plt.xticks(x, labels)
    plt.xlabel("UAV数量")
    plt.ylabel("路径长度")
    plt.title("不同UAV数量下：最优路径长度终值对比", fontsize=14, fontweight="bold")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    # 标注数值
    for b in list(bars1) + list(bars2):
        h = b.get_height()
        plt.text(b.get_x() + b.get_width()/2, h, f"{h:.0f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_final_lengths.png", dpi=200)
    plt.show()

    # 3) 运行时间对比
    rt_vals = [r["运行时间(s)"] for r in results]
    plt.figure(figsize=(9, 5))
    bars = plt.bar(x, rt_vals)
    plt.xticks(x, labels)
    plt.xlabel("UAV数量")
    plt.ylabel("运行时间(s)")
    plt.title("不同UAV数量下：算法运行时间对比", fontsize=13, fontweight="bold")
    plt.grid(axis="y", alpha=0.3)
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width()/2, h, f"{h:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_runtime.png", dpi=200)
    plt.show()


def main():
    # =========================
    # 你要的：只比较 1 / 3 / 5 UAV
    # =========================
    uav_list = [1, 3, 5]

    # 为了“先出结果”，这里默认用较快的 GA 参数（你要更精细可把 generations/pop_size 调大）
    ga_params = dict(
        pop_size=60,        # 初始轨迹数量（种群规模）
        elite_size=10,
        mutation_rate=0.06,
        generations=200
    )

    # 1) 固定设备点位，保证对比公平
    set_seed(42)
    points = generate_points(num_points=100)

    # 2) 逐个 UAV 数量求解
    results = []
    for n_uav in uav_list:
        print(f"\n=== 开始：UAV数量 = {n_uav} ===")
        r = solve_one_uav_setting(points, n_uav=n_uav, ga_params=ga_params, seed=2025 + n_uav, verbose=False)
        print(f"完成：UAV={n_uav} | 总长度={r['总最优路径长度']:.2f} | 最大长度={r['最大UAV路径长度']:.2f} | 时间={r['运行时间(s)']:.2f}s")
        results.append(r)

    # 3) 输出表格（便于你写论文/做PPT）
    df = pd.DataFrame([{
        "UAV数量": r["UAV数量"],
        "总最优路径长度": r["总最优路径长度"],
        "最大UAV路径长度": r["最大UAV路径长度"],
        "运行时间(s)": r["运行时间(s)"]
    } for r in results])
    print("\n===== 汇总结果 =====")
    print(df.to_string(index=False))

    # 4) 可视化（自动保存 PNG 到当前目录）
    plot_fig5_like(results, save_prefix="fig5_uav_1_3_5")


if __name__ == "__main__":
    main()
