import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
import time
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def generate_points(num_points=100, x_range=(1, 1000), y_range=(1, 1000)):
    """生成随机点"""
    points = []
    for i in range(num_points):
        generated_data = [random.randint(x_range[0], x_range[1]),
                          random.randint(y_range[0], y_range[1])]
        points.append(generated_data)
    return points


def algo1_with_model(points, n_clusters=5):
    """K-means聚类并返回模型"""
    points_df = pd.DataFrame(points, columns=['x', 'y'])
    mod = KMeans(n_clusters=n_clusters, max_iter=2000, random_state=42)
    kresult = mod.fit_predict(points_df)
    cluster_dic = {}
    for i in range(n_clusters):
        cluster_dic[i] = []
    for i in range(len(kresult)):
        cluster_dic[kresult[i]].append(points[i])
    return cluster_dic, mod


def plot_work_area_division(points, cluster_dic, kmeans_model):
    """绘制工作区域划分图（仅散点图）"""
    plt.figure(figsize=(10, 8))

    # 设置颜色
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    cluster_names = ['区域1', '区域2', '区域3', '区域4', '区域5']

    # 转换点为numpy数组
    points_array = np.array(points)

    # 绘制散点图
    for cluster_id in range(len(cluster_dic)):
        if cluster_id in cluster_dic and len(cluster_dic[cluster_id]) > 0:
            cluster_points = np.array(cluster_dic[cluster_id])
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                        c=colors[cluster_id], label=cluster_names[cluster_id],
                        alpha=0.7, s=50)

    # 绘制聚类中心
    centers = kmeans_model.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='X', s=200,
                label='聚类中心', edgecolors='white', linewidth=2)

    # 添加区域边界（凸包或简单边界）
    for cluster_id in range(len(cluster_dic)):
        if cluster_id in cluster_dic and len(cluster_dic[cluster_id]) > 0:
            cluster_points = np.array(cluster_dic[cluster_id])
            if len(cluster_points) > 2:
                # 计算凸包
                try:
                    hull = ConvexHull(cluster_points)
                    for simplex in hull.simplices:
                        plt.plot(cluster_points[simplex, 0], cluster_points[simplex, 1],
                                 c=colors[cluster_id], alpha=0.3, linestyle='--')
                except:
                    # 如果凸包计算失败，绘制最小边界矩形
                    x_min, x_max = cluster_points[:, 0].min(), cluster_points[:, 0].max()
                    y_min, y_max = cluster_points[:, 1].min(), cluster_points[:, 1].max()
                    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                         fill=False, edgecolor=colors[cluster_id],
                                         linestyle='--', alpha=0.5)
                    plt.gca().add_patch(rect)

    plt.title('工作区域划分图', fontsize=15, fontweight='bold')
    plt.xlabel('X坐标', fontsize=12)
    plt.ylabel('Y坐标', fontsize=12)

    # 方法1：将图例放在图外右上角
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # 方法2：将图例放在图外下方（如果右上角被截断）
    # plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=6, borderaxespad=0.)

    # 调整布局，为图例留出空间
    plt.subplots_adjust(right=0.8)

    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()

    # 返回统计信息用于表格
    return centers


def plot_work_area_stats(cluster_dic, centers):
    """绘制工作区域统计表格"""
    # 创建表格数据
    table_data = []
    for cluster_id in range(len(cluster_dic)):
        if cluster_id in cluster_dic:
            center_x = centers[cluster_id, 0]
            center_y = centers[cluster_id, 1]
            device_count = len(cluster_dic[cluster_id])
            table_data.append([f'区域{cluster_id + 1}', f'{center_x:.2f}', f'{center_y:.2f}', device_count])

    # 创建表格
    columns = ['工作区域', '中心X坐标', '中心Y坐标', '设备数量']
    df = pd.DataFrame(table_data, columns=columns)

    # 创建单独的表
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')

    # 创建表格
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center',
                     bbox=[0.1, 0.2, 0.8, 0.6])

    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)

    # 设置表头样式
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # 设置表格行颜色
    light_colors = ['#ffcccc', '#ccccff', '#ccffcc', '#e6ccff', '#ffe6cc']
    for i in range(1, len(table_data) + 1):
        for j in range(len(columns)):
            table[(i, j)].set_facecolor(light_colors[i - 1])

    ax.set_title('工作区域统计信息', fontsize=15, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()

    return df


def plot_cluster_distribution(cluster_dic):
    """绘制设备数量分布图"""
    cluster_ids = []
    device_counts = []
    colors = ['red', 'blue', 'green', 'purple', 'orange']

    for cluster_id in sorted(cluster_dic.keys()):
        cluster_ids.append(f'区域{cluster_id + 1}')
        device_counts.append(len(cluster_dic[cluster_id]))

    plt.figure(figsize=(10, 6))
    bars = plt.bar(cluster_ids, device_counts, color=colors, alpha=0.7)

    # 在柱子上添加数值标签
    for bar, count in zip(bars, device_counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(count), ha='center', va='bottom', fontweight='bold')

    plt.title('各工作区域设备数量分布', fontsize=15, fontweight='bold')
    plt.xlabel('工作区域', fontsize=12)
    plt.ylabel('设备数量', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


def calculate_distance(point1, point2):
    """计算两点间欧几里得距离"""
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def calculate_total_distance(path, points):
    """计算路径总距离"""
    if len(path) <= 1:
        return 0

    total_distance = 0
    # 计算路径中相邻点的距离
    for i in range(len(path) - 1):
        total_distance += calculate_distance(points[path[i]], points[path[i + 1]])

    # 回到起点形成闭环
    total_distance += calculate_distance(points[path[-1]], points[path[0]])

    return total_distance


class ImprovedGA:
    """遗传算法（对齐 Fig.4 敏感性分析口径）

    与你 Fig.4 脚本保持一致的关键点：
    - 预计算距离矩阵（加速）
    - copy_rate(w) 机制：对差于平均的个体，以概率 w 用当前最优个体替换，并做 segment_swap 扰动
    - 可选 two-opt（默认在主流程关闭，避免收敛曲线出现“大跳水”）
    """

    def __init__(
        self,
        points,
        pop_size=100,
        elite_size=15,
        mutation_rate=0.08,
        generations=100,
        copy_rate=0.5,
        use_two_opt=False,
        verbose=False,
        seed=None,
    ):
        self.points = np.array(points, dtype=float)
        self.num_points = len(points)
        self.pop_size = int(pop_size)
        self.elite_size = int(elite_size)
        self.mutation_rate = float(mutation_rate)
        self.generations = int(generations)
        self.copy_rate = float(copy_rate) if copy_rate is not None else 0.0
        self.use_two_opt = bool(use_two_opt)
        self.verbose = bool(verbose)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # 预计算距离矩阵
        diff = self.points[:, None, :] - self.points[None, :, :]
        self.dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))

    # --------- 基础：个体/距离 ---------
    def create_individual(self):
        ind = list(range(self.num_points))
        random.shuffle(ind)
        return ind

    def initial_population(self):
        return [self.create_individual() for _ in range(self.pop_size)]

    def path_length(self, individual):
        if len(individual) <= 1:
            return 0.0
        idx = np.array(individual, dtype=int)
        return float(self.dist_matrix[idx, np.roll(idx, -1)].sum())

    def rank_individuals(self, population):
        fitness = []
        for ind in population:
            # 安全：若个体非法则重置
            if len(ind) != self.num_points or len(set(ind)) != self.num_points:
                ind = self.create_individual()
            fitness.append((ind, self.path_length(ind)))
        return sorted(fitness, key=lambda x: x[1])

    # --------- 选择 / 交叉 / 变异 ---------
    def tournament_selection(self, ranked_population, tournament_size=5):
        selected = []
        # 精英保留
        for i in range(min(self.elite_size, len(ranked_population))):
            selected.append(ranked_population[i][0])

        # 锦标赛补齐
        for _ in range(self.pop_size - len(selected)):
            tournament = random.sample(ranked_population, min(tournament_size, len(ranked_population)))
            winner = min(tournament, key=lambda x: x[1])
            selected.append(winner[0])
        return selected

    def ordered_crossover(self, parent1, parent2):
        if len(parent1) != len(parent2):
            return parent1
        size = len(parent1)
        child = [-1] * size
        start, end = sorted(random.sample(range(size), 2))
        child[start:end + 1] = parent1[start:end + 1]
        current_pos = (end + 1) % size
        for gene in parent2:
            if gene not in child:
                while child[current_pos] != -1:
                    current_pos = (current_pos + 1) % size
                child[current_pos] = gene
        return child

    def swap_mutation(self, individual):
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                j = random.randint(0, len(mutated) - 1)
                mutated[i], mutated[j] = mutated[j], mutated[i]
        return mutated

    def inversion_mutation(self, individual):
        if random.random() < self.mutation_rate:
            start, end = sorted(random.sample(range(len(individual)), 2))
            individual = individual[:start] + individual[start:end + 1][::-1] + individual[end + 1:]
        return individual

    # --------- Fig.4 对齐：segment swap & copy ---------
    def segment_swap(self, individual):
        """四断点分段交换：模拟论文 Algorithm 2 Step3 的 Swap 操作"""
        if len(individual) < 6:
            return individual
        a, b, c, d = sorted(random.sample(range(1, len(individual)), 4))
        seg0 = individual[:a]
        seg_ab = individual[a:b]
        seg_bc = individual[b:c]
        seg_cd = individual[c:d]
        seg_d = individual[d:]
        return seg0 + seg_cd + seg_bc + seg_ab + seg_d

    def two_opt(self, individual):
        improved = True
        best_distance = self.path_length(individual)
        while improved:
            improved = False
            for i in range(len(individual) - 1):
                for j in range(i + 2, len(individual)):
                    if j == len(individual) - 1 and i == 0:
                        continue
                    new_ind = individual.copy()
                    new_ind[i:j + 1] = individual[i:j + 1][::-1]
                    new_dist = self.path_length(new_ind)
                    if new_dist < best_distance:
                        individual = new_ind
                        best_distance = new_dist
                        improved = True
                        break
                if improved:
                    break
        return individual

    def evolve(self):
        population = self.initial_population()
        best_individual = None
        best_fitness = float('inf')
        progress = []

        if self.verbose:
            print(f"开始进化，种群大小: {self.pop_size}, 迭代次数: {self.generations}, copy_rate(w): {self.copy_rate}")

        for generation in range(self.generations):
            ranked = self.rank_individuals(population)

            # Step3 风格 copy&swap（对齐 Fig.4 脚本）
            if self.copy_rate and self.copy_rate > 0:
                avg_fit = float(np.mean([fit for _, fit in ranked]))
                best_ref = ranked[0][0]
                new_pop = []
                for ind, fit in ranked:
                    if fit > avg_fit and random.random() < self.copy_rate:
                        replaced = best_ref.copy()
                        replaced = self.segment_swap(replaced)
                        new_pop.append(replaced)
                    else:
                        new_pop.append(ind)
                population = new_pop
                ranked = self.rank_individuals(population)

            cur_best_ind, cur_best_fit = ranked[0]
            if cur_best_fit < best_fitness:
                best_individual = cur_best_ind.copy()
                best_fitness = cur_best_fit
                if self.use_two_opt:
                    best_individual = self.two_opt(best_individual)
                    best_fitness = self.path_length(best_individual)

            progress.append(best_fitness)

            # Step4（论文）是“删除一半最差轨迹”。为保持与你 Fig.4 程序一致，这里仍使用 GA 的选择/交叉生成下一代。
            selected = self.tournament_selection(ranked)

            next_population = []
            next_population.extend(selected[:min(self.elite_size, len(selected))])

            while len(next_population) < self.pop_size:
                parent1, parent2 = random.sample(selected, 2)
                child = self.ordered_crossover(parent1, parent2)
                child = self.swap_mutation(child)
                child = self.inversion_mutation(child)
                next_population.append(child)

            population = next_population[:self.pop_size]

        if self.verbose:
            print(f"进化完成，最终路径长度: {best_fitness:.2f}")

        return best_individual, progress


def algo2_improved(cluster_dic, G=100, copy_rate=0.5, generations=100, elite_size=15, mutation_rate=0.08, use_two_opt=False, verbose=True, seed_base=123):
    """改进的路径规划"""
    path_dic = {}
    progress_dic = {}

    for cluster_id, cluster_points in cluster_dic.items():
        if verbose:
            print(f"\n处理聚类 {cluster_id + 1}: {len(cluster_points)} 个点")

        if len(cluster_points) < 2:
            if verbose:
                print("  聚类只有1个点，跳过路径规划")
            path_dic[cluster_id] = np.array(cluster_points)
            progress_dic[cluster_id] = [0]
            continue

        # 使用改进的遗传算法
        ga = ImprovedGA(points=cluster_points, pop_size=G, elite_size=elite_size, mutation_rate=mutation_rate, generations=generations, copy_rate=copy_rate, use_two_opt=use_two_opt, verbose=False, seed=seed_base + int(cluster_id))
        best_path, progress = ga.evolve()
        progress_dic[cluster_id] = progress

        # 将路径转换为坐标点（形成闭环）
        path_coords = [cluster_points[i] for i in best_path]
        path_coords.append(path_coords[0])  # 回到起点形成闭环
        path_dic[cluster_id] = np.array(path_coords)

        initial_distance = calculate_total_distance(list(range(len(cluster_points))), cluster_points)
        final_distance = calculate_total_distance(best_path, cluster_points)
        improvement = ((initial_distance - final_distance) / initial_distance) * 100
        if verbose:
            print(f"  优化效果: {initial_distance:.2f} → {final_distance:.2f} (改进 {improvement:.1f}%)")

    return path_dic, progress_dic


def plot_improved_paths(path_dic):
    """绘制改进后的路径图"""
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    cluster_names = {i: f'U{i+1}' for i in range(len(path_dic))}

    for cluster_id, path_points in path_dic.items():
        plt.figure(figsize=(10, 8))
        plt.title(f'{cluster_names[cluster_id]}无人机优化路径规划', fontsize=15)
        plt.xlabel('X坐标', fontsize=12)
        plt.ylabel('Y坐标', fontsize=12)

        if len(path_points) > 1:
            x = path_points[:, 0]
            y = path_points[:, 1]

            # 绘制平滑路径
            plt.plot(x, y, 'o-', color=colors[cluster_id],
                     linewidth=2, markersize=6, alpha=0.8,
                     label=f'路径点: {len(path_points) - 1}')

            # 标记起点和终点
            plt.scatter(x[0], y[0], color='green', s=150, marker='*',
                        label='起点', edgecolors='black', zorder=5)
            plt.scatter(x[-2], y[-2], color='red', s=150, marker='s',
                        label='终点', edgecolors='black', zorder=5)

        else:
            plt.scatter(path_points[0, 0], path_points[0, 1],
                        color=colors[cluster_id], s=200, label='单点位置')

        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


def smooth_ema(values, alpha=0.15):
    """指数滑动平均(EMA)平滑，仅影响展示，不改变算法结果"""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    smoothed = np.empty_like(values)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def plot_convergence(progress_dic, smooth=True, alpha=0.15, show_raw=True):
    """绘制收敛曲线（横坐标每10代一个主刻度）"""
    plt.figure(figsize=(12, 8))

    # 颜色池（数量不够时循环）
    color_pool = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']

    cluster_ids = sorted(progress_dic.keys())
    cluster_names = {cid: f'U{cid + 1}' for cid in cluster_ids}

    for idx, cluster_id in enumerate(cluster_ids):
        progress = progress_dic[cluster_id]
        if len(progress) <= 1:
            continue

        raw = np.asarray(progress, dtype=float)
        color = color_pool[idx % len(color_pool)]

        if smooth:
            series = smooth_ema(raw, alpha=alpha)
            if show_raw:
                plt.plot(raw, color=color, linewidth=1, alpha=0.25)
            plt.plot(series, color=color, label=f'{cluster_names[cluster_id]}', linewidth=2.5, alpha=0.9)
            plt.scatter(0, series[0], color=color, s=50, zorder=5)
            plt.scatter(len(series) - 1, series[-1], color=color, s=50, zorder=5)
        else:
            plt.plot(raw, color=color, label=f'{cluster_names[cluster_id]}', linewidth=2, alpha=0.8)
            plt.scatter(0, raw[0], color=color, s=50, zorder=5)
            plt.scatter(len(raw) - 1, raw[-1], color=color, s=50, zorder=5)

    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(10))  # 你要求：每10代一个刻度

    plt.title('遗传算法收敛曲线', fontsize=15)
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('路径长度', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def debug_convergence(progress_dic):
    """调试收敛曲线"""
    print("\n收敛数据分析:")
    for cluster_id, progress in progress_dic.items():
        print(f"\n聚类 {cluster_id + 1}:")
        print(f"  数据点数量: {len(progress)}")
        if len(progress) > 1:
            print(f"  初始值: {progress[0]:.2f}")
            print(f"  最终值: {progress[-1]:.2f}")
            print(f"  变化范围: {max(progress) - min(progress):.2f}")
            print(f"  是否变化: {len(set(progress)) > 1}")


def _path_length_from_coords(path_points: np.ndarray) -> float:
    """闭环路径长度（输入为坐标序列，已含回到起点的最后一个点亦可）"""
    pts = np.asarray(path_points, dtype=float)
    if len(pts) < 2:
        return 0.0
    dif = np.diff(pts, axis=0)
    return float(np.linalg.norm(dif, axis=1).sum())


def _cumulative_arrival_times(path_points: np.ndarray, v_f: float) -> np.ndarray:
    """按路径顺序得到每个设备点的到达时间（不含最终回到起点的那个重复点）"""
    pts = np.asarray(path_points, dtype=float)
    if len(pts) <= 1:
        return np.zeros(len(pts), dtype=float)
    # 去掉闭环最后一个重复起点
    unique = pts[:-1] if np.allclose(pts[0], pts[-1]) and len(pts) > 2 else pts
    times = np.zeros(len(unique), dtype=float)
    for i in range(1, len(unique)):
        seg = float(np.linalg.norm(unique[i] - unique[i - 1]))
        times[i] = times[i - 1] + seg / max(v_f, 1e-9)
    return times


def _init_sellers(
    num_sellers: int,
    transactions_per_seller: int,
    seed: int = 42,
    malicious_ratio: float = 0.2,
):
    """
    初始化卖家（边缘服务器）：
    - trust: 信任值（[0,1]），低于阈值会被过滤
    - f_cpu: 计算能力（cycles/s）
    - ask_per_cycle: 要价（currency/cycle）
    - cost_per_cycle: 计算成本（currency/cycle）
    - drop_prob: 恶意丢弃任务概率（用于模拟不可信服务器）
    - capacity: 可成交次数（简化容量约束）
    - available_time: 排队/串行处理的可用时间（用于等待延迟）
    """
    rng = np.random.default_rng(seed)
    sellers = []
    malicious_flags = rng.random(num_sellers) < malicious_ratio
    for sid in range(num_sellers):
        malicious = bool(malicious_flags[sid])
        trust = float(rng.uniform(0.35, 0.6) if malicious else rng.uniform(0.65, 0.95))
        f_cpu = float(rng.uniform(2.0e9, 6.0e9))  # cycles/s
        ask_per_cycle = float(rng.uniform(2.0e-9, 8.0e-9))  # price per cycle
        cost_per_cycle = float(ask_per_cycle * rng.uniform(0.35, 0.75))  # cost < ask
        drop_prob = float(rng.uniform(0.15, 0.45) if malicious else rng.uniform(0.0, 0.03))
        sellers.append({
            "id": sid,
            "trust": trust,
            "malicious": malicious,
            "f_cpu": f_cpu,
            "ask_per_cycle": ask_per_cycle,
            "cost_per_cycle": cost_per_cycle,
            "drop_prob": drop_prob,
            "capacity": int(transactions_per_seller),
            "available_time": 0.0,
            "accepted": 0,
            "dropped": 0,
            "failed_deadline": 0,
        })
    return sellers


def simulate_transactions(
    path_dic,
    num_sellers: int = 30,
    transactions_per_seller: int = 4,
    trust_threshold: float = 0.6,
    # 多目标权重（用于“选择哪个服务器来卸载”）
    w_energy: float = 0.33,
    w_delay: float = 0.33,
    w_utility: float = 0.34,
    # 系统与任务参数（简化落地版）
    v_f: float = 100.0,        # UAV飞行速度（坐标单位/秒）
    P_f: float = 120.0,        # UAV飞行功率（W）
    P_tx: float = 1.2,         # 通信功率（W，合并设备->UAV->服务器）
    P_cpu: float = 40.0,       # 服务器处理功率（W）
    r_link: float = 8.0e6,     # 链路速率（bps，简化常数）
    mu_delay: float = 0.0,     # 把“时延”折算进效用（currency/s）；默认0表示效用只看价格/成本
    deadline_low: float = 10.0,
    deadline_high: float = 20.0,
    seed: int = 42,
    verbose: bool = False,
    save_prefix: str = "auction_moo"
):
    """
    改进版交易/拍卖（最小改动路线，落地论文(16)式的简化目标）：
      - 双边拍卖：设备(买家)出价 bid_per_cycle，服务器(卖家)要价 ask_per_cycle，满足 bid>=ask 才可能成交
      - 信任过滤：trust < trust_threshold 的服务器不参与
      - 多目标选择：在可行服务器集合中，对(能耗, 时延, 效用)做归一化加权，选综合得分最优者
      - 动态信任：成交后若服务器丢弃任务/超时，则信任下降；否则略升（用于模拟逐步识别恶意服务器）
    输出：
      - 返回 dict，包括总能耗E、总延迟D、总效用U、成功率R、综合评分F
      - 同时把逐任务日志写入 CSV、把汇总写入 TXT（便于你写实验分析）
    """
    # 初始化卖家
    sellers = _init_sellers(num_sellers, transactions_per_seller, seed=seed)

    rng = np.random.default_rng(seed + 7)
    cluster_names = {i: f'U{i+1}' for i in range(len(path_dic))}

    # 逐任务日志
    logs = []

    total_tasks = 0
    success_tasks = 0

    total_flight_energy = 0.0
    total_comm_energy = 0.0
    total_cpu_energy = 0.0
    total_delay = 0.0
    total_utility = 0.0

    # ====== 逐聚类（逐UAV）处理 ======
    for cluster_idx in range(len(path_dic)):
        if cluster_idx not in path_dic:
            continue

        path_points = np.asarray(path_dic[cluster_idx], dtype=float)
        if len(path_points) <= 1:
            continue

        # 路径长度/飞行能耗（对应论文 Ef 与 ∑Li 的关系）
        Li = _path_length_from_coords(path_points)
        Ef_i = P_f * Li / max(v_f, 1e-9)
        total_flight_energy += Ef_i

        # 到达时间（用来体现“等待延迟Dw受UAV收集策略影响”）
        arrival_times = _cumulative_arrival_times(path_points, v_f=v_f)

        # 去除闭环最后一个重复点
        unique_points = path_points[:-1] if np.allclose(path_points[0], path_points[-1]) else path_points

        if verbose:
            print(f"\n{'='*50}\n第{cluster_idx+1}组交易 ({cluster_names.get(cluster_idx, str(cluster_idx))})\n{'='*50}")
            print(f"  设备数: {len(unique_points)}, 轨迹长度Li={Li:.2f}, 飞行能耗Ef={Ef_i:.2f}J")

        # ====== 逐设备任务执行拍卖 ======
        for tid, (pt, t_arrive) in enumerate(zip(unique_points, arrival_times)):
            total_tasks += 1

            # --- 生成任务（简化随机任务） ---
            data_bits = float(rng.uniform(0.5e6, 2.0e6))    # bits
            cycles = float(rng.uniform(0.6e9, 2.2e9))       # CPU cycles

            # 设备估值与出价（按cycle计价）
            valuation_per_cycle = float(rng.uniform(6.0e-9, 1.4e-8))  # currency/cycle
            bid_per_cycle = float(valuation_per_cycle * rng.uniform(0.55, 0.95))
            expected_utility = float(rng.uniform(0.0, 0.15) * (valuation_per_cycle * cycles))  # 期望效用阈值（可选）

            # 截止时间（用于“超时失败”）
            deadline = float(rng.uniform(deadline_low, deadline_high))

            # --- 构造可行卖家集合 ---
            feasible = []
            for s in sellers:
                if s["capacity"] <= 0:
                    continue
                if s["trust"] < trust_threshold:
                    continue
                if bid_per_cycle < s["ask_per_cycle"]:
                    continue

                # 传输时间与能耗（简化为常数链路）
                tx_time = data_bits / max(r_link, 1e-9)
                comm_energy = P_tx * tx_time

                # 服务器排队与处理
                start_compute = max(s["available_time"], t_arrive + tx_time)
                wait_time = max(0.0, start_compute - (t_arrive + tx_time))
                compute_time = cycles / max(s["f_cpu"], 1e-9)
                finish_time = start_compute + compute_time

                # 任务总时延（把UAV到达时间也算进去，体现收集策略影响）
                delay = finish_time

                # 处理能耗（简化为常数功率 * 时间）
                cpu_energy = P_cpu * compute_time

                # 交易支付
                payment = s["ask_per_cycle"] * cycles
                valuation = valuation_per_cycle * cycles

                buyer_utility = valuation - payment - mu_delay * delay
                seller_utility = payment - s["cost_per_cycle"] * cycles
                total_u = buyer_utility + seller_utility

                # 选择用的三目标分量（原始值）
                energy = comm_energy + cpu_energy

                feasible.append({
                    "seller_id": s["id"],
                    "trust": s["trust"],
                    "ask_per_cycle": s["ask_per_cycle"],
                    "payment": payment,
                    "energy": energy,
                    "delay": delay,
                    "utility": total_u,
                    "buyer_utility": buyer_utility,
                    "seller_utility": seller_utility,
                    "tx_time": tx_time,
                    "wait_time": wait_time,
                    "compute_time": compute_time,
                    "finish_time": finish_time,
                })

            # 若无可行卖家 => 拍卖失败
            if not feasible:
                logs.append({
                    "cluster": cluster_idx,
                    "task_id": tid,
                    "x": float(pt[0]),
                    "y": float(pt[1]),
                    "arrival_time": float(t_arrive),
                    "data_bits": data_bits,
                    "cycles": cycles,
                    "bid_per_cycle": bid_per_cycle,
                    "valuation_per_cycle": valuation_per_cycle,
                    "deadline": deadline,
                    "status": "fail_no_feasible_seller",
                })
                continue

            # --- 对可行集合做归一化加权（多目标“选择服务器”） ---
            E_vals = np.array([c["energy"] for c in feasible], dtype=float)
            D_vals = np.array([c["delay"] for c in feasible], dtype=float)
            U_vals = np.array([c["utility"] for c in feasible], dtype=float)

            def _minmax_norm(arr):
                mn, mx = float(arr.min()), float(arr.max())
                if abs(mx - mn) < 1e-12:
                    return np.zeros_like(arr)
                return (arr - mn) / (mx - mn)

            E_n = _minmax_norm(E_vals)          # 越小越好
            D_n = _minmax_norm(D_vals)          # 越小越好
            U_n = _minmax_norm(U_vals)          # 越大越好

            scores = w_energy * E_n + w_delay * D_n - w_utility * U_n

            chosen_idx = int(np.argmin(scores))
            chosen = feasible[chosen_idx]
            sid = chosen["seller_id"]

            # --- 模拟不可信服务器“丢任务” ---
            sref = sellers[sid]
            dropped = (rng.random() < sref["drop_prob"])

            # --- 成功/失败判定 ---
            finished = chosen["finish_time"]
            success = (not dropped) and (finished <= deadline) and (chosen["buyer_utility"] >= expected_utility)

            # --- 更新卖家状态（容量/排队/信任） ---
            sref["capacity"] -= 1
            sref["accepted"] += 1
            # 只有没丢弃才占用计算队列
            if not dropped:
                sref["available_time"] = chosen["finish_time"]
                # 信任略升
                sref["trust"] = float(min(1.0, sref["trust"] + 0.01))
            else:
                sref["dropped"] += 1
                sref["trust"] = float(max(0.0, sref["trust"] - 0.12))

            if finished > deadline:
                sref["failed_deadline"] += 1
                sref["trust"] = float(max(0.0, sref["trust"] - 0.05))

            # --- 汇总系统指标（只统计“成功任务”的U、以及成功任务的能耗/时延更贴近论文口径） ---
            status = "success" if success else ("fail_drop" if dropped else "fail_deadline_or_utility")

            if success:
                success_tasks += 1
                total_comm_energy += float(P_tx * chosen["tx_time"])
                total_cpu_energy += float(P_cpu * chosen["compute_time"])
                total_delay += float(chosen["delay"])
                total_utility += float(chosen["utility"])
            else:
                # 失败任务也会消耗通信（至少发起过传输尝试），这里保守计入通信能耗
                total_comm_energy += float(P_tx * chosen["tx_time"])

            logs.append({
                "cluster": cluster_idx,
                "task_id": tid,
                "x": float(pt[0]),
                "y": float(pt[1]),
                "arrival_time": float(t_arrive),
                "data_bits": data_bits,
                "cycles": cycles,
                "bid_per_cycle": bid_per_cycle,
                "valuation_per_cycle": valuation_per_cycle,
                "deadline": deadline,
                "seller_id": sid,
                "seller_trust": float(sref["trust"]),
                "seller_malicious": bool(sref["malicious"]),
                "ask_per_cycle": chosen["ask_per_cycle"],
                "payment": chosen["payment"],
                "tx_time": chosen["tx_time"],
                "wait_time": chosen["wait_time"],
                "compute_time": chosen["compute_time"],
                "finish_time": chosen["finish_time"],
                "energy_comm": float(P_tx * chosen["tx_time"]),
                "energy_cpu": float(P_cpu * chosen["compute_time"]),
                "energy_task": float(chosen["energy"]),
                "delay_task": float(chosen["delay"]),
                "utility_total": float(chosen["utility"]),
                "buyer_utility": float(chosen["buyer_utility"]),
                "seller_utility": float(chosen["seller_utility"]),
                "score_local": float(scores[chosen_idx]),
                "status": status,
            })

    # ====== 汇总（对应论文四目标：E、D、U、R） ======
    total_energy = float(total_flight_energy + total_comm_energy + total_cpu_energy)
    success_rate = float(success_tasks / total_tasks) if total_tasks > 0 else 0.0

    # 系统级“综合评分”（把三目标合成一个标量，便于你做对比/敏感性分析）
    # 注意：此处同样做归一化，避免不同量纲主导。
    # 你可以把这个当作“多目标优化落地”的最终输出指标。
    E_sys = total_energy
    D_sys = total_delay
    U_sys = total_utility

    # 防止除0
    E_norm = E_sys / (E_sys + 1e-9)
    D_norm = D_sys / (D_sys + 1e-9)
    U_norm = U_sys / (abs(U_sys) + 1e-9)

    F = float(w_energy * E_norm + w_delay * D_norm - w_utility * U_norm)

    # ====== 输出文件 ======
    log_df = pd.DataFrame(logs)
    csv_path = f"{save_prefix}_task_log.csv"
    txt_path = f"{save_prefix}_summary.txt"
    sellers_path = f"{save_prefix}_sellers.csv"

    log_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    sellers_df = pd.DataFrame(sellers)
    sellers_df.to_csv(sellers_path, index=False, encoding="utf-8-sig")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== Utility-based Double Auction + Multi-objective Selection (Simplified) ===\n")
        f.write(f"num_sellers={num_sellers}, transactions_per_seller={transactions_per_seller}\n")
        f.write(f"trust_threshold={trust_threshold}, malicious_ratio(sim) ~ internal\n")
        f.write(f"weights: w_energy={w_energy}, w_delay={w_delay}, w_utility={w_utility}\n")
        f.write(f"UAV params: v_f={v_f}, P_f={P_f}\n")
        f.write(f"Link params: r_link={r_link}, P_tx={P_tx}\n")
        f.write(f"CPU params: P_cpu={P_cpu}\n")
        f.write("\n--- System Results ---\n")
        f.write(f"total_tasks={total_tasks}\n")
        f.write(f"success_tasks={success_tasks}\n")
        f.write(f"success_rate={success_rate:.4f}\n")
        f.write(f"total_energy_J={total_energy:.4f}\n")
        f.write(f"  flight_energy_J={total_flight_energy:.4f}\n")
        f.write(f"  comm_energy_J={total_comm_energy:.4f}\n")
        f.write(f"  cpu_energy_J={total_cpu_energy:.4f}\n")
        f.write(f"total_delay_s={total_delay:.4f}\n")
        f.write(f"total_utility={total_utility:.4f}\n")
        f.write(f"composite_score_F={F:.6f}\n")

    return {
        "total_tasks": total_tasks,
        "success_tasks": success_tasks,
        "success_rate": success_rate,
        "total_energy_J": total_energy,
        "flight_energy_J": float(total_flight_energy),
        "comm_energy_J": float(total_comm_energy),
        "cpu_energy_J": float(total_cpu_energy),
        "total_delay_s": float(total_delay),
        "total_utility": float(total_utility),
        "composite_score_F": F,
        "task_log_csv": csv_path,
        "summary_txt": txt_path,
        "sellers_csv": sellers_path,
    }



def plot_all_paths_together(path_dic, cluster_dic):
    """在同一张图上绘制所有路径"""
    plt.figure(figsize=(12, 10))
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    cluster_names = {i: f'U{i+1}' for i in range(len(path_dic))}

    # 先绘制所有点
    for cluster_id, points in cluster_dic.items():
        points_array = np.array(points)
        plt.scatter(points_array[:, 0], points_array[:, 1],
                    c=colors[cluster_id], label=cluster_names[cluster_id],
                    alpha=0.6, s=50)

    # 再绘制路径
    for cluster_id, path_points in path_dic.items():
        if len(path_points) > 1:
            x = path_points[:, 0]
            y = path_points[:, 1]
            plt.plot(x, y, 'o-', color=colors[cluster_id],
                     linewidth=2, markersize=4, alpha=0.8)

            # 标记起点
            plt.scatter(x[0], y[0], color='green', s=100, marker='*',
                        edgecolors='black', zorder=5)

    plt.title('多无人机协同路径规划总览', fontsize=15)
    plt.xlabel('X坐标', fontsize=12)
    plt.ylabel('Y坐标', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def _pad_progress(progress, generations):
    if progress is None or len(progress) == 0:
        return [0.0] * generations
    if len(progress) >= generations:
        return [float(x) for x in progress[:generations]]
    last = float(progress[-1])
    return [float(x) for x in progress] + [last] * (generations - len(progress))


def total_trajectory_progress(progress_dic, generations):
    """把每个UAV的收敛曲线相加，得到系统总轨迹长度随迭代的变化（与 Fig.4 口径一致）"""
    total = np.zeros(generations, dtype=float)
    for _, prog in progress_dic.items():
        total += np.array(_pad_progress(prog, generations), dtype=float)
    return total


def plot_total_convergence(total_progress, tick_step=10):
    """绘制系统总轨迹长度收敛曲线（Fig.4 口径）"""
    total_progress = np.asarray(total_progress, dtype=float)
    plt.figure(figsize=(12, 6))
    plt.plot(total_progress, linewidth=2.5, alpha=0.9)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(tick_step))
    plt.title('系统总轨迹长度收敛曲线（∑UAV）', fontsize=15)
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('总轨迹长度', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()





# =========================
# 交易与多目标结果可视化（新增）
# =========================
def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def _save_show(fig, out_path: str, show: bool = False):
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

def visualize_auction_results(
    auction_metrics: dict,
    show: bool = False,
    save_dir: str = "viz_outputs",
    do_weight_sweep: bool = True,
    sweep_points: int = 11,
    sweep_seed: int = 42,
    # 以下参数用于 weight sweep（会重复调用 simulate_transactions；为了公平会做固定seed）
    path_dic=None,
    num_sellers: int = 30,
    transactions_per_seller: int = 4,
    trust_threshold: float = 0.6,
):
    """对“交易模拟 + 多目标选择”输出做可视化。
    依赖：simulate_transactions() 已写出的 CSV 文件（task_log_csv, sellers_csv）。
    产出：多张 PNG 图，保存在 save_dir；返回路径字典。
    """
    save_dir = _ensure_dir(save_dir)

    task_csv = auction_metrics.get("task_log_csv")
    sellers_csv = auction_metrics.get("sellers_csv")
    if not task_csv or not os.path.exists(task_csv):
        raise FileNotFoundError(f"未找到逐任务日志CSV: {task_csv}")
    if not sellers_csv or not os.path.exists(sellers_csv):
        raise FileNotFoundError(f"未找到卖家CSV: {sellers_csv}")

    log_df = pd.read_csv(task_csv)
    sellers_df = pd.read_csv(sellers_csv)

    outputs = {"save_dir": save_dir}

    # 1) KPI：能耗分解 + 成功率/时延/效用（概览）
    outputs["kpi_overview"] = plot_kpi_overview(auction_metrics, save_dir, show=show)

    # 2) 任务层：成功/失败原因统计
    outputs["task_status"] = plot_task_status(log_df, save_dir, show=show)

    # 3) 任务层：E-D-U 三维权衡（散点图）
    outputs["tradeoff_scatter"] = plot_tradeoff_scatter(log_df, save_dir, show=show)

    # 4) 聚类层：各UAV/区域的交易表现对比
    outputs["cluster_comparison"] = plot_cluster_comparison(log_df, save_dir, show=show)

    # 5) 卖家层：信任/恶意/丢弃率关系
    outputs["seller_dashboard"] = plot_seller_dashboard(sellers_df, save_dir, show=show)

    # 6) 多目标权重扫描（可选）：展示“权重改变→E/D/U 变化”的权衡曲线
    if do_weight_sweep and path_dic is not None:
        outputs["weight_sweep"] = plot_weight_sweep(
            path_dic=path_dic,
            save_dir=save_dir,
            points=sweep_points,
            seed=sweep_seed,
            num_sellers=num_sellers,
            transactions_per_seller=transactions_per_seller,
            trust_threshold=trust_threshold,
            show=show,
        )

    return outputs


def plot_kpi_overview(auction_metrics: dict, save_dir: str, show: bool = False):
    flight = float(auction_metrics.get("flight_energy_J", 0.0))
    comm = float(auction_metrics.get("comm_energy_J", 0.0))
    cpu = float(auction_metrics.get("cpu_energy_J", 0.0))
    total_E = float(auction_metrics.get("total_energy_J", 0.0))

    success_rate = float(auction_metrics.get("success_rate", 0.0))
    total_D = float(auction_metrics.get("total_delay_s", 0.0))
    total_U = float(auction_metrics.get("total_utility", 0.0))
    F = float(auction_metrics.get("composite_score_F", 0.0))

    fig = plt.figure(figsize=(11, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    labels = ["飞行能耗", "通信能耗", "处理能耗"]
    vals = [flight, comm, cpu]
    ax1.bar(labels, vals)
    ax1.set_title("能耗分解 (J)")
    ax1.grid(alpha=0.25, axis="y")
    for i, v in enumerate(vals):
        ax1.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=10)

    ax2 = fig.add_subplot(1, 2, 2)
    labels2 = ["成功率R", "总时延D(s)", "总效用U", "综合评分F"]
    vals2 = [success_rate, total_D, total_U, F]
    ax2.bar(labels2, vals2)
    ax2.set_title("系统指标概览")
    ax2.grid(alpha=0.25, axis="y")
    for i, v in enumerate(vals2):
        # 成功率显示4位，其余显示2位
        txt = f"{v:.4f}" if labels2[i] == "成功率R" else f"{v:.2f}"
        ax2.text(i, v, txt, ha="center", va="bottom", fontsize=10)

    fig.suptitle("拍卖与多目标优化结果概览", fontsize=14, fontweight="bold")
    out = os.path.join(save_dir, "viz_kpi_overview.png")
    _save_show(fig, out, show=show)
    return out


def plot_task_status(log_df: pd.DataFrame, save_dir: str, show: bool = False):
    if "status" not in log_df.columns:
        return None

    counts = log_df["status"].value_counts().sort_values(ascending=False)

    fig = plt.figure(figsize=(10.5, 5.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title("任务成交结果统计（成功/失败原因）", fontsize=13, fontweight="bold")
    ax.set_xlabel("状态")
    ax.set_ylabel("任务数量")
    ax.grid(alpha=0.25, axis="y")
    plt.xticks(rotation=20, ha="right")

    for i, v in enumerate(counts.values):
        ax.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=10)

    out = os.path.join(save_dir, "viz_task_status_counts.png")
    _save_show(fig, out, show=show)
    return out


def plot_tradeoff_scatter(log_df: pd.DataFrame, save_dir: str, show: bool = False):
    # 只取有三指标的记录
    needed = {"energy_task", "delay_task", "utility_total", "status"}
    if not needed.issubset(set(log_df.columns)):
        return None

    # 成功任务作为主视图；如果成功很少则退化为全部
    df = log_df.copy()
    if (df["status"] == "success").sum() >= 10:
        df = df[df["status"] == "success"].copy()

    if len(df) == 0:
        return None

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    sc = ax.scatter(df["delay_task"], df["energy_task"], c=df["utility_total"], s=22, alpha=0.85)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("总效用 U")

    ax.set_title("任务级权衡：时延-能耗（颜色=效用）", fontsize=13, fontweight="bold")
    ax.set_xlabel("任务时延 D (s)")
    ax.set_ylabel("任务能耗 E (J)")
    ax.grid(alpha=0.25)

    out = os.path.join(save_dir, "viz_tradeoff_scatter_EDU.png")
    _save_show(fig, out, show=show)
    return out


def plot_cluster_comparison(log_df: pd.DataFrame, save_dir: str, show: bool = False):
    if "cluster" not in log_df.columns or "status" not in log_df.columns:
        return None

    g = log_df.groupby("cluster")
    total = g.size()
    succ = g.apply(lambda x: (x["status"] == "success").sum())
    success_rate = (succ / total).fillna(0.0)

    # 对成功任务再统计平均能耗/时延/效用
    succ_df = log_df[log_df["status"] == "success"].copy()
    metrics = succ_df.groupby("cluster").agg(
        avg_energy=("energy_task", "mean"),
        avg_delay=("delay_task", "mean"),
        avg_utility=("utility_total", "mean"),
        succ_count=("status", "count"),
    )
    # 对齐所有cluster
    metrics = metrics.reindex(success_rate.index).fillna(0.0)

    clusters = [f"U{int(c)+1}" for c in success_rate.index.tolist()]

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(2, 2, 1)
    ax.bar(clusters, success_rate.values)
    ax.set_title("各UAV成功率 R")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.25, axis="y")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.bar(clusters, metrics["avg_delay"].values)
    ax2.set_title("各UAV平均时延 D (成功任务)")
    ax2.grid(alpha=0.25, axis="y")

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.bar(clusters, metrics["avg_energy"].values)
    ax3.set_title("各UAV平均能耗 E (成功任务)")
    ax3.grid(alpha=0.25, axis="y")

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.bar(clusters, metrics["avg_utility"].values)
    ax4.set_title("各UAV平均效用 U (成功任务)")
    ax4.grid(alpha=0.25, axis="y")

    fig.suptitle("按UAV/区域分解的交易表现对比", fontsize=14, fontweight="bold")
    out = os.path.join(save_dir, "viz_cluster_comparison.png")
    _save_show(fig, out, show=show)
    return out


def plot_seller_dashboard(sellers_df: pd.DataFrame, save_dir: str, show: bool = False):
    needed = {"trust", "malicious", "accepted", "dropped", "failed_deadline"}
    if not needed.issubset(set(sellers_df.columns)):
        return None

    df = sellers_df.copy()
    df["drop_rate"] = df.apply(lambda r: (r["dropped"] / r["accepted"]) if r["accepted"] > 0 else 0.0, axis=1)
    df["deadline_fail_rate"] = df.apply(lambda r: (r["failed_deadline"] / r["accepted"]) if r["accepted"] > 0 else 0.0, axis=1)

    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(1, 2, 1)
    # 信任分布（恶意/正常分开）
    mal = df[df["malicious"] == True]["trust"].values
    norm = df[df["malicious"] == False]["trust"].values
    ax1.hist(norm, bins=10, alpha=0.7, label="正常服务器")
    ax1.hist(mal, bins=10, alpha=0.7, label="恶意服务器")
    ax1.set_title("服务器信任值分布")
    ax1.set_xlabel("trust")
    ax1.set_ylabel("数量")
    ax1.grid(alpha=0.25, axis="y")
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    # 信任 vs 丢弃率
    colors = df["malicious"].apply(lambda x: 1 if x else 0).values
    sc = ax2.scatter(df["trust"], df["drop_rate"], c=colors, s=40, alpha=0.85)
    ax2.set_title("信任值 vs 丢弃率")
    ax2.set_xlabel("trust")
    ax2.set_ylabel("drop_rate (dropped/accepted)")
    ax2.grid(alpha=0.25)

    out = os.path.join(save_dir, "viz_seller_dashboard.png")
    _save_show(fig, out, show=show)
    return out



def plot_weight_sweep(
    path_dic,
    save_dir: str,
    points: int = 11,
    seed: int = 42,
    num_sellers: int = 30,
    transactions_per_seller: int = 4,
    trust_threshold: float = 0.6,
    show: bool = False,
    keep_tmp_files: bool = False,
):
    """扫描权重，展示 E-D-U 的权衡（近似 Pareto 前沿）。
    设计：固定 w_utility=0.34，把剩余0.66在(能耗, 时延)之间按 λ 分配。

    说明：
    - 为保证“横向对比公平”，每个 λ 都用同一个 seed 重新初始化 sellers/tasks。
    - simulate_transactions 会落地写文件；这里默认把临时文件写到 viz_outputs/_sweep_tmp/，
      并在 keep_tmp_files=False 时自动清理，避免工作目录里出现大量临时CSV/TXT。
    """
    lambdas = np.linspace(0.0, 1.0, points)
    records = []

    tmp_dir = os.path.join(save_dir, "_sweep_tmp")
    _ensure_dir(tmp_dir)

    for i, lam in enumerate(lambdas):
        w_u = 0.34
        remain = 1.0 - w_u
        w_e = remain * lam
        w_d = remain * (1.0 - lam)

        prefix = os.path.join(tmp_dir, f"sweep_{i:02d}_lam{int(lam*1000):04d}")
        m = simulate_transactions(
            path_dic=path_dic,
            num_sellers=num_sellers,
            transactions_per_seller=transactions_per_seller,
            trust_threshold=trust_threshold,
            w_energy=float(w_e),
            w_delay=float(w_d),
            w_utility=float(w_u),
            seed=seed,
            verbose=False,
            save_prefix=prefix,
        )

        records.append({
            "lambda": float(lam),
            "w_energy": float(w_e),
            "w_delay": float(w_d),
            "w_utility": float(w_u),
            "E": float(m["total_energy_J"]),
            "D": float(m["total_delay_s"]),
            "U": float(m["total_utility"]),
            "R": float(m["success_rate"]),
            "F": float(m["composite_score_F"]),
        })

        if not keep_tmp_files:
            for k in ("task_log_csv", "summary_txt", "sellers_csv"):
                p = m.get(k)
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass

    df = pd.DataFrame(records)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    sc = ax.scatter(df["D"], df["E"], c=df["lambda"], s=55, alpha=0.9)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("λ（能耗权重占比）")
    ax.set_xlabel("总时延 D (s)")
    ax.set_ylabel("总能耗 E (J)")
    ax.set_title("权重扫描的权衡曲线：D-E 平面（颜色=λ）", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.25)

    out = os.path.join(save_dir, "viz_weight_sweep_DE.png")
    _save_show(fig, out, show=show)

    # 同时保存扫描数据，方便你写论文/画表
    df_path = os.path.join(save_dir, "viz_weight_sweep_metrics.csv")
    df.to_csv(df_path, index=False, encoding="utf-8-sig")
    return {"plot": out, "csv": df_path}


# 主函数
def main():
    print("=" * 60)
    print("           STMTO系统完整模拟")
    print("=" * 60)

    # 记录开始时间
    start_time = time.time()

    # 1. 生成随机点
    print("\n1. 生成设备位置...")
    random.seed(42)
    np.random.seed(42)
    points = generate_points(100)
    print(f"生成了 {len(points)} 个设备位置")

    # 2. 聚类分析
    print("\n2. 进行K-means聚类...")
    cluster_dic, kmeans_model = algo1_with_model(points)

    # 显示聚类结果
    print("\n聚类结果统计:")
    for i in range(len(cluster_dic)):
        print(f"  聚类{i + 1}: {len(cluster_dic[i])}个设备")

    # 3. 绘制工作区域划分图（散点图）
    print("\n3. 生成工作区域划分图（散点图）...")
    centers = plot_work_area_division(points, cluster_dic, kmeans_model)

    # 4. 绘制工作区域统计表格
    print("\n4. 生成工作区域统计表格...")
    stats_df = plot_work_area_stats(cluster_dic, centers)

    # 5. 绘制设备数量分布图
    print("\n5. 生成设备数量分布图...")
    plot_cluster_distribution(cluster_dic)

    # 6. 显示统计信息
    print("\n6. 工作区域统计信息:")
    print(stats_df.to_string(index=False))

    # 7. 路径规划
    print("\n7. 进行路径规划...")
    path_dic, progress_dic = algo2_improved(cluster_dic, G=100, copy_rate=0.5, generations=100, elite_size=15, mutation_rate=0.08, use_two_opt=False, verbose=True, seed_base=1000)

    # 8. 调试收敛数据
    print("\n8. 分析收敛数据...")
    debug_convergence(progress_dic)

    # 9. 可视化路径
    print("\n9. 生成优化路径可视化...")
    plot_improved_paths(path_dic)

    # 10. 显示所有路径总览
    print("\n10. 生成多无人机路径总览图...")
    plot_all_paths_together(path_dic, cluster_dic)

    # 11. 显示收敛曲线
    print("\n11. 显示算法收敛曲线...")
    plot_convergence(progress_dic)

    # Fig.4 口径：系统总轨迹长度收敛曲线
    total_prog = total_trajectory_progress(progress_dic, generations=100)
    plot_total_convergence(total_prog, tick_step=10)

    # 12. 交易模拟
    print("\n12. 开始交易模拟...")
    auction_metrics = simulate_transactions(
        path_dic,
        num_sellers=30,
        transactions_per_seller=4,
        trust_threshold=0.6,
        w_energy=0.33,
        w_delay=0.33,
        w_utility=0.34,
        verbose=False,
        save_prefix="auction_moo"
    )
    total_success = auction_metrics["success_tasks"]
    print("\n多目标拍卖汇总:")
    print(f"  成功率R: {auction_metrics['success_rate']:.4f}")
    print(f"  总能耗E(J): {auction_metrics['total_energy_J']:.2f} (飞行 {auction_metrics['flight_energy_J']:.2f}, 通信 {auction_metrics['comm_energy_J']:.2f}, 处理 {auction_metrics['cpu_energy_J']:.2f})")
    print(f"  总时延D(s): {auction_metrics['total_delay_s']:.2f}")
    print(f"  总效用U: {auction_metrics['total_utility']:.2f}")
    print(f"  综合评分F: {auction_metrics['composite_score_F']:.6f}")
    print(f"  逐任务日志CSV: {auction_metrics['task_log_csv']}")
    print(f"  汇总TXT: {auction_metrics['summary_txt']}")
    print(f"  卖家状态CSV: {auction_metrics['sellers_csv']}")

    # 13. 交易与多目标结果可视化（新增，图片保存在 viz_outputs/）
    print("\n13. 生成交易与多目标优化结果可视化...")
    viz = visualize_auction_results(
        auction_metrics,
        show=False,  # 默认只保存图片不弹窗，避免阻塞；需要弹窗可改 True
        save_dir="viz_outputs",
        do_weight_sweep=True,
        sweep_points=11,
        sweep_seed=42,
        path_dic=path_dic,
        num_sellers=30,
        transactions_per_seller=4,
        trust_threshold=0.6,
    )
    print("  可视化已保存到:", viz.get("save_dir"))

    # 计算总运行时间
    end_time = time.time()
    execution_time = end_time - start_time

    print("\n" + "=" * 60)
    print("模拟完成!")
    print(f"总运行时间: {execution_time:.2f} 秒")
    print(f"总成功交易数: {total_success}")
    print("=" * 60)


# 运行主函数
if __name__ == "__main__":
    main()