import random
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
import time

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ===== Fig.4 参数敏感性分析开关（默认开启） =====
RUN_SENSITIVITY_ANALYSIS = True   # 是否生成论文Fig.4风格的敏感性分析图
RUN_FULL_SIMULATION = False       # 是否继续执行后续拍卖/交易等完整流程（可选）

FIG4_GENERATIONS = 100            # 论文Table 1: Iteration times R = 100
FIG4_TRIALS = 3                   # 每组参数重复运行次数，用于取均值
FIG4_G_LIST = (20, 50, 100, 150)  # 初始轨迹数量 |G| 的测试集合
FIG4_W_LIST = (0.0, 0.2, 0.5, 0.7, 0.9)  # 替换概率 w（copy rate）测试集合


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
    for cluster_id in range(5):
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
    for cluster_id in range(5):
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
    for cluster_id in range(5):
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
    """改进的遗传算法类"""

    def __init__(self, points, pop_size=150, elite_size=20, mutation_rate=0.05, generations=1000, copy_rate=0.5, use_two_opt=True):
        self.points = np.array(points, dtype=float)
        # 预计算距离矩阵：显著加速适应度评估（对敏感性分析的多次重复运行非常关键）
        diff = self.points[:, None, :] - self.points[None, :, :]
        self.dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
        self.use_two_opt = use_two_opt
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.copy_rate = copy_rate  # 论文Algorithm 2中的copy rate w（替换概率）
        self.num_points = len(points)

    def create_individual(self):
        """创建个体（随机路径）"""
        individual = list(range(self.num_points))
        random.shuffle(individual)
        return individual

    def initial_population(self):
        """初始化种群"""
        return [self.create_individual() for _ in range(self.pop_size)]

    def path_length(self, individual):
        """使用预计算距离矩阵计算闭环路径长度（比逐点sqrt循环更快）"""
        if len(individual) <= 1:
            return 0.0
        idx = np.array(individual, dtype=int)
        return float(self.dist_matrix[idx, np.roll(idx, -1)].sum())

    def rank_individuals(self, population):
        """对种群排序（适应度从好到坏）"""
        fitness = []
        for ind in population:
            # 确保每个个体都是有效的路径
            if len(set(ind)) != self.num_points:
                # 如果路径无效，创建一个新的随机个体
                ind = self.create_individual()
            fitness.append((ind, self.path_length(ind)))

        return sorted(fitness, key=lambda x: x[1])

    def tournament_selection(self, ranked_population, tournament_size=5):
        """锦标赛选择"""
        selected = []

        # 精英选择：直接保留最优的前elite_size个个体
        for i in range(self.elite_size):
            selected.append(ranked_population[i][0])

        # 锦标赛选择剩余个体
        for _ in range(self.pop_size - self.elite_size):
            # 随机选择tournament_size个个体进行比赛
            tournament = random.sample(ranked_population, tournament_size)
            # 选择比赛中适应度最好的个体
            winner = min(tournament, key=lambda x: x[1])
            selected.append(winner[0])

        return selected

    def ordered_crossover(self, parent1, parent2):
        """顺序交叉（OX）"""
        if len(parent1) != len(parent2):
            return parent1  # 安全回退

        size = len(parent1)
        child = [-1] * size

        # 随机选择交叉点
        start, end = sorted(random.sample(range(size), 2))

        # 从parent1复制片段
        child[start:end + 1] = parent1[start:end + 1]

        # 从parent2填充剩余位置
        current_pos = (end + 1) % size
        for gene in parent2:
            if gene not in child:
                while child[current_pos] != -1:
                    current_pos = (current_pos + 1) % size
                child[current_pos] = gene

        return child

    def swap_mutation(self, individual):
        """交换变异"""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                j = random.randint(0, len(mutated) - 1)
                mutated[i], mutated[j] = mutated[j], mutated[i]
        return mutated

    def inversion_mutation(self, individual):
        """倒位变异"""
        if random.random() < self.mutation_rate:
            start, end = sorted(random.sample(range(len(individual)), 2))
            individual = individual[:start] + individual[start:end + 1][::-1] + individual[end + 1:]
        return individual

    def segment_swap(self, individual):
        """四断点分段交换：模拟论文 Algorithm 2 Step3 的 Swap 操作，用于增加多样性"""
        if len(individual) < 6:
            return individual
        a, b, c, d = sorted(random.sample(range(1, len(individual)), 4))
        seg0 = individual[:a]
        seg_ab = individual[a:b]
        seg_bc = individual[b:c]
        seg_cd = individual[c:d]
        seg_d = individual[d:]
        # 交换 seg_ab 与 seg_cd：seg0 + seg_cd + seg_bc + seg_ab + seg_d
        return seg0 + seg_cd + seg_bc + seg_ab + seg_d

    def two_opt(self, individual):
        """2-opt局部优化"""
        improved = True
        best_distance = self.path_length(individual)

        while improved:
            improved = False
            for i in range(len(individual) - 1):
                for j in range(i + 2, len(individual)):
                    if j == len(individual) - 1 and i == 0:
                        continue  # 避免首尾特殊情况

                    # 创建新路径：反转i到j之间的片段
                    new_individual = individual.copy()
                    new_individual[i:j + 1] = individual[i:j + 1][::-1]

                    new_distance = self.path_length(new_individual)

                    if new_distance < best_distance:
                        individual = new_individual
                        best_distance = new_distance
                        improved = True
                        break
                if improved:
                    break

        return individual

    def evolve(self, verbose=True):
        """进化过程"""
        population = self.initial_population()
        best_individual = None
        best_fitness = float('inf')
        progress = []

        if verbose:
            print(f"开始进化，种群大小: {self.pop_size}, 迭代次数: {self.generations}")

        for generation in range(self.generations):
            # 评估种群
            ranked_population = self.rank_individuals(population)
            # 复制替换机制（copy rate w）：将差解以概率w替换为优解并做分段交换，模拟论文Algorithm 2 Step3
            if self.copy_rate is not None and self.copy_rate > 0:
                avg_fitness_tmp = np.mean([fit for _, fit in ranked_population])
                best_ref = ranked_population[0][0]
                new_population = []
                for ind, fit in ranked_population:
                    if fit > avg_fitness_tmp and random.random() < self.copy_rate:
                        replaced = best_ref.copy()
                        replaced = self.segment_swap(replaced)
                        new_population.append(replaced)
                    else:
                        new_population.append(ind)
                population = new_population
                ranked_population = self.rank_individuals(population)


            # 更新最佳个体
            current_best_individual, current_best_fitness = ranked_population[0]

            if current_best_fitness < best_fitness:
                best_individual = current_best_individual.copy()
                best_fitness = current_best_fitness
                # 对最佳个体进行局部优化
                if self.use_two_opt:
                    best_individual = self.two_opt(best_individual)
                best_fitness = self.path_length(best_individual)
            progress.append(best_fitness)

            # 每100代显示进度
            if verbose and generation % 100 == 0:
                avg_fitness = np.mean([fit for _, fit in ranked_population])
                print(f"第{generation}代: 最佳={best_fitness:.2f}, 平均={avg_fitness:.2f}")

            # 选择
            selected = self.tournament_selection(ranked_population)

            # 交叉和变异生成下一代
            next_population = []

            # 保留精英
            next_population.extend(selected[:self.elite_size])

            # 交叉生成后代
            while len(next_population) < self.pop_size:
                parent1, parent2 = random.sample(selected, 2)
                child = self.ordered_crossover(parent1, parent2)

                # 应用两种变异
                child = self.swap_mutation(child)
                child = self.inversion_mutation(child)

                next_population.append(child)

            population = next_population[:self.pop_size]

        if verbose:
            print(f"进化完成，最终路径长度: {best_fitness:.2f}")
        return best_individual, progress


def algo2_improved(cluster_dic, G=100, copy_rate=0.5, generations=100, elite_size=15, mutation_rate=0.08, use_two_opt=True, verbose=True):
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
        ga = ImprovedGA(
            points=cluster_points,
            pop_size=G,  # 初始轨迹数量 |G|
            elite_size=elite_size,
            mutation_rate=mutation_rate,
            generations=generations,
            copy_rate=copy_rate,  # 替换概率 w（copy rate）
            use_two_opt=use_two_opt
        )

        best_path, progress = ga.evolve(verbose=verbose)
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
    cluster_names = {0: 'U1', 1: 'U2', 2: 'U3', 3: 'U4', 4: 'U5'}

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
    """指数滑动平均(EMA)平滑，用于让收敛曲线更平滑（仅影响展示，不改变算法结果）"""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    smoothed = np.empty_like(values)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def plot_convergence(progress_dic, smooth=True, alpha=0.15, show_raw=True, save_path=None, show=True):
    """绘制收敛曲线（默认平滑显示）"""
    plt.figure(figsize=(12, 8))

    # 颜色池：数量不够时循环使用
    color_pool = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']

    # 使得U编号随簇ID动态生成（避免固定死5架无人机）
    sorted_ids = sorted(progress_dic.keys())
    cluster_names = {cid: f'U{cid + 1}' for cid in sorted_ids}

    for idx, cluster_id in enumerate(sorted_ids):
        progress = progress_dic[cluster_id]
        if len(progress) <= 1:
            continue

        raw = np.asarray(progress, dtype=float)
        color = color_pool[idx % len(color_pool)]

        if smooth:
            smooth_series = smooth_ema(raw, alpha=alpha)
            if show_raw:
                # 原始曲线（淡）
                plt.plot(raw, color=color, linewidth=1, alpha=0.25)
            # 平滑曲线（主展示）
            plt.plot(smooth_series, color=color, label=f'{cluster_names[cluster_id]}(EMA)', linewidth=2.5, alpha=0.9)
            # 标记起点/终点（使用平滑值更美观）
            plt.scatter(0, smooth_series[0], color=color, s=50, zorder=5)
            plt.scatter(len(smooth_series) - 1, smooth_series[-1], color=color, s=50, zorder=5)
        else:
            # 不平滑：保持原样
            plt.plot(raw, color=color, label=f'{cluster_names[cluster_id]}', linewidth=2, alpha=0.8)
            plt.scatter(0, raw[0], color=color, s=50, zorder=5)
            plt.scatter(len(raw) - 1, raw[-1], color=color, s=50, zorder=5)

    plt.title('遗传算法收敛曲线', fontsize=15)
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('路径长度', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()



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


def simulate_transactions(path_dic, num_sellers=30, transactions_per_seller=4):
    """交易模拟函数"""
    transaction_num = [transactions_per_seller for _ in range(num_sellers)]
    cluster_names = {0: 'U1', 1: 'U2', 2: 'U3', 3: 'U4', 4: 'U5'}

    total_successful_transactions = 0

    for cluster_idx in range(5):  # 遍历5个聚类
        print(f"\n{'=' * 50}")
        print(f"第{cluster_idx + 1}组交易 ({cluster_names[cluster_idx]})")
        print(f"{'=' * 50}")

        if cluster_idx not in path_dic:
            print("该聚类无路径数据，跳过")
            continue

        path_points = path_dic[cluster_idx]
        # 去除重复的起点（路径是闭环，起点和终点相同）
        unique_points = path_points[:-1] if len(path_points) > 1 else path_points

        successful_in_cluster = 0

        for point_idx, point in enumerate(unique_points):
            print(f"\n买家 {point_idx + 1} 坐标: {point}")
            buyer_price = random.uniform(10, 10000)
            print(f"买家出价: {buyer_price:.2f}")

            success = False
            sell_prices = []

            # 第一轮拍卖
            for seller_idx in range(num_sellers):
                if transaction_num[seller_idx] == 0:
                    continue  # 卖家额度已用完

                sell_price = random.uniform(10, 10000)
                sell_prices.append((seller_idx, sell_price))

                if buyer_price >= sell_price:
                    print(f"  第一轮: 与卖家{seller_idx}匹配成功 (卖家价格: {sell_price:.2f})")
                    transaction_num[seller_idx] -= 1
                    success = True
                    successful_in_cluster += 1
                    break

            # 第二轮拍卖（如果第一轮失败）
            if not success and sell_prices:
                # 买家调整价格
                buyer_price_adjusted = random.uniform(10, 10000)
                print(f"第一轮失败，买家调整价格至: {buyer_price_adjusted:.2f}")

                for seller_idx, sell_price in sell_prices:
                    if transaction_num[seller_idx] > 0 and buyer_price_adjusted >= sell_price:
                        print(f"  第二轮: 与卖家{seller_idx}匹配成功 (卖家价格: {sell_price:.2f})")
                        transaction_num[seller_idx] -= 1
                        success = True
                        successful_in_cluster += 1
                        break

            if not success:
                print("  交易失败: 买卖双方价格不匹配")

        print(f"\n本组合计成功匹配: {successful_in_cluster} 个交易")
        total_successful_transactions += successful_in_cluster

    print(f"\n{'=' * 50}")
    print(f"所有交易完成!")
    print(f"总成功交易数: {total_successful_transactions}")
    print(f"{'=' * 50}")

    return total_successful_transactions


def plot_all_paths_together(path_dic, cluster_dic):
    """在同一张图上绘制所有路径"""
    plt.figure(figsize=(12, 10))
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    cluster_names = {0: 'U1', 1: 'U2', 2: 'U3', 3: 'U4', 4: 'U5'}

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
def _pad_progress(progress, generations):
    """确保progress长度等于generations（用于汇总多UAV总轨迹长度）"""
    if progress is None or len(progress) == 0:
        return [0.0] * generations
    if len(progress) >= generations:
        return [float(x) for x in progress[:generations]]
    last = float(progress[-1])
    return [float(x) for x in progress] + [last] * (generations - len(progress))


def total_trajectory_progress(progress_dic, generations):
    """把每个UAV的收敛曲线相加，得到系统总轨迹长度随迭代的变化"""
    total = np.zeros(generations, dtype=float)
    for _, prog in progress_dic.items():
        total += np.array(_pad_progress(prog, generations), dtype=float)
    return total


def run_algorithm2_once(cluster_dic, G, w, generations=100, seed=None):
    """在同一拓扑(cluster_dic)下，运行一次Algorithm2(轨迹优化)并返回总轨迹收敛曲线"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    t0 = time.time()
    _, progress_dic = algo2_improved(
        cluster_dic,
        G=G,
        copy_rate=w,
        generations=generations,
        elite_size=15,
        mutation_rate=0.08,
        use_two_opt=False,  # 敏感性分析阶段关闭two-opt以对齐论文Algorithm2并加速
        verbose=False
    )
    t1 = time.time()
    total_prog = total_trajectory_progress(progress_dic, generations)
    return {
        "progress_dic": progress_dic,
        "total_progress": total_prog,
        "final_total_length": float(total_prog[-1]),
        "runtime_sec": float(t1 - t0),
    }


def plot_sensitivity_curves(curves_dict, title, save_path=None, show=True):
    """绘制多条收敛曲线（用于Fig.4(b)/(c)）"""
    plt.figure(figsize=(12, 8))
    for label, series in curves_dict.items():
        plt.plot(series, linewidth=2, alpha=0.9, label=str(label))
    plt.title(title, fontsize=15)
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('总轨迹长度', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def plot_sensitivity_final(x_vals, y_vals, title, xlabel, save_path=None, show=True):
    """绘制参数-最终值关系（辅助敏感性分析）"""
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, marker='o', linewidth=2, alpha=0.9)
    plt.title(title, fontsize=15)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('最终总轨迹长度', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def plot_sensitivity_runtime(x_vals, y_vals, title, xlabel, save_path=None, show=True):
    """绘制参数-平均运行时间关系（论文也强调G过大会降低效率）"""
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, marker='o', linewidth=2, alpha=0.9)
    plt.title(title, fontsize=15)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('平均运行时间（秒）', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def sensitivity_analysis_figure4(
    cluster_dic,
    save_dir="fig4_sensitivity",
    generations=100,
    trials=3,
    G_list=(20, 50, 100, 150),
    w_list=(0.0, 0.2, 0.5, 0.7, 0.9),
    baseline_G=100,
    baseline_w=0.5,
    seed_base=123
):
    """
    复现论文Fig.4的三类图：
    (a) 基准参数下的收敛曲线（按UAV分别画）
    (b) 初始轨迹数量 |G| 的敏感性（多条收敛曲线 + 最终值/运行时间）
    (c) 替换概率 w 的敏感性（多条收敛曲线 + 最终值/运行时间）

    注意：这里的“替换概率w”在实现上对应论文Algorithm2 Step3的copy rate机制：
    对差解（高于平均长度）以概率w替换为优解拷贝，并执行分段交换以保持多样性。
    """
    os.makedirs(save_dir, exist_ok=True)

    # -------- baseline convergence (per UAV) --------
    baseline_res = run_algorithm2_once(
        cluster_dic, G=baseline_G, w=baseline_w, generations=generations, seed=seed_base
    )
    fig4a_path = os.path.join(save_dir, "Fig4a_convergence_per_uav.png")
    plot_convergence(
        baseline_res["progress_dic"],
        smooth=True,
        alpha=0.15,
        show_raw=True,
        save_path=fig4a_path,
        show=False
    )

    # --------  sensitivity on |G| --------
    curves_G = {}
    finals_G = []
    runtimes_G = []
    for G in G_list:
        trial_progress = []
        trial_finals = []
        trial_times = []
        for t in range(trials):
            res = run_algorithm2_once(
                cluster_dic, G=G, w=baseline_w, generations=generations, seed=seed_base + 1000 + 10 * G + t
            )
            trial_progress.append(res["total_progress"])
            trial_finals.append(res["final_total_length"])
            trial_times.append(res["runtime_sec"])
        avg_prog = np.mean(np.vstack(trial_progress), axis=0)
        curves_G[f"G={G}"] = avg_prog
        finals_G.append(float(np.mean(trial_finals)))
        runtimes_G.append(float(np.mean(trial_times)))

    fig4b_path = os.path.join(save_dir, "Fig4b_effect_of_G.png")
    plot_sensitivity_curves(curves_G, title="初始轨迹数量 |G| 对优化收敛的影响", save_path=fig4b_path, show=False)

    fig4b_final_path = os.path.join(save_dir, "Fig4b_G_vs_final.png")
    plot_sensitivity_final(list(G_list), finals_G, title="|G| 与最终总轨迹长度（越小越好）", xlabel="初始轨迹数量 |G|", save_path=fig4b_final_path, show=False)

    fig4b_time_path = os.path.join(save_dir, "Fig4b_G_vs_runtime.png")
    plot_sensitivity_runtime(list(G_list), runtimes_G, title="|G| 与平均运行时间", xlabel="初始轨迹数量 |G|", save_path=fig4b_time_path, show=False)

    # -------- Fig.4(c): sensitivity on w --------
    curves_w = {}
    finals_w = []
    runtimes_w = []
    for w in w_list:
        trial_progress = []
        trial_finals = []
        trial_times = []
        for t in range(trials):
            res = run_algorithm2_once(
                cluster_dic, G=baseline_G, w=w, generations=generations, seed=seed_base + 2000 + int(100 * w) * 10 + t
            )
            trial_progress.append(res["total_progress"])
            trial_finals.append(res["final_total_length"])
            trial_times.append(res["runtime_sec"])
        avg_prog = np.mean(np.vstack(trial_progress), axis=0)
        curves_w[f"w={w}"] = avg_prog
        finals_w.append(float(np.mean(trial_finals)))
        runtimes_w.append(float(np.mean(trial_times)))

    fig4c_path = os.path.join(save_dir, "Fig4c_effect_of_w.png")
    plot_sensitivity_curves(curves_w, title=" 替换概率 w（copy rate）对优化收敛的影响", save_path=fig4c_path, show=False)

    fig4c_final_path = os.path.join(save_dir, "c_w_vs_final.png")
    plot_sensitivity_final(list(w_list), finals_w, title="w 与最终总轨迹长度（越小越好）", xlabel="替换概率 w（copy rate）", save_path=fig4c_final_path, show=False)

    fig4c_time_path = os.path.join(save_dir, "c_w_vs_runtime.png")
    plot_sensitivity_runtime(list(w_list), runtimes_w, title="w 与平均运行时间", xlabel="替换概率 w（copy rate）", save_path=fig4c_time_path, show=False)

    # 汇总表
    df_G = pd.DataFrame({"G": list(G_list), "final_total_length": finals_G, "avg_runtime_sec": runtimes_G})
    df_w = pd.DataFrame({"w": list(w_list), "final_total_length": finals_w, "avg_runtime_sec": runtimes_w})
    df_G.to_csv(os.path.join(save_dir, "sensitivity_G.csv"), index=False, encoding="utf-8-sig")
    df_w.to_csv(os.path.join(save_dir, "sensitivity_w.csv"), index=False, encoding="utf-8-sig")

    return {
        "baseline": baseline_res,
        "df_G": df_G,
        "df_w": df_w,
        "fig_paths": {
            "Fig4a": fig4a_path,
            "Fig4b": fig4b_path,
            "Fig4b_final": fig4b_final_path,
            "Fig4b_runtime": fig4b_time_path,
            "Fig4c": fig4c_path,
            "Fig4c_final": fig4c_final_path,
            "Fig4c_runtime": fig4c_time_path,
        }
    }

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# 主函数
def main():
    print("=" * 60)
    print("           STMTO系统完整模拟")
    print("=" * 60)

    # 记录开始时间
    start_time = time.time()

    # 1. 生成随机点
    print("\n1. 生成设备位置...")
    points = generate_points(100)
    print(f"生成了 {len(points)} 个设备位置")

    # 2. 聚类分析
    print("\n2. 进行K-means聚类...")
    cluster_dic, kmeans_model = algo1_with_model(points)

    # 2.1 参数敏感性分析（论文Fig.4：|G| 与 copy rate w）
    if RUN_SENSITIVITY_ANALYSIS:
        print("\n3. 进行Algorithm 2参数敏感性分析...")
        res = sensitivity_analysis_figure4(
            cluster_dic,
            save_dir="fig4_sensitivity",
            generations=FIG4_GENERATIONS,
            trials=FIG4_TRIALS,
            G_list=FIG4_G_LIST,
            w_list=FIG4_W_LIST,
            baseline_G=100,
            baseline_w=0.5,
            seed_base=123
        )
        print("敏感性分析完成，图像与CSV已保存到目录: fig4_sensitivity")
        if not RUN_FULL_SIMULATION:
            return


    # 显示聚类结果
    print("\n聚类结果统计:")
    for i in range(5):
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
    path_dic, progress_dic = algo2_improved(cluster_dic)

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

    # 12. 交易模拟
    print("\n12. 开始交易模拟...")
    total_success = simulate_transactions(path_dic)

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