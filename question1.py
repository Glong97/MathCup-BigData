# 导入包
import warnings
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
warnings.filterwarnings('ignore')   # 抑制警告输出
pd.set_option('display.max_columns', 100)   # 设置显示所有列

# 导入数据
order_data = pd.read_csv('order.csv')  # 导入订单数据
server_data = pd.read_csv('aunt_data.csv')  # 导入阿姨数据

# b问取前50个订单和前20个阿姨
# order_data = order_data[:50]
# server_data = server_data[:20]

# 数据预处理
order_data = order_data.sort_values('serviceFirstTime', kind='stable')  # 按照服务最早开始服务时间排序（使用稳定的排序算法）
# print(order_data)

# 初始化数据
n = len(order_data)   # 订单数量
n_score = np.zeros(n)   # 每个订单匹配的阿姨的服务分
n_distance = np.zeros(n)    # 每个订单通行距离
n_start_time = np.zeros(n).astype(int)  # 订单服务开始时间
n_interval_time = np.zeros(n)   # 每个订单的间隔时间（单位是小时）

m = len(server_data)    # 阿姨的数量
server_speed = 15000   # 阿姨行使速度
m_end_time = np.zeros(m).astype(int)    # 用于记录每个阿姨上一单服务结束的时间

# 问题1的result1
result1 = pd.DataFrame(columns=["id", "serviceStartTime", "auntId"])
# 问题1b的结果(运行b问把注释去掉)
# result_b = pd.DataFrame(columns=["orderId", "auntId", "order_x",
#                                  "order_y", "serviceStartTime", "serviceEndTime",
#                                  "serviceUnitTime", "distance"])

for i_order in range(n):
    # 处理第i个订单
    order_id = order_data.iloc[i_order, 0]
    serviceFirstTime = order_data.iloc[i_order, 3]  # 服务最早开始时间
    serviceLastTime = order_data.iloc[i_order, 5]  # 服务最晚开始时间
    service_unit_time = order_data.iloc[i_order, 7]  # 服务时长

    order_location = order_data.iloc[i_order, [8, 9]]  # 获取第i个订单的坐标
    order_location = order_location.to_numpy().reshape(1, -1)  # 把订单坐标转化为二维的

    # 每个阿姨上次服务结束的时间
    server_data.loc[:, "last_end_time"] = m_end_time.astype(int)
    # 筛选出可以分配的阿姨的坐标 (阿姨上次服务结束时间大于订单的最晚开始时间就过滤掉)
    server_x_y = server_data.loc[:, ["x", "y"]][server_data["last_end_time"] < serviceLastTime]

    y = [1] * len(server_x_y)  # 因为不做分类，所有点的标签都设为一样
    knn = KNeighborsClassifier(algorithm="kd_tree")
    knn.fit(server_x_y, y)  # 训练模型

    k_neighbors = 130  # k个离得最近的阿姨
    distance, points = knn.kneighbors(order_location, n_neighbors=k_neighbors, return_distance=True)  # 执行算法得到最近10个阿姨

    # 候选阿姨
    candidate = server_data[server_data["id"].isin(points.flatten())]

    # 把候选阿姨到订单的距离添加到新的一列上
    distance_points = np.append(distance.reshape(-1, 1), points.reshape(-1, 1), axis=1)
    distance_points = pd.DataFrame(distance_points, columns=["distance", "id"])
    distance_points.sort_values("id", inplace=True)
    candidate.insert(loc=len(candidate.columns), column="distance", value=distance_points["distance"].values)

    # 候选阿姨通行所花时间（单位秒）
    candidate.loc[:, "passing_time"] = candidate.loc[:, "distance"] / (server_speed / 3600)

    # 候选阿姨的服务最早开始时间（尽量选取最早开始）
    candidate.loc[:, "service_start_time"] = 0
    for i in range(len(candidate.loc[:, "service_start_time"])):
        last_end_time = candidate.iloc[i, 4]  # 阿姨上次服务结束时间
        passing_time = candidate.iloc[i, 6]  # 阿姨通行所花时间
        total_time = last_end_time + passing_time
        # 总时间<=最早开始服务时间
        if total_time <= serviceFirstTime:
            candidate.iloc[i, 7] = serviceFirstTime
            continue
        # 总时间>=最晚开始服务时间
        if total_time >= serviceLastTime:
            candidate.iloc[i, 7] = serviceLastTime
            continue
        # 总时间在最早和最晚之间
        start_time = serviceFirstTime
        while total_time > start_time:
            start_time = start_time + 1800
        candidate.iloc[i, 7] = start_time

    # 候选阿姨服务订单间隔时间 (单位小时)
    candidate.loc[:, "interval_time"] = 0
    for i in range(len(candidate)):
        last_end_time = candidate.iloc[i, 4]  # 阿姨上次服务结束时间
        service_start_time = candidate.iloc[i, 7]  # 阿姨服务开始时间
        if last_end_time == 0:  # 如果阿姨上一次服务结束时间为0，则为第一次服务，订单间隔时间为0.5小时
            candidate.iloc[i, 8] = 0.5
        else:  # 否则为本次订单开始时间减去该阿姨上次服务结束时间
            candidate.iloc[i, 8] = (service_start_time - last_end_time) / 3600

    # 把最后能到阿姨给筛选出来
    candidate = candidate[candidate.interval_time > 0]

    # 候选阿姨最终评分（按题目中所给的公式计算：αA-βB-γC）
    candidate.loc[:, "finalScore"] = candidate.loc[:, ["serviceScore", "distance", "interval_time"]].apply(
        lambda x: 0.78 * x["serviceScore"] - 0.025 * x["distance"] / 1000 - 0.195 * x["interval_time"], axis=1)

    final_server = candidate[candidate.finalScore == candidate.finalScore.max()]
    final_server = final_server[:1]

    n_distance[i_order] = final_server["distance"] / 1000  # 记录该订单的通行距离
    n_score[i_order] = final_server["serviceScore"]  # 记录该订单匹配阿姨的服务分
    n_start_time[i_order] = final_server["service_start_time"]  # 记录该订单服务开始时间
    n_interval_time[i_order] = final_server["interval_time"]  # 记录该订单的间隔时间

    # 更新所分配阿姨订单服务结束时间为本次服务结束时间
    m_end_time[final_server["id"]] = n_start_time[i_order] + (service_unit_time * 60)

    # 把所分配配阿姨的坐标修改为订单的坐标
    server_data.iloc[final_server["id"], 2] = order_data.iloc[i_order, 8]
    server_data.iloc[final_server["id"], 3] = order_data.iloc[i_order, 9]

    result1 = result1.append({"id": order_id, "serviceStartTime": n_start_time[i_order],
                              "auntId": final_server.iloc[0, 0]}, ignore_index=True)
    # result_b = result_b.append({"orderId":order_id, "auntId": final_server.iloc[0, 0],
    #                             "order_x": order_location[0, 0], "order_y": order_location[0, 1],
    #                             "serviceStartTime": n_start_time[i_order],
    #                             "serviceEndTime": m_end_time[final_server["id"]],
    #                             "distance": n_distance[i_order], "serviceUnitTime": service_unit_time},
    #                             ignore_index=True)

# 总体目标值
total_score = 0.78 * n_score.mean() - 0.025 * n_distance.mean() - 0.195 * n_interval_time.mean()
print("A:", n_score.mean())
print("B:", n_distance.mean())
print("C:", n_interval_time.mean())
print("αA-βB-γC：", total_score)

result1.to_csv("result1.csv")
# result_b.to_csv("result_b.csv")

