# 导入包
import warnings
import numpy as np
import pandas as pd
import time

from sklearn.neighbors import KNeighborsClassifier
warnings.filterwarnings('ignore')   # 抑制警告输出
pd.set_option('display.max_columns', 100)   # 设置显示所有列

# 导入数据
order_data = pd.read_csv('order.csv')  # 导入订单数据
server_data = pd.read_csv('aunt_data.csv')  # 导入阿姨数据

# 初始化数据
n = len(order_data)   # 订单数量
n_score = np.zeros(n)   # 每个订单匹配的阿姨的服务分
n_distance = np.zeros(n)    # 每个订单通行距离
n_start_time = np.zeros(n).astype(int)  # 订单服务开始时间
n_interval_time = np.zeros(n)   # 每个订单的间隔时间（单位是小时）

m = len(server_data)    # 阿姨的数量
server_speed = 15000   # 阿姨行使速度
m_end_time = np.zeros(m).astype(int)    # 用于记录每个阿姨上一单服务结束的时间

# 初始化当前时间
current_time = 1661423400   # 第一次处理订单的时间：2022-08-25 18:30:00（第一个订单创建的时间 + 15分钟）
time_step = 30 * 60     # 每次处理的时间间隔30分钟
timeArray = time.localtime(current_time)
otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
print("当前时间：", otherStyleTime)

# 压单所满足的时间（serviceFirstTime - current_time所要大于的时间）
retain_time = 2 * 60 * 60   # 2小时

# 最终决策结果
result21 = pd.DataFrame(columns=["id", "serviceStartTime", "serviceStartData", "auntId"])
# 每次决策结果
result22 = pd.DataFrame(columns=["currentTime", "currentData", "id", "serviceStartTime", "serviceStartData", "auntId", "retainable"])

# 当还有订单未处理完就一直循环
while len(order_data) > 0:
    # 当前时间的日期格式
    timeArray = time.localtime(current_time)
    currentData = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)

    # 筛选出当前半小时内产生的订单和之前压单的订单
    order_set = order_data.loc[order_data.iloc[:, 1] <= current_time]
    # 把该半小时产生的订单以及压单的订单全部处理一遍
    for i_order in range(len(order_set)):
        # 处理第i个订单
        order_id = order_set.iloc[i_order, 0]  # 订单id
        order_id = int(order_id)
        serviceFirstTime = order_set.iloc[i_order, 3]  # 服务最早开始时间
        # 满足压单条件就进行压单处理
        if (serviceFirstTime - current_time) > retain_time:
            # print("serviceFirstTime - current_time:", serviceFirstTime - current_time)
            # 记录本次决策结果
            result22 = result22.append({"currentTime": current_time, "currentData": currentData, "id": order_id,
                                        "serviceStartTime": -1, "serviceStartData": "00", "auntId": -1,
                                        "retainable": 1}, ignore_index=True)
            continue  # 直接处理下一个订单

        # 不满足压单条件，分配一个得分最高的阿姨
        serviceLastTime = order_set.iloc[i_order, 5]  # 服务最晚开始时间
        service_unit_time = order_set.iloc[i_order, 7]  # 服务时长
        order_location = order_set.iloc[i_order, [8, 9]]  # 获取第i个订单的坐标
        order_location = order_location.to_numpy().reshape(1, -1)  # 把订单坐标转化为二维的

        # 每个阿姨上次服务结束的时间
        server_data.loc[:, "last_end_time"] = m_end_time.astype(int)
        # 筛选出可以分配的阿姨的坐标 (阿姨上次服务结束时间大于订单的最晚开始时间就过滤掉)
        server_x_y = server_data.loc[:, ["x", "y"]][server_data["last_end_time"] < serviceLastTime]
        y = [1] * len(server_x_y)  # 因为不做分类，所有点的标签都设为一样
        knn = KNeighborsClassifier(algorithm="kd_tree")
        knn.fit(server_x_y, y)  # 训练模型

        k_neighbors = 10  # k个离得最近的阿姨
        distance, points = knn.kneighbors(order_location, n_neighbors=k_neighbors,
                                          return_distance=True)  # 执行算法得到最近10个阿姨

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

        n_distance[order_id] = final_server["distance"] / 1000  # 记录该订单的通行距离
        n_score[order_id] = final_server["serviceScore"]  # 记录该订单匹配阿姨的服务分
        n_start_time[order_id] = final_server["service_start_time"]  # 记录该订单服务开始时间
        n_interval_time[order_id] = final_server["interval_time"]  # 记录该订单的间隔时间

        # 服务开始时间的日期格式
        timeArray2 = time.localtime(n_start_time[order_id])
        serviceStartData = time.strftime("%Y-%m-%d %H:%M:%S", timeArray2)

        # 更新所分配阿姨订单服务结束时间为本次服务结束时间
        m_end_time[final_server["id"]] = n_start_time[order_id] + (service_unit_time * 60)

        # 把所分配配阿姨的坐标修改为订单的坐标
        server_data.iloc[final_server["id"], 2] = order_set.iloc[i_order, 8]
        server_data.iloc[final_server["id"], 3] = order_set.iloc[i_order, 9]

        # 记录最终决策结果
        result21 = result21.append({"id": order_id, "serviceStartTime": final_server["service_start_time"].values[0],
                                    "serviceStartData": serviceStartData, "auntId": final_server.iloc[0, 0]},
                                   ignore_index=True)
        # 记录每次决策结果
        result22 = result22.append({"currentTime": current_time, "currentData": currentData, "id": order_id,
                                    "serviceStartTime": final_server["service_start_time"].values[0],
                                    "serviceStartData": serviceStartData, "auntId": final_server.iloc[0, 0],
                                    "retainable": 0}, ignore_index=True)
        # 把处理好的订单移除
        order_data = order_data.drop(order_data[order_data.id == order_id].index)

    # 更新时间
    current_time = current_time + time_step
    print("当前时间：", current_time)
    print("当前订单数量：", len(order_data))

# 总体目标值
total_score = 0.78 * n_score.mean() - 0.025 * n_distance.mean() - 0.195 * n_interval_time.mean()
print("A:", n_score.mean())
print("B:", n_distance.mean())
print("C:", n_interval_time.mean())
print("αA-βB-γC：", total_score)

# 输出最终结果到文件
result21.to_csv("result21.csv")
result22.to_csv("result22.csv")
