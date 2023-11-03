import os
import math
import time


total_orbits_cnt = 72
total_planes_cnt = 22
total_satellites_cnt = total_orbits_cnt * total_planes_cnt
max_visible_times = 120 # 预测未来21天，最多不超过120次

# 实验设置中的常量
Tansmission_power = 10 # 单位W
Gain_Antenna = 5 # 天线增益，即放大倍数
Carrier_frequency = 2.4 * math.pow(10, 9)  # 单位Hz
Noise_temperature = 354.81  # 单位K
Boltzmann_constant = 1.38064852 * math.pow(10, -23)  # 单位J/K
Bandwidth_up = 2.1 * math.pow(2, 30) # 单位Hz 2.1GHz
Bandwidth_down = 1.3 / total_orbits_cnt * math.pow(2, 30) # 单位Hz 1.3GHz,按照论文 先考虑N=72 均分
Bandwidth_sat2sat = 33 * math.pow(2, 30) # 单位Hz 33GHz
Spectral_efficiency = 1 # 通信链路的频谱效率 单位bit/s/Hz
Speed_of_light = 299792.458  # 单位km/s

fsize = os.path.getsize('./save/w_glob.pkl')
Size_of_model = fsize * 8 # 模型大小，单位bit
sat_local_epochs = 20

# 训练开始时间
stk_train_start1 = "2023-06-01 00:00:00"
stk_train_start11 = time.strptime(stk_train_start1, "%Y-%m-%d %H:%M:%S")
stk_train_start_stamp = time.mktime(stk_train_start11)

# 训练结束时间
stk_train_end1 = "2023-06-22 00:00:00"
stk_train_end11 = time.strptime(stk_train_end1, "%Y-%m-%d %H:%M:%S")
stk_train_end_stamp = time.mktime(stk_train_end11)


class Record:
    def __init__(self):
        self.start_time = time.time()
        self.stop_time = time.time()
        self.start_tstamp = 0.0 # float
        self.stop_tstamp = 0.0
        self.delta_tstamp = 0.0 # 上述两者的差值,单位s
        self.range = 0.0 # 过顶时，卫星与地面站的距离，单位km
        self.com_time_up = 0.0  # 上行链路通信时延，接收模型
        self.com_time_down = 0.0 # 下行链路通信时延,传递模型给GS
        

class Satellite:
    def __init__(self):
        self.orbit_id = 0
        self.plane_id = 0
        self.total_visible_times = 0
        self.records = [Record() for _ in range(max_visible_times)]
        self.visible_idx = 0 # 当前过顶下标索引
        self.visited = 0 # 当前索引的可见期是否被访问过
        self.backward_neighbor = -1
        self.forward_neighbor = -1
        self.backward_range = 0.0
        self.forward_range = 0.0

        self.received = 0  # 是否收到模型
        self.time_received = 0  # 收到模型的时间
        self.time_trained = 0  # 模型训练完成的时间
        self.time_get_model = 0  # 聚合时收到参数的时间
        self.time_send_model = 0  # 聚合时发送参数的时间

        self.prev = None  # 前继节点
        self.next = None  # 后继节点
        self.next2 = None  # 后继节点2，只限于根节点使用


class Orbit:
    def __init__(self):
        self.orbit_id = -1
        self.satellites = [Satellite() for i in range(total_planes_cnt)]
        self.source_node_id = -1
        self.sink_node_id = -1
        self.source_node_received = 0


class Ground_Station:
    def __init__(self):
        self.orbit_model_sended = set([]) # GS已经发送模型的轨道
        self.orbit_model_received = set([]) # 已经收到局部模型的轨道


class sat_list:
    # 初始化
    def __init__(self):
        self._head = None

    def append(self, new_node):
        if self._head is None:
            # 头部结点指针修改为新结点
            self._head = new_node
        else:
            current = self._head
            while current.next is not None:
                current = current.next
            current.next = new_node
            new_node.prev = current

    def get_tail(self):
        current = self._head
        while current.next is not None:
            current = current.next
        return current

    def get_head(self):
        return self._head


def pathloss_K_GS(dis):
    res = 4 * math.pi * dis * Carrier_frequency / Speed_of_light
    res = res * res
    return res

# 相邻卫星之间才可以通信，路由必须是一跳一条的
def pathloss_K_K(src_sat, dest_sat):
    dis = 0.0
    if src_sat.forward_neighbor == dest_sat.plane_id:
        dis = src_sat.forward_range
    elif src_sat.backward_neighbor == dest_sat.plane_id:
        dis = src_sat.backward_range
    res = 4 * math.pi * dis * Carrier_frequency / Speed_of_light
    res = res * res
    return res


def cal_rate_adj_sat2sat(src_sat, dest_sat):   
    SNR_sat2sat = (Tansmission_power*Gain_Antenna*Gain_Antenna) / (Boltzmann_constant*Noise_temperature*Bandwidth_sat2sat* \
                    pathloss_K_K(src_sat, dest_sat) +float("1e-8"))

    rate_sat2sat = Bandwidth_sat2sat *  math.log(1+SNR_sat2sat, 2)
    return rate_sat2sat


def cal_sat_train():
    ck = 1000
    fk = math.pow(10, 9)
    filePath_images = '../data/emnist/EMNIST/raw/emnist-bymerge-train-images-idx3-ubyte'
    filePath_labels = '../data/emnist/EMNIST/raw/emnist-bymerge-train-labels-idx1-ubyte'
    fsize = os.path.getsize(filePath_images) + os.path.getsize(filePath_labels)
    train_time = (fsize * 8 / total_satellites_cnt) * ck / fk
    return train_time * sat_local_epochs
    # return 60


sat_train_time = cal_sat_train()
satellites = [Satellite() for i in range(total_satellites_cnt+1)]
orbits = [Orbit() for i in range(total_orbits_cnt)]
MoRolla = Ground_Station()


# 传播 聚合 交接模型阶段，路由都是最短路径
def sat_to_sat_delay(orbit, src, dest):
    dest_received = orbit.satellites[dest].received
    total_delay = 0.0
    if ((src < dest) & ((dest-src) >= (total_planes_cnt-dest+src))) | ((src > dest) & ((src-dest) <= (total_planes_cnt-src+dest))):  # -1方向传播，即为list1，list1共有6个节点
        while src != dest:
            send_time = 5 - (src - orbit.sink_node_id) % total_planes_cnt
            if dest_received == 1:
                #print(send_time)
                total_delay += ((orbit.satellites[src].forward_range / Speed_of_light) + (Size_of_model * send_time / cal_rate_adj_sat2sat(orbit.satellites[src], orbit.satellites[orbit.satellites[src].forward_neighbor])))
            else:
                total_delay += ((orbit.satellites[src].forward_range / Speed_of_light) + (Size_of_model / cal_rate_adj_sat2sat(orbit.satellites[src], orbit.satellites[orbit.satellites[src].forward_neighbor])))
            src = orbit.satellites[src].forward_neighbor
    elif ((src < dest) & ((dest-src) < (total_planes_cnt-dest+src))) | ((src > dest) & ((src-dest) > (total_planes_cnt-src+dest))):  # +1方向传播，即为list2，list2共有5个节点
        while src != dest:
            send_time = 4 - (orbit.sink_node_id - src) % total_planes_cnt
            if dest_received == 1:
                #print(send_time)
                total_delay += ((orbit.satellites[src].backward_range / Speed_of_light) + (Size_of_model * send_time / cal_rate_adj_sat2sat(orbit.satellites[src], orbit.satellites[orbit.satellites[src].backward_neighbor])))
            else:
                total_delay += ((orbit.satellites[src].backward_range / Speed_of_light) + (Size_of_model / cal_rate_adj_sat2sat(orbit.satellites[src], orbit.satellites[orbit.satellites[src].backward_neighbor])))
            src = orbit.satellites[src].backward_neighbor
    return total_delay


def vir_src_propagation(orbit):
    sat_source = orbit.satellites[orbit.source_node_id]
    com_times = math.ceil(total_planes_cnt / 2)  # 双向传播，所以传播次数为卫星数除以2

    sat_source.source_neighbor = -1
    # 模型传播、训练过程
    for i in range(com_times):
        idx = (sat_source.plane_id + i + 1) % total_planes_cnt       
        orbit.satellites[idx].source_neighbor = (sat_source.plane_id + i) % total_planes_cnt
        
        idx = (sat_source.plane_id - i - 1) % total_planes_cnt
        orbit.satellites[idx].source_neighbor = (sat_source.plane_id - i) % total_planes_cnt


def vir_sat_reset():
    for orbit in orbits:
        for sat in orbit.satellites:
            sat.prev = None
            sat.next = None
            sat.next2 = None
            sat.time_get_model = None
            sat.received = 0
            sat.time_send_model = 0
            sat.time_received = 0
            sat.time_trained = 0


def vir_three_step_dca_time(orbit):
    sat_source = orbit.satellites[orbit.source_node_id]
    t_train = sat_train_time

    sat_source.received = 1
    sat_source.time_received = 0  # 开始时间设置为0
    sat_source.time_trained = sat_source.time_received + t_train

    com_times = math.ceil(total_planes_cnt / 2)  # 双向传播，所以传播次数为卫星数除以2
    sat_link = list(range(0, total_planes_cnt))  # 该轨道的卫星list

    sat_link[sat_source.plane_id] = sat_source

    # 模型传播、训练过程
    for i in range(com_times):
        idx = (sat_source.plane_id + i + 1) % total_planes_cnt
        send_idx = (sat_source.plane_id + i) % total_planes_cnt
        orbit.satellites[idx].time_received = sat_to_sat_delay(orbit, orbit.satellites[send_idx].plane_id, orbit.satellites[idx].plane_id) \
                                            + orbit.satellites[send_idx].time_received
        orbit.satellites[idx].received = 1
        orbit.satellites[idx].time_trained = orbit.satellites[idx].time_received + t_train
        sat_link[orbit.satellites[idx].plane_id] = orbit.satellites[idx]


        if((sat_source.plane_id + i + 1) % total_planes_cnt != (sat_source.plane_id - i - 1) % total_planes_cnt):
            idx =  (sat_source.plane_id - i - 1) % total_planes_cnt
            send_idx = (sat_source.plane_id - i) % total_planes_cnt
            orbit.satellites[idx].time_received = sat_to_sat_delay(orbit, orbit.satellites[send_idx].plane_id, orbit.satellites[idx].plane_id,) \
                                                + orbit.satellites[send_idx].time_received
            orbit.satellites[idx].received = 1
            orbit.satellites[idx].time_trained = orbit.satellites[idx].time_received + t_train
            sat_link[orbit.satellites[idx].plane_id] = orbit.satellites[idx]
    
    time_sink_received = 0
    max_time = 0
    for sat in orbit.satellites:
        time_sink_received = sat.time_trained + sat_to_sat_delay(orbit, sat.plane_id, orbit.sink_node_id)
        max_time = max(max_time, time_sink_received)
    vir_sat_reset()

    return max_time