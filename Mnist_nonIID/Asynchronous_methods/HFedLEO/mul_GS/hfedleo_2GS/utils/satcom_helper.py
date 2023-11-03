import os
import math
import time
import sys
import inspect


total_orbits_cnt = 20
total_planes_cnt = 10
total_satellites_cnt = total_orbits_cnt * total_planes_cnt
per_gs_visible_times = 80
number_of_gs = 2
max_visible_times = per_gs_visible_times * number_of_gs # 预测未来21天


# 实验设置中的常量
Tansmission_power = 10 # 单位W
Gain_Antenna = 5 # 天线增益，即放大倍数
Carrier_frequency = 2.4 * math.pow(10, 9)  # 单位Hz
Noise_temperature = 354.81  # 单位K
Boltzmann_constant = 1.38064852 * math.pow(10, -23)  # 单位J/K
Bandwidth_up = 2.1 * math.pow(2, 30) # 单位Hz 2.1GHz
Bandwidth_down = 1.3 * math.pow(2, 30) # 单位Hz 1.3GHz
Bandwidth_sat2sat = 33 * math.pow(2, 30) # 单位Hz 33GHz
Speed_of_light = 299792.458  # 单位km/s

fsize = os.path.getsize('./save/w_glob.pkl') # 342K
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
        self.total_visible_times = [0 for _ in range(number_of_gs)]
        self.records = [Record() for _ in range(max_visible_times)]
        self.visible_idx = 0 # 当前过顶下标索引
        self.gs_visible_idxs = [i * per_gs_visible_times for i in range(number_of_gs)]
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

        self.spin_around_me_time = 0.0 # 以自己为中心，在轨道内广播一圈的时间


class Orbit:
    def __init__(self):
        self.orbit_id = -1
        self.satellites = [Satellite() for i in range(total_planes_cnt)]
        self.com_gs_id_up = -1
        self.com_gs_id_down = -1
        self.source_node_id = -1
        self.sink_node_id = -1
        self.source_node_received = 0

        # 记录轨道当前模型是哪一轮的
        self.model_round_id = -1
        self.training_over_time = 0


class OrMeta_Down:
    def __init__(self):
        self.orbit_id = -1
        self.acc_global = 0.0
        # 还有一个局部聚合模型


class Ground_Station:
    def __init__(self):
        self.orbit_model_sended = set([]) # GS已经发送模型的轨道
        self.orbit_model_received = set([]) # 已经收到局部模型的轨道

        # 为了导出轨道覆盖率，新增两个list变量
        self.order_rcvd_orbit_id = []
        self.order_rcvd_orbit_time = []


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
    filePath_images = '../data/mnist/MNIST/raw/train-images-idx3-ubyte'
    filePath_labels = '../data/mnist/MNIST/raw/train-labels-idx1-ubyte'
    fsize = os.path.getsize(filePath_images) + os.path.getsize(filePath_labels)
    train_time = (fsize * 8 / total_satellites_cnt) * ck / fk
    # print(train_time * local_epochs) 
    return train_time * sat_local_epochs


def orbit_avg_isl_rate(orbit):
    # 因为Tc,p计算时需要考虑到ISL的传输速率，但是这里只是为了估计，因此决定取orbit的平均值
    rate_sum = 0.0
    for sat in orbit.satellites:
        rate_sum += cal_rate_adj_sat2sat(sat, orbit.satellites[sat.forward_neighbor])
    return rate_sum / total_planes_cnt


def cal_time_e_p(orbit):
    time_c_p_max = 0.0
    time_l_p_max = sat_train_time
    for sat in orbit.satellites:
        if (sat.backward_range / Speed_of_light) > time_c_p_max:
            time_c_p_max = sat.backward_range / Speed_of_light
        if (sat.forward_range / Speed_of_light) > time_c_p_max:
            time_c_p_max = sat.forward_range / Speed_of_light
    time_c_p_max_p = time_c_p_max + Size_of_model / orbit_avg_isl_rate(orbit) # 分发阶段只传输1个model
    # time_c_p_max_a = time_c_p_max + (Size_of_model * math.ceil(total_planes_cnt/2)) / orbit_avg_isl_rate(orbit) # 聚合阶段最多传输4个model
    time_c_p_max_a = time_c_p_max + (Size_of_model + get_size(OrMeta_Down())) / orbit_avg_isl_rate(orbit)
    return math.ceil(total_planes_cnt/2) * time_c_p_max_p + time_l_p_max    # + math.ceil(total_planes_cnt/2) * time_c_p_max_a


sat_train_time = cal_sat_train()
satellites = [Satellite() for i in range(total_satellites_cnt+1)]
orbits = [Orbit() for i in range(total_orbits_cnt)]
Kernel_GS = Ground_Station() # 虚拟，暂时不具体确定是哪一个GS，仅假设存在这样一个GS担任核心聚合分发任务，不考虑GS之间通信延迟问题


# 获取类的大小
def get_size(obj, seen=None):
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if hasattr(obj, '__dict__'):
        for cls in obj.__class__.__mro__:
            if '__dict__' in cls.__dict__:
                d = cls.__dict__['__dict__']
                if inspect.isgetsetdescriptor(d) or inspect.ismemberdescriptor(d):
                    size += get_size(obj.__dict__, seen)
                break
    if isinstance(obj, dict):
        # 这里避免重复计算
        size += sum((get_size(v, seen) for v in obj.values() if not isinstance(v, (str, int, float, bytes, bytearray))))
        # size += sum((get_size(k, seen) for k in obj.keys())) 
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        # 这里避免重复计算
        size += sum((get_size(i, seen) for i in obj if not isinstance(i, (str, int, float, bytes, bytearray))))

    if hasattr(obj, '__slots__'): 
        size += sum(get_size(getattr(obj, s), seen) for s in obj.__slots__ if hasattr(obj, s))

    return size


# 获取列表的第二个元素
def takeSecond(elem):
    return elem[1]


# 该函数的src 和 dest允许差多跳
def sat_to_sat_delay(orbit, src, dest):
    dest_received = orbit.satellites[dest].received
    total_delay = 0.0
    if ((src < dest) & ((dest-src) >= (total_planes_cnt-dest+src))) | ((src > dest) & ((src-dest) <= (total_planes_cnt-src+dest))):  # -1方向传播，即为list1，list1共有6个节点
        while src != dest:
            total_delay += (orbit.satellites[src].forward_range / Speed_of_light) 
            if dest_received == 1:
                total_delay += ((Size_of_model + get_size(OrMeta_Down())) / cal_rate_adj_sat2sat(orbit.satellites[src], orbit.satellites[orbit.satellites[src].forward_neighbor]))
            else:
                total_delay += (Size_of_model / cal_rate_adj_sat2sat(orbit.satellites[src], orbit.satellites[orbit.satellites[src].forward_neighbor]))
            src = orbit.satellites[src].forward_neighbor
    elif ((src < dest) & ((dest-src) < (total_planes_cnt-dest+src))) | ((src > dest) & ((src-dest) > (total_planes_cnt-src+dest))):  # +1方向传播，即为list2，list2共有5个节点
        while src != dest:
            total_delay += (orbit.satellites[src].backward_range / Speed_of_light)
            if dest_received == 1:
                total_delay += ((Size_of_model + get_size(OrMeta_Down())) / cal_rate_adj_sat2sat(orbit.satellites[src], orbit.satellites[orbit.satellites[src].backward_neighbor]))
            else:
                total_delay += (Size_of_model / cal_rate_adj_sat2sat(orbit.satellites[src], orbit.satellites[orbit.satellites[src].backward_neighbor]))
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
    sat_sink = orbit.satellites[orbit.sink_node_id]
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
        orbit.satellites[idx].received = 1       
        orbit.satellites[idx].time_received = sat_to_sat_delay(orbit, orbit.satellites[send_idx].plane_id, orbit.satellites[idx].plane_id) \
                                            + orbit.satellites[send_idx].time_received
        orbit.satellites[idx].time_trained = orbit.satellites[idx].time_received + t_train
        sat_link[orbit.satellites[idx].plane_id] = orbit.satellites[idx]

        idx =  (sat_source.plane_id - i - 1) % total_planes_cnt
        send_idx = (sat_source.plane_id - i) % total_planes_cnt
        orbit.satellites[idx].received = 1
        orbit.satellites[idx].time_received = sat_to_sat_delay(orbit, orbit.satellites[send_idx].plane_id, orbit.satellites[idx].plane_id,) \
                                            + orbit.satellites[send_idx].time_received
        orbit.satellites[idx].time_trained = orbit.satellites[idx].time_received + t_train
        sat_link[orbit.satellites[idx].plane_id] = orbit.satellites[idx]

    # 构建链表
    sink_idx = sat_sink.plane_id
    list1 = sat_list()
    list2 = sat_list()
    for i in range(com_times):
        list1.append(sat_link[(sink_idx + i + 1) % total_planes_cnt])
        if (sink_idx + i + 1) % total_planes_cnt is not (sink_idx - i - 1) % total_planes_cnt:
            list2.append(sat_link[(sink_idx - i - 1) % total_planes_cnt])

    list1.get_head().prev = sat_link[sink_idx]
    sat_link[sink_idx].next = list1.get_head()
    list2.get_head().prev = sat_link[sink_idx]
    sat_link[sink_idx].next2 = list2.get_head()
    cur = list1.get_tail()
    cur.time_get_model = cur.time_trained

    # 模型训练完成后聚合过程
    while cur.prev is not None:
        cur.time_send_model = max(cur.time_get_model, cur.time_trained)
        pre = cur.prev
        pre.time_get_model = cur.time_send_model + sat_to_sat_delay(orbit, cur.plane_id, pre.plane_id)
        cur = cur.prev
    cur = list2.get_tail()
    cur.time_get_model = cur.time_trained
    while cur.prev is not None:
        cur.time_send_model = max(cur.time_get_model, cur.time_trained)
        pre = cur.prev
        pre.time_get_model = cur.time_send_model + sat_to_sat_delay(orbit, cur.plane_id, pre.plane_id)
        cur = cur.prev
    time1 = sat_sink.next.time_send_model + sat_to_sat_delay(orbit, sat_sink.next.plane_id, sat_sink.plane_id)
    time2 = sat_sink.next2.time_send_model + sat_to_sat_delay(orbit, sat_sink.next2.plane_id, sat_sink.plane_id)
    time_total = max(time1, time2)
    sat_link[sink_idx].time_get_model = time_total
    vir_sat_reset()
    return time_total


# 计算轨道上每一颗卫星以自己为中心时，转一圈的时间
def cal_taking_a_spin_around(orbits):
    for orbit in orbits:
        for sat in orbit.satellites:
            forward_time = sat_to_sat_delay(orbit, sat.plane_id, (sat.plane_id + math.ceil(total_planes_cnt/2)) % total_planes_cnt)
            backward_time = sat_to_sat_delay(orbit, sat.plane_id, (sat.plane_id + math.ceil(total_planes_cnt/2) - 1) % total_planes_cnt)
            if forward_time >= backward_time:
                sat.spin_around_me_time =  forward_time
            else:
                sat.spin_around_me_time = backward_time