import time
import datetime
import ast
import numpy as np
import math
import os


Tansmission_power = 10 # 单位W
Gain_Antenna = 5 # 放大倍数
Carrier_frequency = 2.4 * math.pow(10, 9)  # 单位Hz
Noise_temperature = 354.81  # 单位K
Boltzmann_constant = 1.38064852 * math.pow(10, -23)  # 单位J/K
Bandwidth_up = 2.1 * math.pow(2, 30) # 单位Hz 2.1GHz
Bandwidth_down = 1.3 * math.pow(2, 30) # 单位Hz 1.3GHz
Speed_of_light = 299792.458  # 单位km/s
local_epochs = 20

# 训练开始时间
stk_train_start1 = "2023-06-01 00:00:00"
stk_train_start11 = time.strptime(stk_train_start1, "%Y-%m-%d %H:%M:%S")
stk_train_start_stamp = time.mktime(stk_train_start11)

# 训练结束时间
stk_train_end1 = "2023-06-06 12:00:00"
stk_train_end11 = time.strptime(stk_train_end1, "%Y-%m-%d %H:%M:%S")
stk_train_end_stamp = time.mktime(stk_train_end11)

frontest_time = stk_train_start1


class Record:
    def __init__(self):
        self.start_time = time.time()
        self.stop_time = time.time()
        self.start_tstamp = 0.0 # float
        self.stop_tstamp = 0.0
        self.delta_tstamp = 0.0 # 上述两者的差值,单位s
        self.range = 0.0 # 过顶时，卫星与地面站的距离，单位km
        self.com_time_down = 0.0 # 下行链路通信时延,传递模型给GS
        self.com_time_up = 0.0  # 上行链路通信时延，接收模型

total_orbits_cnt = 20
total_planes_cnt = 10
total_satellites_cnt = total_orbits_cnt * total_planes_cnt
max_visible_times = 70 # 预测未来21天，最多不超过120次

class Satellite:
    def __init__(self):
        self.orbit_id = 0
        self.plane_id = 0
        self.total_visible_times = 0
        self.records = [Record() for _ in range(max_visible_times)]
        self.visible_idx = 0 # 当前过顶下标索引
        self.model_received = 0
        self.model_trained = 0
        self.model_sended = 0
        self.visited = 0 # 当前索引的可见期是否被访问过

        self.up = 0
        self.local_round = 0


def cal_sat_train():
    ck = 1000
    fk = math.pow(10, 9)
    filePath_images = '../data/mnist/MNIST/raw/train-images-idx3-ubyte'
    filePath_labels = '../data/mnist/MNIST/raw/train-labels-idx1-ubyte'
    fsize = os.path.getsize(filePath_images) + os.path.getsize(filePath_labels)
    train_time = (fsize * 8 / total_satellites_cnt) * ck / fk
    # print(train_time * local_epochs) 
    return train_time * local_epochs


sat_train_time = cal_sat_train()
satellites = [Satellite() for i in range(total_satellites_cnt + 1)]


def read_time_file():
    sat_idx = 0
    tmpsat = Satellite()
    tmpsat.orbit_id = tmpsat.plane_id = -1

    fileHandler = open("./input_info/encounter_time.txt", "r")
    while True:
        # Get next line from file
        line = fileHandler.readline()
        # If line is empty then end of file reached
        if not line:
            satellites[sat_idx] = tmpsat
            break
        line = line.strip()
        slices = line.split(':', 3)

        tmprcd = Record()
        for i, slice in enumerate(slices):
            if i == 0:
                # 例如Sat0_0
                orbit_id = int(slice.split('_')[0][3:])
                plane_id = int(slice.split('_')[1])
                if (tmpsat.orbit_id != orbit_id) | (tmpsat.plane_id != plane_id):
                    # 说明当前该卫星的tmpsat尚未存在
                    satellites[sat_idx] = tmpsat
                    sat_idx = orbit_id * total_planes_cnt + plane_id + 1
                    tmpsat = Satellite()
                    tmpsat.orbit_id = orbit_id
                    tmpsat.plane_id = plane_id
            elif i == 1:
                tmpsat.total_visible_times = int(slice)
            elif i == 2:
                tmpsat.visible_idx = int(slice)
            elif i == 3:
                tms = slice.split(',')
                tms[0] = tms[0][2:-5]
                tms[1] = tms[1][2:-6]

                dt0 = datetime.datetime.strptime(tms[0], '%d %b %Y %H:%M:%S')
                dt1 = datetime.datetime.strptime(tms[1], '%d %b %Y %H:%M:%S')
                tmprcd.start_time = dt0
                tmprcd.stop_time = dt1
                tmprcd.start_tstamp = dt0.timestamp()
                tmprcd.stop_tstamp = dt1.timestamp()
                tmprcd.delta_tstamp = dt1.timestamp() - dt0.timestamp()
        tmpsat.records[tmpsat.visible_idx] = tmprcd

    fileHandler.close()


fsize = os.path.getsize('./save/w_glob.pkl')
Size_of_model = fsize * 8  # 模型大小，单位bit


def PathLoss_k_GS(dis):
    res = 4 * math.pi * dis * Carrier_frequency / Speed_of_light
    res = res * res
    return res


def read_range_file():
    pre_orbit = -1
    pre_plane = -1
    sat_idx = 0

    fileHandler = open("./input_info/encounter_range.txt", "r")
    while True:
        # Get next line from file
        line = fileHandler.readline()
        # If line is empty then end of file reached
        if not line:
            break
        line = line.strip()
        slices = line.split(':', 3)

        orbit_id = 0  # 轨道编号
        plane_id = 0  # 所在轨道下 卫星编号
        visible_idx = 0
        for i, slice in enumerate(slices):
            if i == 0:
                # 例如Sat0_0
                orbit_id = int(slice.split('_')[0][3:])
                plane_id = int(slice.split('_')[1])
                if (pre_orbit != orbit_id) | (pre_plane != plane_id):
                    sat_idx = sat_idx = orbit_id * total_planes_cnt + plane_id + 1
                    pre_orbit = orbit_id
                    pre_plane = plane_id
            # elif i == 1:
            # assert satellites[sat_idx].total_visible_times == int(slice)
            elif i == 2:
                visible_idx = int(slice)
            elif i == 3:
                satellites[sat_idx].records[visible_idx].range = ast.literal_eval(slice[:-2])
                tp = satellites[sat_idx].records[visible_idx].range / Speed_of_light
                # 直接以无量纲形式计算SNR
                SNR_up = (Tansmission_power * Gain_Antenna * Gain_Antenna) / (
                        Boltzmann_constant * Noise_temperature * Bandwidth_up * PathLoss_k_GS(
                    satellites[sat_idx].records[visible_idx].range))
                SNR_down = (Tansmission_power * Gain_Antenna * Gain_Antenna) / (
                        Boltzmann_constant * Noise_temperature * Bandwidth_down * PathLoss_k_GS(
                    satellites[sat_idx].records[visible_idx].range))

                # 信噪比
                tt_uplink = Size_of_model / (Bandwidth_up * math.log(1 + SNR_up, 2))
                tt_downlink = Size_of_model / (Bandwidth_down * math.log(1 + SNR_down, 2))
                satellites[sat_idx].records[visible_idx].com_time_up = tp + tt_uplink
                satellites[sat_idx].records[
                    visible_idx].com_time_down = tp + tt_downlink  # tt_downlink 目前尚不考虑多颗卫星并行抢占带宽的可能

    fileHandler.close()
    return satellites


def vir_train_init(access_time):
    for sat in satellites:
        sat.visible_idx = 0
        sat.up = 0  # 一开始所有卫星都是在等待接收全局模型
        access_time.append(sat.records[sat.visible_idx].start_tstamp)


# 获取列表的第二个元素
def takeSecond(elem):
    return elem[1]


def vir_train_process(filename):
    global frontest_time
    # 统一所有卫星的索引下标，置为0
    access_order = [i for i in range(len(satellites))]
    access_time = []
    vir_train_init(access_time)
    round = 0
    
    merge_list = list(zip(access_order, access_time))
    # 指定第二个元素排序
    merge_list.sort(key=takeSecond)
    # print(len(merge_list))
    # print(merge_list[:5])

    with open(filename,'w') as f:
        while merge_list[0][1] < stk_train_end_stamp:
            cur_sat_id, frontest_time = merge_list[0]
            merge_list.remove((cur_sat_id, frontest_time))
            # print("现在处理的卫星是：", cur_sat_id)
            if satellites[cur_sat_id].up == 0:
                if satellites[cur_sat_id].records[satellites[cur_sat_id].visible_idx].stop_tstamp - frontest_time > \
                    satellites[cur_sat_id].records[satellites[cur_sat_id].visible_idx].com_time_up: # 有机会传输全局模型给他
                        # 就让它去训练
                        satellites[cur_sat_id].local_round = round
                        # 然后把结束的时间加进来
                        tmp_time = frontest_time + satellites[cur_sat_id].records[satellites[cur_sat_id].visible_idx].com_time_up + \
                                sat_train_time
                        merge_list.append((cur_sat_id, tmp_time))
                        satellites[cur_sat_id].up = 1
                else:   # 没有机会传输全局模型给它了，那么就等到下一个可见期吧
                    satellites[cur_sat_id].visible_idx += 1
                    merge_list.append((cur_sat_id, satellites[cur_sat_id].records[satellites[cur_sat_id].visible_idx].start_tstamp))
            else:   #该卫星的可见是需要下发模型
                # 更新全局模型，写文件：卫星编号 + 本地的轮次
                f.write(str(cur_sat_id) + " " + str(satellites[cur_sat_id].local_round) + "\n")
                round += 1

                satellites[cur_sat_id].up = 0
                # 更新状态
                tmp_time = frontest_time + satellites[cur_sat_id].records[satellites[cur_sat_id].visible_idx].com_time_down
                merge_list.append((cur_sat_id, tmp_time))
            merge_list.sort(key=takeSecond)
            # print(merge_list[:5])

            if round % 50 == 0:
                # 转换成localtime
                time_local = time.localtime(frontest_time)
                # 转换成新的时间格式(2016-05-05 20:28:54)
                dt = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
                print("Epoch No." + str(round), "epoch is over, now time is", dt)
        
    print(round)


if __name__ == '__main__':
    # 获取必要的数据存储到结构体中
    read_time_file()
    read_range_file()

    # 开始训练过程
    satellites = satellites[1:]

    total_time_period = stk_train_end_stamp - stk_train_start_stamp

    if os.path.exists('./output_info/satellites_order.txt'): 
        os.remove('./output_info/satellites_order.txt')
    vir_train_process("./output_info/satellites_order.txt")

    stamp2 = frontest_time
    t1 = time.localtime(stk_train_start_stamp)
    t2 = time.localtime(stamp2)
    t1 = time.strftime("%Y-%m-%d %H:%M:%S",t1)
    t2 = time.strftime("%Y-%m-%d %H:%M:%S", t2)
    time1 = datetime.datetime.strptime(t1,"%Y-%m-%d %H:%M:%S")
    time2 = datetime.datetime.strptime(t2, "%Y-%m-%d %H:%M:%S")
    print("constellation StarLink training start time:", time1)
    print("constellation StarLink training end time:", time2)