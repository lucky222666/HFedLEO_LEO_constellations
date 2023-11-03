import time
import datetime
import ast
import numpy as np
import math
import os

Tansmission_power = 10  # 单位W
Gain_Antenna = 5  # 放大倍数
Carrier_frequency = 2.4 * math.pow(10, 9)  # 单位Hz
Noise_temperature = 354.81  # 单位K
Boltzmann_constant = 1.38064852 * math.pow(10, -23)  # 单位J/K
Bandwidth_up = 2.1 * math.pow(2, 30)  # 单位Hz 2.1GHz
Bandwidth_down = 1.3 * math.pow(2, 30)  # 单位Hz 1.3GHz
Speed_of_light = 299792.458  # 单位km/s
local_epochs = 20

# 训练开始时间
stk_train_start1 = "2023-06-01 00:00:00"
stk_train_start11 = time.strptime(stk_train_start1, "%Y-%m-%d %H:%M:%S")
stk_train_start_stamp = time.mktime(stk_train_start11)
epoch_start_time = stk_train_start_stamp

# 训练结束时间
stk_train_end1 = "2023-06-22 00:00:00"
stk_train_end11 = time.strptime(stk_train_end1, "%Y-%m-%d %H:%M:%S")
stk_train_end_stamp = time.mktime(stk_train_end11)


class Record:
    def __init__(self):
        self.start_time = time.time()
        self.stop_time = time.time()
        self.start_tstamp = 0.0  # float
        self.stop_tstamp = 0.0
        self.delta_tstamp = 0.0  # 上述两者的差值,单位s
        self.range = 0.0  # 过顶时，卫星与地面站的距离，单位km
        self.com_time_down = 0.0  # 下行链路通信时延,传递模型给GS
        self.com_time_up = 0.0  # 上行链路通信时延，接收模型


total_orbits_cnt = 72
total_planes_cnt = 22
total_satellites_cnt = total_orbits_cnt * total_planes_cnt
max_visible_times = 120  # 预测未来21天，最多不超过120次

late_time = 84 * 60
whether_late = False


class Satellite:
    def __init__(self):
        self.orbit_id = 0
        self.plane_id = 0
        self.total_visible_times = 0
        self.records = [Record() for _ in range(max_visible_times)]
        self.visible_idx = 0  # 当前过顶下标索引
        self.model_received = 0
        self.model_trained = 0
        self.model_sended = 0
        self.visited = 0  # 当前索引的可见期是否被访问过


class Ground_Station:
    def __init__(self):
        self.model_up = set([])  # 列表
        self.all_received = 0


def cal_sat_train():
    ck = 1000
    fk = math.pow(10, 9)
    filePath_images = '../data/emnist/EMNIST/raw/emnist-bymerge-train-images-idx3-ubyte'
    filePath_labels = '../data/emnist/EMNIST/raw/emnist-bymerge-train-labels-idx1-ubyte'
    fsize = os.path.getsize(filePath_images) + os.path.getsize(filePath_labels)
    train_time = (fsize * 8 / total_satellites_cnt) * ck / fk
    # print(train_time * local_epochs) 
    return train_time * local_epochs
    # return 60


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


def vir_train_init():
    for sat in satellites:
        sat.visible_idx = 0


MoRolla = Ground_Station()
frontest_time = time.time()


def model_trained_unsended(choosed_sat, cur_idx):
    time_gap = satellites[choosed_sat].records[cur_idx].start_tstamp + satellites[choosed_sat].records[
        cur_idx].com_time_down - epoch_start_time
    if time_gap > late_time:
        global whether_late
        whether_late = True
        return
    satellites[choosed_sat].model_sended = 1  # 本地局部模型已上传，否则仍标记为0

    MoRolla.model_up.add(choosed_sat)
    # print("Sat"+str(satellites[choosed_sat].orbit_id)+"_"+str(satellites[choosed_sat].plane_id), "sended local model to GS")

    satellites[choosed_sat].records[cur_idx].start_tstamp += satellites[choosed_sat].records[cur_idx].com_time_down
    satellites[choosed_sat].records[cur_idx].delta_tstamp -= satellites[choosed_sat].records[cur_idx].com_time_down

    global frontest_time
    frontest_time = satellites[choosed_sat].records[cur_idx].start_tstamp

    satellites[choosed_sat].model_received = 0
    satellites[choosed_sat].model_trained = 0
    satellites[choosed_sat].model_sended = 0
    satellites[choosed_sat].visited = 0


def model_will_received(choosed_sat, cur_idx):
    if satellites[choosed_sat].records[cur_idx].com_time_up < satellites[choosed_sat].records[cur_idx].delta_tstamp:
        satellites[choosed_sat].model_received = 1  # 全局模型接收到,否则仍标记为0
        if satellites[choosed_sat].records[cur_idx].com_time_up + sat_train_time < satellites[choosed_sat].records[
            cur_idx].delta_tstamp:
            satellites[choosed_sat].model_trained = 1  # 本地局部模型已训练出来，否则仍标记为0

    if (satellites[choosed_sat].model_received == 1) & (satellites[choosed_sat].model_trained == 1):
        satellites[choosed_sat].model_sended = 1  # 本地局部模型已上传，否则仍标记为0

        # print("Sat"+str(satellites[choosed_sat].orbit_id)+"_"+str(satellites[choosed_sat].plane_id), "sended local model to GS")

        total_com_time = satellites[choosed_sat].records[cur_idx].com_time_up + satellites[choosed_sat].records[
            cur_idx].com_time_down
        time_gap = satellites[choosed_sat].records[
                       cur_idx].start_tstamp + total_com_time + sat_train_time - epoch_start_time
        if time_gap > late_time:
            global whether_late
            whether_late = True
            return
        MoRolla.model_up.add(choosed_sat)
        satellites[choosed_sat].records[cur_idx].start_tstamp += total_com_time + sat_train_time
        satellites[choosed_sat].records[cur_idx].delta_tstamp -= total_com_time + sat_train_time

        global frontest_time
        frontest_time = satellites[choosed_sat].records[cur_idx].start_tstamp

        satellites[choosed_sat].model_received = 0
        satellites[choosed_sat].model_trained = 0
        satellites[choosed_sat].model_sended = 0
        satellites[choosed_sat].visited = 0
    elif (satellites[choosed_sat].model_received == 1) & (satellites[choosed_sat].model_trained == 0):
        satellites[choosed_sat].model_sended = 0
        # 这个过顶期间应该是不能发送模型了，那么就等到下一个吧
        satellites[choosed_sat].visible_idx += 1


def vir_train_judge(choosed_sat):
    if choosed_sat not in MoRolla.model_up:
        cur_idx = satellites[choosed_sat].visible_idx
        # 本地训练模型未上传的卫星
        if (satellites[choosed_sat].model_received == 1) & (satellites[choosed_sat].model_trained == 1) & (
                satellites[choosed_sat].model_sended == 0):
            model_trained_unsended(choosed_sat, cur_idx)
        # 准备接收新一轮全局模型的卫星
        elif (satellites[choosed_sat].model_received == 0) & (satellites[choosed_sat].model_trained == 0) & (
                satellites[choosed_sat].model_sended == 0):
            model_will_received(choosed_sat, cur_idx)
        # 还没有训练好模型的卫星
        elif (satellites[choosed_sat].model_received == 1) & (satellites[choosed_sat].model_trained == 0):
            return  # 继续本地先默默完成训练


def vir_train_update():
    for sat in satellites:
        if (sat.model_received == 1) & (sat.model_trained == 0):  # 为此类卫星选择及时更新模型训练结果
            if (sat.visible_idx > 0) & (sat.records[sat.visible_idx - 1].start_tstamp + sat.records[
                sat.visible_idx - 1].com_time_up + sat_train_time < frontest_time):
                sat.model_trained = 1
                sat.model_sended = 0

        if sat.records[sat.visible_idx].stop_tstamp < frontest_time:
            sat.visible_idx += 1
            sat.visited = 0


def choose_satellite():
    global frontest_time
    choosed_sat = -1
    for i, sat in enumerate(satellites):
        if (sat.visited == 0) & (sat.records[sat.visible_idx].start_tstamp <= frontest_time):
            frontest_time = sat.records[sat.visible_idx].start_tstamp
            choosed_sat = i
    return choosed_sat


def vir_train_process(epochs_needed, fraction, filename):
    global frontest_time
    frontest_time = satellites[0].records[0].start_tstamp

    # 统一所有卫星的索引下标，置为0
    vir_train_init()
    round = 0

    with open(filename, 'w') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
        while round < epochs_needed:
            MoRolla.model_up.clear()
            MoRolla.all_received = 0
            # 更新时间轴在frontest_time的
            for sat in satellites:
                sat.model_received = 0
                sat.model_trained = 0
                sat.model_sended = 0
                if (sat.records[sat.visible_idx].start_tstamp <= frontest_time) & (
                        sat.records[sat.visible_idx].stop_tstamp > frontest_time):
                    sat.visited = 0
                    sat.records[sat.visible_idx].start_time = frontest_time
            while True:
                # 确定选择哪一颗卫星
                choosed_sat = choose_satellite()
                if choosed_sat == -1:
                    # 没有选择到合适的，frontest_time给的不合适
                    frontest_time = stk_train_end_stamp
                    continue

                satellites[choosed_sat].visited = 1
                # 更新frontest_time
                vir_train_update()
                vir_train_judge(choosed_sat)
                global whether_late
                if len(MoRolla.model_up) == math.ceil(total_satellites_cnt * fraction) or whether_late:
                    whether_late = False
                    global epoch_start_time
                    if whether_late:
                        epoch_start_time+=late_time
                    else:
                        epoch_start_time = frontest_time
                    MoRolla.all_received = 1
                    round += 1

                    f.write(str(MoRolla.model_up) + "\n")

                    # 转换成localtime
                    time_local = time.localtime(epoch_start_time)
                    # 转换成新的时间格式(2016-05-05 20:28:54)
                    dt = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
                    print("Epoch No." + str(round), "epoch is over, now time is", dt)
                    # print("Num of Satellites: "+str(len(MoRolla.model_up)))
                    break


if __name__ == '__main__':
    # 获取必要的数据存储到结构体中
    read_time_file()
    read_range_file()

    # 开始训练过程
    satellites = satellites[1:]

    vir_train_process(200, 0.8, "./output_info/participant_sats.txt")

    stamp2 = frontest_time
    t1 = time.localtime(stk_train_start_stamp)
    t2 = time.localtime(stamp2)
    t1 = time.strftime("%Y-%m-%d %H:%M:%S", t1)
    t2 = time.strftime("%Y-%m-%d %H:%M:%S", t2)
    time1 = datetime.datetime.strptime(t1, "%Y-%m-%d %H:%M:%S")
    time2 = datetime.datetime.strptime(t2, "%Y-%m-%d %H:%M:%S")
    print("constellation StarLink training start time:", time1)
    print("constellation StarLink training end time:", time2)
