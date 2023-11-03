import time
import datetime
import ast
import numpy as np
import math
import os
from helper import *


def read_time_file():    
    sat_idx = 0
    tmpsat = Satellite()
    tmpsat.orbit_id = tmpsat.plane_id = -1

    fileHandler = open("./input_info/sat_gs_encounter_time.txt", "r")
    while True:
    # Get next line from file
        line = fileHandler.readline()
        # If line is empty then end of file reached
        if not line:
            satellites[sat_idx] = tmpsat
            break;
        line = line.strip()
        slices = line.split(':', 3)
        
        
        tmprcd = Record()
        for i, slice in enumerate(slices):
            if i == 0:
                # 例如Sat10_10
                orbit_id = int(slice.split('_')[0][3:])
                plane_id = int(slice.split('_')[1])
                if (tmpsat.orbit_id != orbit_id) | (tmpsat.plane_id != plane_id):
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


def read_range1_file():  
    pre_orbit = -1
    pre_plane = -1
    sat_idx = 0

    fileHandler = open("./input_info/sat_gs_encounter_range.txt", "r")
    while True:
    # Get next line from file
        line = fileHandler.readline()
        # If line is empty then end of file reached
        if not line:
            break;
        line = line.strip()
        slices = line.split(':', 3)

        orbit_id = 0 # 轨道编号
        plane_id = 0 # 所在轨道下 卫星编号
        visible_idx = 0
        for i, slice in enumerate(slices):
            if i == 0:
                # 例如Sat10_10
                orbit_id = int(slice.split('_')[0][3:])
                plane_id = int(slice.split('_')[1])
                if (pre_orbit != orbit_id) | (pre_plane != plane_id):
                    sat_idx = orbit_id * total_planes_cnt + plane_id + 1
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
                SNR_up = (Tansmission_power*Gain_Antenna*Gain_Antenna) / (Boltzmann_constant*Noise_temperature*Bandwidth_up*pathloss_K_GS(satellites[sat_idx].records[visible_idx].range))
                SNR_down = (Tansmission_power*Gain_Antenna*Gain_Antenna) / (Boltzmann_constant*Noise_temperature*Bandwidth_down*pathloss_K_GS(satellites[sat_idx].records[visible_idx].range))
    
                # 信噪比
                tt_uplink = Size_of_model / (Bandwidth_up *  math.log(1+SNR_up, 2))
                tt_downlink = Size_of_model / (Bandwidth_down * math.log(1+SNR_down, 2))
                satellites[sat_idx].records[visible_idx].com_time_up = tp + tt_uplink
                # if sat_idx == 1:
                #     print(Size_of_model, (Bandwidth_up *  math.log(1+SNR_up, 2)), tt_uplink)
                satellites[sat_idx].records[visible_idx].com_time_down = tp + tt_downlink # 目前尚不考虑多颗卫星并行抢占带宽的可能

    fileHandler.close()
    return satellites


def read_range2_file():
    sat_idx = 1
    line_idx = 0

    fileHandler = open("./input_info/sat_sat_encounter_range.txt", "r")
    while True:
    # Get next line from file
        line = fileHandler.readline()
        # If line is empty then end of file reached
        if not line:
            break;
        line_idx += 1
        line = line.strip()
        slices = line.split(' ', 3)

        # 类似Sat0_11/Reciver_Sat0_11 to Sat0_10/Transmitter_Sat0_10 1969.9712572628584
        c_slices = slices[2].split('/', 1)
        cc_slices = c_slices[0].split('_', 1)
        c_orbit = int(cc_slices[0][3:])
        c_plane = int(cc_slices[1])
        sat_idx = c_orbit * total_planes_cnt + c_plane + 1
        for i, slice in enumerate(slices):
            if i == 0:
                t_slices = slice.split('/', 1)
                tt_slices = t_slices[0].split('_', 1)
                plane_id = int(tt_slices[1])
                if line_idx % 2 == 1: # 奇数行，是自己的backward      
                    satellites[sat_idx].backward_neighbor = plane_id
                else:
                    satellites[sat_idx].forward_neighbor = plane_id
            elif i == 3:
                range = ast.literal_eval(slice)
                if line_idx % 2 == 1: # 奇数行，是自己的backward
                    satellites[sat_idx].backward_range = range
                else:
                    satellites[sat_idx].forward_range = range


def sat_belongto_orbit():
    for sat in satellites:
        orbits[sat.orbit_id].orbit_id = sat.orbit_id
        orbits[sat.orbit_id].satellites[sat.plane_id] = sat


def vir_train_init():
    for orbit in orbits:
        for sat in orbit.satellites:
            sat.visible_idx = 0


def choose_orbit_source(orbit, current_time):
    visited_cnt = 0
    timeline_advancement = 0.0
    for sat in orbit.satellites:
        if sat.visited == 1:
            visited_cnt += 1
    
    max_current_time = current_time
    if visited_cnt == total_planes_cnt: # 说明该轨道所有卫星都已经尝试过在当前可见期作为source 但是都失败了，于是大家都进去下一个可见期
        for sat in orbit.satellites:
            if sat.records[sat.visible_idx].stop_tstamp > max_current_time:
                max_current_time = sat.records[sat.visible_idx].stop_tstamp 
                sat.visible_idx += 1
                sat.visited = 0
        timeline_advancement = max_current_time - current_time
        current_time = max_current_time   


    # 确保轨道的所有卫星都在当前时间附近，没有滞后
    for sat in orbit.satellites:
        while sat.records[sat.visible_idx].stop_tstamp < current_time:
            sat.visible_idx += 1

    # 优先选择正在可见期的卫星(若有多个，则选择距离最短的一个)
    min_range = 35786   # 最远假设有35786km GEO高度
    for sat in orbit.satellites:
        if (current_time >= sat.records[sat.visible_idx].start_tstamp) & \
            (current_time < sat.records[sat.visible_idx].stop_tstamp) & (sat.visited == 0): # visted用来防止那些可见期太短的卫星反复被选中
            if sat.records[sat.visible_idx].range < min_range:  # 多个可见时，按照距离择优选source node
                if min_range != 35786:
                    orbit.satellites[orbit.source_node_id].visited = 0
            
                min_range = sat.records[sat.visible_idx].range
                orbit.source_node_id = sat.plane_id
                sat.visited = 1
    if min_range != 35786:
        return (0 + timeline_advancement)
        
    # 退而求其次选择最早即将到可见期的卫星
    # min_waiting_time = orbit.satellites[0].records[orbit.satellites[0].visible_idx].start_tstamp - current_time
    min_waiting_time = stk_train_end_stamp - current_time
    for sat in orbit.satellites:
        if ((sat.records[sat.visible_idx].start_tstamp - current_time) <= min_waiting_time) & (sat.visited == 0):
            min_waiting_time = sat.records[sat.visible_idx].start_tstamp - current_time
            orbit.source_node_id = sat.plane_id
            sat.visited = 1
    return (min_waiting_time + timeline_advancement)


def choose_orbit_sink(orbit):
    t_n = orbit.satellites[orbit.source_node_id].records[orbit.satellites[orbit.source_node_id].visible_idx].start_tstamp + orbit.satellites[orbit.source_node_id].records[orbit.satellites[orbit.source_node_id].visible_idx].com_time_up
    T_e_p = cal_time_e_p(orbit)
    current_time = t_n + T_e_p
    # print(T_e_p)
    sum_visible_time_max = 0.0
    for sat in orbit.satellites:
        sum_visible_time_tmp = 0.0
        # 在时间段之前的情况
        while sat.records[sat.visible_idx].stop_tstamp <= current_time:
            sat.visible_idx += 1
        
        pre_visible_idx = sat.visible_idx

        if sat.records[sat.visible_idx].start_tstamp < current_time:
            sum_visible_time_tmp += sat.records[sat.visible_idx].stop_tstamp - current_time
            sat.visible_idx += 1

        while sat.visible_idx < sat.total_visible_times:
            sum_visible_time_tmp += sat.records[sat.visible_idx].delta_tstamp
            sat.visible_idx += 1

        # 恢复原来的可见期索引，上面只是在预测而已
        sat.visible_idx = pre_visible_idx
        # 如果可见期最大 则选作sink node
        if sum_visible_time_tmp > sum_visible_time_max:
            sum_visible_time_max = sum_visible_time_tmp
            orbit.sink_node_id = sat.plane_id


def choose_orbit_sink_again(orbit, current_time):
    # 同步更新所有卫星的当前索引
    for sat in orbit.satellites:
        while sat.records[sat.visible_idx].stop_tstamp < current_time:
            sat.visible_idx += 1

    # 首先看有没有当前就在可见期的卫星
    for sat in orbit.satellites:
        if (current_time >= sat.records[sat.visible_idx].start_tstamp) & \
             (current_time <= sat.records[sat.visible_idx].stop_tstamp):
            orbit.sink_node_id = sat.plane_id
            return 0
    
    # 如果没有，那么就选择距离cur time最近的卫星
    min_waiting_time = orbit.satellites[orbit.sink_node_id].records[orbit.satellites[orbit.sink_node_id].visible_idx].start_tstamp - current_time
    for sat in orbit.satellites:
        if (sat.records[sat.visible_idx].start_tstamp - current_time) < min_waiting_time:
            min_waiting_time = sat.records[sat.visible_idx].start_tstamp - current_time
            orbit.sink_node_id = sat.plane_id
    return min_waiting_time


def handle_idle_sat(orbit, current_time):
    for sat in orbit.satellites:
        if (sat.records[sat.visible_idx].start_tstamp - orbit.satellites[orbit.source_node_id].records[orbit.satellites[orbit.source_node_id].visible_idx].start_tstamp) \
              < orbit.satellites[orbit.source_node_id].records[orbit.satellites[orbit.source_node_id].visible_idx].com_time_up + sat_to_sat_delay(orbit, orbit.source_node_id, sat.plane_id):
            # 全局模型已经发送 & 已经接收到，那么就等待（第一 二种情况）
            if (orbit.orbit_id in MoRolla.orbit_model_sended) & (orbit.source_node_received == 1):
                continue
            # 全局模型已经发送 & 没有收到，说明可见期太短了，这个source node不合格（第三种情况）
            elif (orbit.orbit_id in MoRolla.orbit_model_sended) & (orbit.source_node_received == 0):
                while orbit.satellites[orbit.source_node_id].records[orbit.satellites[orbit.source_node_id].visible_idx].delta_tstamp \
                    < orbit.satellites[orbit.source_node_id].records[orbit.satellites[orbit.source_node_id].visible_idx].com_time_up:
                    choose_orbit_source(orbit, current_time)


def sink_send_model_PS(orbit, current_time):
    # 同步更新所有卫星的当前索引
    for sat in orbit.satellites:
        while sat.records[sat.visible_idx].stop_tstamp < current_time:
            sat.visible_idx += 1

    orbit_end = current_time
    # 在sink node的当前可见期中完成任务
    if (current_time >= orbit.satellites[orbit.sink_node_id].records[orbit.satellites[orbit.sink_node_id].visible_idx].start_tstamp) & \
        (current_time <= orbit.satellites[orbit.sink_node_id].records[orbit.satellites[orbit.sink_node_id].visible_idx].stop_tstamp):
        orbit_end = current_time + orbit.satellites[orbit.sink_node_id].records[orbit.satellites[orbit.sink_node_id].visible_idx].com_time_down
    # 在sink node的当前可见期之前/之后，任务过早/过晚完成
    else: 
        pre_sink_node = orbit.sink_node_id
        min_waiting_time = choose_orbit_sink_again(orbit, current_time) # 有可能遇到可见期的直接发 0；也有可能需要等待
        orbit_end = current_time + min_waiting_time + sat_to_sat_delay(orbit, pre_sink_node, orbit.sink_node_id) + \
                    orbit.satellites[orbit.sink_node_id].records[orbit.satellites[orbit.sink_node_id].visible_idx].com_time_down
    return orbit_end


# 以轨道为单位进行逻辑处理
def vir_train_process(epochs_needed, vir_start_time):
    # 统一所有轨道上卫星的索引下标，置为0
    vir_train_init()
    per_epoch_start_time = vir_start_time
    round = 0

    while round < epochs_needed:
        for orbit in orbits:
            # 更新轨道卫星的状态start_tstamp 相当于重启
            for sat in orbit.satellites:
                if (sat.records[sat.visible_idx].start_tstamp <= per_epoch_start_time) & \
                    (sat.records[sat.visible_idx].stop_tstamp > per_epoch_start_time):
                    sat.records[sat.visible_idx].start_time = per_epoch_start_time
                sat.visited = 0 # 新的一轮能否作为source node，与上一轮无关

        orbits_end_times = []
        MoRolla.orbit_model_received.clear()
        MoRolla.orbit_model_sended.clear()
        for orbit in orbits:
            # 选好source node
            min_waiting_time = choose_orbit_source(orbit, per_epoch_start_time)    
            MoRolla.orbit_model_sended.add(orbit.orbit_id)
            if orbit.satellites[orbit.source_node_id].records[orbit.satellites[orbit.source_node_id].visible_idx].delta_tstamp > \
                orbit.satellites[orbit.source_node_id].records[orbit.satellites[orbit.source_node_id].visible_idx].com_time_up:
                orbit.source_node_received = 1 # 假设直接收到全局模型
            else: # 可见期太短，没有收到全局模型
                orbit.source_node_received = 0

            # 仿真一下，构造neighbor
            vir_src_propagation(orbit)

            handle_idle_sat(orbit, per_epoch_start_time)

            # 选好sink node
            choose_orbit_sink(orbit)
            three_step_total_delay = vir_three_step_dca_time(orbit)
            # print(three_step_total_delay)

            orbit_three_over_time = per_epoch_start_time + min_waiting_time +\
                        orbit.satellites[orbit.source_node_id].records[orbit.satellites[orbit.source_node_id].visible_idx].com_time_up + three_step_total_delay

            orbit_end_time = sink_send_model_PS(orbit, orbit_three_over_time)

            # print("Orbit", orbit.orbit_id, ", source node:", orbit.source_node_id, "sink node:", orbit.sink_node_id)
            # t1 = time.localtime(per_epoch_start_time)
            # t2 = time.localtime(orbit_end_time)
            # t1 = time.strftime("%Y-%m-%d %H:%M:%S",t1)
            # t2 = time.strftime("%Y-%m-%d %H:%M:%S", t2)
            # t3 = time.localtime(orbit_three_over_time - three_step_total_delay)
            # t3 = time.strftime("%Y-%m-%d %H:%M:%S",t3)
            # t4 = time.localtime(orbit_three_over_time)
            # t4 = time.strftime("%Y-%m-%d %H:%M:%S",t4)
            # print("Orbit", orbit.orbit_id, "epoch start time:", t1)
            # print("training start time", t3, "training end time:",t4, "this orbit epoch end time:", t2)
            MoRolla.orbit_model_received.add(orbit.orbit_id)
            orbits_end_times.append(orbit_end_time)
        
        # 选择五个轨道里结束时间最晚的，一轮训练结束
        per_epoch_start_time = max(orbits_end_times)
        t_end = time.localtime(per_epoch_start_time)
        t_end = time.strftime("%Y-%m-%d %H:%M:%S",t_end)
        print("Epoch No."+str(round), "training is over, now it is ", t_end)
        # if len(MoRolla.orbit_model_received) == total_orbits_cnt:
        round += 1       

    return per_epoch_start_time


if __name__ == '__main__':
    # 获取必要的数据存储到结构体中
    read_time_file()
    read_range1_file()
    read_range2_file()

    satellites = satellites[1:]

    sat_belongto_orbit()

    # 开始训练过程
    stamp2 = vir_train_process(20, stk_train_start_stamp)
    t1 = time.localtime(stk_train_start_stamp)
    t2 = time.localtime(stamp2)
    t1 = time.strftime("%Y-%m-%d %H:%M:%S",t1)
    t2 = time.strftime("%Y-%m-%d %H:%M:%S", t2)
    time1 = datetime.datetime.strptime(t1,"%Y-%m-%d %H:%M:%S")
    time2 = datetime.datetime.strptime(t2, "%Y-%m-%d %H:%M:%S")
    print("constellation StarLink training start time:", time1)
    print("constellation StarLink training end time:", time2)
