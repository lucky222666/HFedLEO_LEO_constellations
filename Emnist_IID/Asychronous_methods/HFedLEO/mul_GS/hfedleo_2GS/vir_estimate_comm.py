import time
import datetime
import ast
import math
from utils.satcom_helper import *


def read_time_file_init(filename, base_visible_idx):    
    sat_idx = 0
    tmpsat = Satellite()
    tmpsat.orbit_id = tmpsat.plane_id = -1

    fileHandler = open(filename, "r")
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
                tmpsat.total_visible_times[0] = int(slice)
            elif i == 2:
                tmpsat.visible_idx = base_visible_idx + int(slice)
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


def read_time_file_add(filename, base_visible_idx):    
    pre_orbit = -1
    pre_plane = -1
    sat_idx = 0

    fileHandler = open(filename, "r")
    while True:
    # Get next line from file
        line = fileHandler.readline()
        # If line is empty then end of file reached
        if not line:
            break;
        line = line.strip()
        slices = line.split(':', 3)
              
        tmprcd = Record()
        for i, slice in enumerate(slices):
            if i == 0:
                # 例如Sat0_0
                orbit_id = int(slice.split('_')[0][3:])
                plane_id = int(slice.split('_')[1])
                if (pre_orbit != orbit_id) | (pre_plane != plane_id):
                    sat_idx = orbit_id * total_planes_cnt + plane_id + 1
                    pre_orbit = orbit_id
                    pre_plane = plane_id
            elif i == 1:
                total_visible_times = int(slice)
            elif i == 2:
                visible_idx = base_visible_idx + int(slice)
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
                satellites[sat_idx].records[visible_idx] = tmprcd
                satellites[sat_idx].total_visible_times[int(base_visible_idx / per_gs_visible_times)] = total_visible_times
    
    fileHandler.close()
    
    
def read_range1_file_add(filename, base_visible_idx):  
    pre_orbit = -1
    pre_plane = -1
    sat_idx = 0

    fileHandler = open(filename, "r")
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
                # 例如Sat0_0
                orbit_id = int(slice.split('_')[0][3:])
                plane_id = int(slice.split('_')[1])
                if (pre_orbit != orbit_id) | (pre_plane != plane_id):
                    sat_idx = orbit_id * total_planes_cnt + plane_id + 1
                    pre_orbit = orbit_id
                    pre_plane = plane_id
            # elif i == 1:
                # assert satellites[sat_idx].total_visible_times == int(slice)
            elif i == 2:
                visible_idx = base_visible_idx + int(slice)
            elif i == 3:
                satellites[sat_idx].records[visible_idx].range = ast.literal_eval(slice[:-2])
                tp = satellites[sat_idx].records[visible_idx].range / Speed_of_light
                # 直接以无量纲形式计算SNR
                SNR_up = (Tansmission_power*Gain_Antenna*Gain_Antenna) / (Boltzmann_constant*Noise_temperature*Bandwidth_up*pathloss_K_GS(satellites[sat_idx].records[visible_idx].range))
                SNR_down = (Tansmission_power*Gain_Antenna*Gain_Antenna) / (Boltzmann_constant*Noise_temperature*Bandwidth_down*pathloss_K_GS(satellites[sat_idx].records[visible_idx].range))
    
                # 信噪比
                tt_uplink = Size_of_model / (Bandwidth_up *  math.log(1+SNR_up, 2))
                tt_downlink = (Size_of_model + get_size(OrMeta_Down())) / (Bandwidth_up *  math.log(1+SNR_down, 2))
                satellites[sat_idx].records[visible_idx].com_time_up = tp + tt_uplink
                satellites[sat_idx].records[visible_idx].com_time_down = tp + tt_downlink # tt_downlink 目前尚不考虑多颗卫星并行抢占带宽的可能
                # print("uplink transmission delay, downlink transmission delay and propagation delay: ", tt_uplink, tt_downlink, tp)
    fileHandler.close()
    return satellites


def read_range2_file_add():
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
                    sat_idx += 1


def sat_belongto_orbit():
    for sat in satellites:
        orbits[sat.orbit_id].orbit_id = sat.orbit_id
        orbits[sat.orbit_id].satellites[sat.plane_id] = sat


def vir_train_init():
    for orbit in orbits:
        for sat in orbit.satellites:
            sat.visible_idx = 0


def choose_orbit_pair_GS_up(orbit, current_time):
    # idx_0-3 更新到satellite对不同GS最近的时间
    for sat in orbit.satellites:
        for i in range(number_of_gs):
            while sat.records[sat.gs_visible_idxs[i]].stop_tstamp < current_time:
                sat.gs_visible_idxs[i] += 1    
            
    # 针对若干个GS，一一遍历所有卫星的可见情况
    flag_cur_visits = [0 for _ in range(number_of_gs)]
    sat_cur_visits = [Satellite() for _ in range(number_of_gs)]
    for i in range(number_of_gs):
        for sat in orbit.satellites:
            if (current_time >= sat.records[sat.gs_visible_idxs[i]].start_tstamp) & \
                (current_time < sat.records[sat.gs_visible_idxs[i]].stop_tstamp) & (sat.visited == 0): # visted用来防止那些可见期太短的卫星反复被选中
                if flag_cur_visits[i] == 0:
                    flag_cur_visits[i] = 1
                    sat_cur_visits[i] = sat
                elif (sat.records[sat.gs_visible_idxs[i]].com_time_up + sat.spin_around_me_time) \
                      < (sat_cur_visits[i].records[sat_cur_visits[i].gs_visible_idxs[i]].com_time_up + sat_cur_visits[i].spin_around_me_time):   # flag_cur_visits[i] = 1    
                    sat_cur_visits[i] = sat  
                
    
    # 有正在过顶的GS
    min_broadcast_time = stk_train_end_stamp - current_time
    if 1 in flag_cur_visits:
        for i in range(number_of_gs):
            if (flag_cur_visits[i] == 1) & ((sat_cur_visits[i].records[sat_cur_visits[i].gs_visible_idxs[i]].com_time_up + \
                                            sat_cur_visits[i].spin_around_me_time) < min_broadcast_time):
                min_broadcast_time = sat_cur_visits[i].records[sat_cur_visits[i].gs_visible_idxs[i]].com_time_up + sat_cur_visits[i].spin_around_me_time
                orbit.com_gs_id_up = i 
        return       
    
    # 轨道对若干个GS此时都不过顶，那就选择最早到可见期的GS
    min_waiting_time = stk_train_end_stamp - current_time
    min_waiting_times = [min_waiting_time for _ in range(number_of_gs)]
    for i in range(number_of_gs):
        for sat in orbit.satellites:       
            if ((sat.records[sat.gs_visible_idxs[i]].start_tstamp - current_time) <= min_waiting_times[i]) & (sat.visited == 0):
                min_waiting_times[i] = sat.records[sat.gs_visible_idxs[i]].start_tstamp - current_time

    for i in range(number_of_gs):
        if min_waiting_times[i] <= min_waiting_time:
            orbit.com_gs_id_up = i
            min_waiting_time = min_waiting_times[i]
            

def choose_orbit_pair_GS_down(orbit, current_time):
    # idx_0-3 更新到satellite对不同GS最近的时间
    for sat in orbit.satellites:
        for i in range(number_of_gs):
            while sat.records[sat.gs_visible_idxs[i]].stop_tstamp < current_time:
                sat.gs_visible_idxs[i] += 1
    
    # 针对若干个GS，一一遍历所有卫星的可见情况
    flag_cur_visits = [0 for _ in range(number_of_gs)]
    sat_cur_visits = [Satellite() for _ in range(number_of_gs)]
    for i in range(number_of_gs):
        for sat in orbit.satellites:
            if (current_time >= sat.records[sat.gs_visible_idxs[i]].start_tstamp) & \
                (current_time < sat.records[sat.gs_visible_idxs[i]].stop_tstamp):
                if flag_cur_visits[i] == 0:
                    flag_cur_visits[i] = 1
                    sat_cur_visits[i] = sat
                elif (sat.records[sat.gs_visible_idxs[i]].com_time_down + sat.spin_around_me_time) < \
                    (sat_cur_visits[i].records[sat_cur_visits[i].gs_visible_idxs[i]].com_time_down + sat_cur_visits[i].spin_around_me_time):   # flag_cur_visits[i] = 1    
                    sat_cur_visits[i] = sat  
                
    
    # 有正在过顶的GS
    min_aggregation_time = stk_train_end_stamp - current_time
    if 1 in flag_cur_visits:
        for i in range(number_of_gs):
            if (flag_cur_visits[i] == 1) & ((sat_cur_visits[i].records[sat_cur_visits[i].gs_visible_idxs[i]].com_time_down +\
                                             sat_cur_visits[i].spin_around_me_time) < min_aggregation_time):
                min_aggregation_time = sat_cur_visits[i].records[sat_cur_visits[i].gs_visible_idxs[i]].com_time_down + sat_cur_visits[i].spin_around_me_time
                orbit.com_gs_id_down = i 
        return       
    
    # 轨道对若干个GS此时都不过顶，那就选择最早到可见期的GS
    min_waiting_time = stk_train_end_stamp - current_time
    min_waiting_times = [min_waiting_time for _ in range(number_of_gs)]
    for i in range(number_of_gs):
        for sat in orbit.satellites:       
            if (sat.records[sat.gs_visible_idxs[i]].start_tstamp - current_time) <= min_waiting_times[i]:
                min_waiting_times[i] = sat.records[sat.gs_visible_idxs[i]].start_tstamp - current_time

    for i in range(number_of_gs):
        if min_waiting_times[i] <= min_waiting_time:
            orbit.com_gs_id_down = i
            min_waiting_time = min_waiting_times[i]


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
        sat.visible_idx = sat.gs_visible_idxs[orbit.com_gs_id_up]
        while sat.records[sat.visible_idx].stop_tstamp < current_time:
            sat.visible_idx += 1

    # 优先选择正在可见期的卫星
    min_broadcast_time = stk_train_end_stamp - current_time
    for sat in orbit.satellites:
        if (current_time >= sat.records[sat.visible_idx].start_tstamp) & \
            (current_time < sat.records[sat.visible_idx].stop_tstamp) & (sat.visited == 0): # visted用来防止那些可见期太短的卫星反复被选中
            if (sat.records[sat.visible_idx].com_time_up + sat.spin_around_me_time) < min_broadcast_time:
                if min_broadcast_time != (stk_train_end_stamp - current_time):
                    orbit.satellites[orbit.source_node_id].visited = 0
            
                min_broadcast_time = sat.records[sat.visible_idx].com_time_up + sat.spin_around_me_time
                orbit.source_node_id = sat.plane_id
                sat.visited = 1
    if min_broadcast_time != (stk_train_end_stamp - current_time):
        return (0 + timeline_advancement)
    
        
    # 退而求其次选择最早即将到可见期的卫星
    min_waiting_time = orbit.satellites[0].records[orbit.satellites[0].visible_idx].start_tstamp - current_time
    for sat in orbit.satellites:
        if ((sat.records[sat.visible_idx].start_tstamp - current_time) <= min_waiting_time) & (sat.visited == 0):
            min_waiting_time = sat.records[sat.visible_idx].start_tstamp - current_time
            orbit.source_node_id = sat.plane_id
            sat.visited = 1
    return (min_waiting_time + timeline_advancement)


def estimate_DC_over_time(orbit):
    t_n = orbit.satellites[orbit.source_node_id].records[orbit.satellites[orbit.source_node_id].visible_idx].start_tstamp + orbit.satellites[orbit.source_node_id].records[orbit.satellites[orbit.source_node_id].visible_idx].com_time_up
    T_e_p = cal_time_e_p(orbit)
    current_time = t_n + T_e_p
    return current_time


def choose_orbit_sink(orbit, current_time):
    # 确保轨道的所有卫星都在当前时间附近，没有滞后
    for sat in orbit.satellites:
        sat.visible_idx = sat.gs_visible_idxs[orbit.com_gs_id_down]
        while sat.records[sat.visible_idx].stop_tstamp <= current_time:
            sat.visible_idx += 1

        # 优先选择正在可见期的卫星
        min_aggregation_time = stk_train_end_stamp - current_time
        for sat in orbit.satellites:
            if (current_time >= sat.records[sat.visible_idx].start_tstamp) & \
                (current_time < sat.records[sat.visible_idx].stop_tstamp):
                if (sat.records[sat.visible_idx].com_time_down + sat.spin_around_me_time) < min_aggregation_time:
                    min_aggregation_time = sat.records[sat.visible_idx].com_time_down + sat.spin_around_me_time
                    orbit.sink_node_id = sat.plane_id
        if min_aggregation_time != (stk_train_end_stamp - current_time):
            return
            
        # 退而求其次选择最早即将到可见期的卫星
        min_waiting_time = stk_train_end_stamp - current_time
        for sat in orbit.satellites:
            if (sat.records[sat.visible_idx].start_tstamp - current_time) <= min_waiting_time:
                min_waiting_time = sat.records[sat.visible_idx].start_tstamp - current_time
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
            if (orbit.orbit_id in Kernel_GS.orbit_model_sended) & (orbit.source_node_received == 1):
                continue
            # 全局模型已经发送 & 没有收到，说明可见期太短了，这个source node不合格（第三种情况）
            elif (orbit.orbit_id in Kernel_GS.orbit_model_sended) & (orbit.source_node_received == 0):
                while orbit.satellites[orbit.source_node_id].records[orbit.satellites[orbit.source_node_id].visible_idx].delta_tstamp \
                    < orbit.satellites[orbit.source_node_id].records[orbit.satellites[orbit.source_node_id].visible_idx].com_time_up:
                    choose_orbit_pair_GS_up(orbit, current_time)
                    min_waiting_time = choose_orbit_source(orbit, current_time)
                    current_time += min_waiting_time


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
        choose_orbit_pair_GS_down(orbit, current_time)  # 立即重新选择GS
        min_waiting_time = choose_orbit_sink_again(orbit, current_time) # 有可能遇到可见期的直接发 0；也有可能需要等待
        orbit_end = current_time + min_waiting_time + sat_to_sat_delay(orbit, pre_sink_node, orbit.sink_node_id) + \
                    orbit.satellites[orbit.sink_node_id].records[orbit.satellites[orbit.sink_node_id].visible_idx].com_time_down
    return orbit_end


def read_participated_orbits(filename, orbit_idxs):
    try:
        #打开文件
        fp = open(filename, "r")
        line_no = 1
        for line in fp.readlines(): 
            if line_no > 1: # 只读一行
                break       
            line = line.replace('\n','')
            line = line.split(',')
            #以逗号为分隔符把数据转化为列表
            for o in line:
                if o[0] == '[':
                    o = o[1:]
                elif o[-1] == ']':
                    o = o[:-1]
                orbit_idxs.append(int(o))
            line_no += 1
        fp.close()
    except IOError:
        print("file open error, %s is not existing" % filename)


def handle_rest_orbits(orbit, per_epoch_start_time, orbits_end_times):
    sink_visible_idx = orbit.satellites[orbit.sink_node_id].visible_idx
    # 还来得及将全局模型在本地测试集的准确率计算结果捎带下来
    if (orbit.training_over_time - per_epoch_start_time) > orbit.satellites[orbit.sink_node_id].records[sink_visible_idx].com_time_up:         
            real_end_time = orbit.training_over_time
            orbits_end_times.append(real_end_time)
    # 来不及将全局模型在本地测试集的准确率计算结果捎带下来，需要单独再走一遍 
    else:   
        # print("时间太短, accuracy来不及计算了, 辛苦再通信一趟吧")
        # 再跑一趟，可以，如果sink sat可见期时间允许，那么当前可见期就可以完成
        if (orbit.satellites[orbit.sink_node_id].records[sink_visible_idx].stop_tstamp - per_epoch_start_time) > \
            (orbit.satellites[orbit.sink_node_id].records[sink_visible_idx].com_time_up + orbit.satellites[orbit.sink_node_id].records[sink_visible_idx].com_time_down):
            real_end_time = orbit.training_over_time + \
                orbit.satellites[orbit.sink_node_id].records[sink_visible_idx].com_time_up + \
                orbit.satellites[orbit.sink_node_id].records[sink_visible_idx].com_time_down
            orbits_end_times.append(real_end_time)
        else:   # 如果sink sat可见期时间不允许，那么就需要重新找该轨道的下一个sink sat
            min_waiting_time = choose_orbit_sink_again(orbit, per_epoch_start_time)
            real_end_time = per_epoch_start_time + min_waiting_time + \
                    orbit.satellites[orbit.sink_node_id].records[orbit.satellites[orbit.sink_node_id].visible_idx].com_time_up + \
                    orbit.satellites[orbit.sink_node_id].records[orbit.satellites[orbit.sink_node_id].visible_idx].com_time_down
            orbits_end_times.append(real_end_time)

    Kernel_GS.order_rcvd_orbit_id.append(orbit.orbit_id)
    Kernel_GS.order_rcvd_orbit_time.append(real_end_time)


# 以轨道为单位进行逻辑处理
def vir_train_process(history_selected_orbits, epochs_needed, fraction, vir_start_time, gid, filename):
    per_epoch_start_time = vir_start_time
    round = 0

    # 参与训练的轨道不再是自己决定
    selected_orbits = []
    if os.path.exists(filename):
        orbit_idxs = []
        read_participated_orbits(filename, orbit_idxs)
        for idx in orbit_idxs:
            selected_orbits.append(orbits[idx])
    else:
        selected_orbits = orbits

    with open(filename,'w') as f: # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
        while round < epochs_needed:
            for orbit in selected_orbits:
                # 记录该轨道的当前模型编号
                orbit.model_round_id = gid

                # 更新轨道卫星的状态start_tstamp 相当于重启
                for sat in orbit.satellites:            
                    # 消除因为时间轴走的过快带来的影响
                    for i in range(number_of_gs):
                        # 应该弃用的轨道，时间倒退 索引回退
                        while (sat.records[sat.gs_visible_idxs[i]].start_tstamp > per_epoch_start_time) & (sat.gs_visible_idxs[i] > (i*per_gs_visible_times)):
                            sat.gs_visible_idxs[i] -= 1

                    # 裁剪那些卡在当前时间的可见期
                    for i in range(number_of_gs):
                        if (sat.records[sat.gs_visible_idxs[i]].start_tstamp <= per_epoch_start_time) & \
                            (sat.records[sat.gs_visible_idxs[i]].stop_tstamp > per_epoch_start_time):
                            sat.records[sat.gs_visible_idxs[i]].start_tstamp = per_epoch_start_time
                            sat.records[sat.gs_visible_idxs[i]].delta_tstamp = sat.records[sat.gs_visible_idxs[i]].stop_tstamp - sat.records[sat.gs_visible_idxs[i]].start_tstamp

                    sat.visited = 0 # 新的一轮能否作为source node，与上一轮无关
                    

            orbits_end_times = []
            Kernel_GS.orbit_model_received.clear()
            Kernel_GS.orbit_model_sended.clear()
            Kernel_GS.order_rcvd_orbit_id.clear()
            Kernel_GS.order_rcvd_orbit_time.clear()
            for orbit in selected_orbits:
                # 选好该轨道本轮使用的上行GS
                choose_orbit_pair_GS_up(orbit, per_epoch_start_time)
                # 选好source node
                min_waiting_time = choose_orbit_source(orbit, per_epoch_start_time)    
                Kernel_GS.orbit_model_sended.add(orbit.orbit_id)
                if orbit.satellites[orbit.source_node_id].records[orbit.satellites[orbit.source_node_id].visible_idx].delta_tstamp > \
                    orbit.satellites[orbit.source_node_id].records[orbit.satellites[orbit.source_node_id].visible_idx].com_time_up:
                    orbit.source_node_received = 1 # 假设直接收到全局模型
                else: # 可见期太短，没有收到全局模型
                    orbit.source_node_received = 0

                # 仿真一下，构造neighbor
                vir_src_propagation(orbit)

                handle_idle_sat(orbit, per_epoch_start_time)

                predicted_orbit_DCA_over = estimate_DC_over_time(orbit)            
                # 选好该轨道本轮使用的下行GS         
                choose_orbit_pair_GS_down(orbit, predicted_orbit_DCA_over)
                # 选好sink node
                choose_orbit_sink(orbit, predicted_orbit_DCA_over)
                three_step_total_delay = vir_three_step_dca_time(orbit)

                orbit_three_over_time = per_epoch_start_time + min_waiting_time +\
                            orbit.satellites[orbit.source_node_id].records[orbit.satellites[orbit.source_node_id].visible_idx].com_time_up + three_step_total_delay

                orbit_end_time = sink_send_model_PS(orbit, orbit_three_over_time)
                
                Kernel_GS.orbit_model_received.add(orbit.orbit_id)
                orbits_end_times.append(orbit_end_time)
            
                Kernel_GS.order_rcvd_orbit_id.append(orbit.orbit_id)
                Kernel_GS.order_rcvd_orbit_time.append(orbit_end_time)
                orbit.training_over_time = orbit_end_time


            # 上面处理了80%的轨道 还剩下20%的轨道可以继续训练
            rest_orbits = orbits.copy()
            for o in selected_orbits:
                rest_orbits.remove(o)

            for orbit in rest_orbits:
                handle_rest_orbits(orbit, per_epoch_start_time, orbits_end_times)


            # 选择72个轨道里结束时间在选取比例中最晚的，一轮训练结束
            orbits_end_times2 = orbits_end_times.copy() # 复制一份，不破坏原来的列表
            orbits_end_times2.sort()    # 从小到大排序
            
            # 导出参与训练的卫星
            merge_list = list(zip(Kernel_GS.order_rcvd_orbit_id, Kernel_GS.order_rcvd_orbit_time))
            # 指定第二个元素排序
            merge_list.sort(key=takeSecond)
            merge_list = merge_list[:math.ceil(total_orbits_cnt * fraction)]
           
            random_orbits = [i[0] for i in merge_list]

        
            if fraction != 1:
                per_epoch_start_time = orbits_end_times2[math.ceil(total_orbits_cnt * fraction)-1]
            else:
                per_epoch_start_time = max(orbits_end_times)

            # 重新用轨道的source node的start_tstamp 比较 per_epoch_start_time
            if gid != 0:    # 说明不是第一轮
                for orbit in selected_orbits:
                    if (orbit not in random_orbits) & (orbit.satellites[orbit.source_node_id].\
                                                       records[orbit.satellites[orbit.source_node_id].visible_idx].start_tstamp > per_epoch_start_time):  # 上一轮选中了，但是这一轮没有被选中
                        history_selected_orbits.append(orbit)
            
                for orbit in history_selected_orbits:
                    # 如果轨道的可见期在本轮范围内，那么即可接收本轮模型
                    if orbit.satellites[orbit.source_node_id].\
                        records[orbit.satellites[orbit.source_node_id].visible_idx].start_tstamp < per_epoch_start_time:
                        orbit.model_round_id = gid
                        history_selected_orbits.remove(orbit)


            # 收集这些轨道的轮次
            orbit_round_ids = []
            for o in random_orbits:
                orbit_round_ids.append(orbits[o].model_round_id)

            # 记录某一轮在当前fraction下参与训练的轨道编号 & 对应的模型编号
            f.write(str(random_orbits) + "\n")
            f.write(str(orbit_round_ids) + "\n")
            
            round += 1       

    return per_epoch_start_time


def vir_comm_main(history_selected_orbits, fraction, epoch_start_stamp, gid):  
    epoch_end_stamp = vir_train_process(history_selected_orbits, 1, fraction, epoch_start_stamp, gid, "./output_info/participants.txt")
        
    t1 = time.localtime(epoch_start_stamp)
    t2 = time.localtime(epoch_end_stamp)
    t1 = time.strftime("%Y-%m-%d %H:%M:%S", t1)
    t2 = time.strftime("%Y-%m-%d %H:%M:%S", t2)
    time1 = datetime.datetime.strptime(t1,"%Y-%m-%d %H:%M:%S")
    time2 = datetime.datetime.strptime(t2, "%Y-%m-%d %H:%M:%S")
    print("constellation StarLink training start time:", time1)
    print("constellation StarLink training end time:", time2)
    return epoch_end_stamp
