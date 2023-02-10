import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D


def dot_matrix(x, y, z, l):
    """
    :param x: x轴方向，试件的长度
    :param y: y轴方向，试件的长度
    :param z: z轴方向，试件的长度
    :param l: 每一根梁的长度
    :return: 把点阵分成343个block，每个block里面所有的点，是一个3维数组
    """
    dot_list = np.array([])
    for i in range(0, int(x / l)):
        for j in range(0, int(y / l)):
            for k in range(0, int(z / l)):
                x_0, y_0, z_0 = 0 + i * l, 0 + j * l, 0 + k * l
                x_1, y_1, z_1 = x_0, y_0 + l, z_0
                x_2, y_2, z_2 = x_0 + l, y_0 + l, z_0
                x_3, y_3, z_3 = x_0 + l, y_0, z_0
                x_4, y_4, z_4 = x_0, y_0, z_0 + l
                x_5, y_5, z_5 = x_0, y_0 + l, z_0 + l
                x_6, y_6, z_6 = x_0 + l, y_0 + l, z_0 + l
                x_7, y_7, z_7 = x_0 + l, y_0, z_0 + l
                x_8, y_8, z_8 = (x_0 + x_2) / 2.0, (y_0 + y_2) / 2.0, (z_0 + z_2) / 2.0
                x_9, y_9, z_9 = (x_1 + x_6) / 2.0, (y_1 + y_6) / 2.0, (z_1 + z_6) / 2.0
                x_10, y_10, z_10 = (x_5 + x_7) / 2.0, (y_5 + y_7) / 2.0, (z_5 + z_7) / 2.0
                x_11, y_11, z_11 = (x_3 + x_4) / 2.0, (y_3 + y_4) / 2.0, (z_3 + z_4) / 2.0
                x_12, y_12, z_12 = (x_2 + x_7) / 2.0, (y_2 + y_7) / 2.0, (z_2 + z_7) / 2.0
                x_13, y_13, z_13 = (x_0 + x_5) / 2.0, (y_0 + y_5) / 2.0, (z_0 + z_5) / 2.0
                x_14, y_14, z_14 = (x_1 + x_7) / 2.0, (y_1 + y_7) / 2.0, (z_1 + z_7) / 2.0
                n_0 = np.array([x_0, y_0, z_0])
                n_1 = np.array([x_1, y_1, z_1])
                n_2 = np.array([x_2, y_2, z_2])
                n_3 = np.array([x_3, y_3, z_3])
                n_4 = np.array([x_4, y_4, z_4])
                n_5 = np.array([x_5, y_5, z_5])
                n_6 = np.array([x_6, y_6, z_6])
                n_7 = np.array([x_7, y_7, z_7])
                n_8 = np.array([x_8, y_8, z_8])
                n_9 = np.array([x_9, y_9, z_9])
                n_10 = np.array([x_10, y_10, z_10])
                n_11 = np.array([x_11, y_11, z_11])
                n_12 = np.array([x_12, y_12, z_12])
                n_13 = np.array([x_13, y_13, z_13])
                n_14 = np.array([x_14, y_14, z_14])
                dot_list = np.append(dot_list, [n_0, n_1, n_2, n_3, n_4, n_5, n_6, n_7, n_8, n_9, n_10, n_11, n_12, n_13, n_14])
                dot_matrix_list = dot_list.reshape(int(len(dot_list) / 45), 15, 3)
    return dot_matrix_list


def index_return(array_new, array_old):
    index_array = np.array([])
    for i in range(0, len(array_new)):
        for j in range(0, len(array_old)):
            if all(array_new[i] == array_old[j]):
                index = j
                index_array = np.append(index_array, index)
            else:
                pass
    return index_array


def stress_rotation(sig_direction):
    s = []
    a0 = sig_direction[0]
    b0 = sig_direction[1]
    c0 = sig_direction[2]
    a = a0 / np.sqrt(a0 ** 2 + b0 ** 2 + c0 ** 2)
    b = b0 / np.sqrt(a0 ** 2 + b0 ** 2 + c0 ** 2)
    c = c0 / np.sqrt(a0 ** 2 + b0 ** 2 + c0 ** 2)
    v_i = np.array([a, b, c])
    if b == 0.0 and c == 0.0:
        v_j = np.array([0, 1.0, 0])
    else:
        v_j = np.array([0, c / np.sqrt(b ** 2 + c ** 2), -b / np.sqrt(b ** 2 + c ** 2)])
    v_k = np.cross(v_i, v_j)
    v_k = normal_v(v_k)
    s.append(v_i)
    s.append(v_j)
    s.append(v_k)
    lam = np.array(s)
    #print(lam)
    sig_o = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    sig_n = np.dot(np.dot(lam.T, sig_o), lam)
    sig_v = np.array([sig_n[0][0], sig_n[1][1], sig_n[2][2], sig_n[0][1], sig_n[1][2], sig_n[0][2]])
    return sig_v

def strain_rotation(eps_v, sig_direction):
    eps_o = np.array([[eps_v[0], 0.5 * eps_v[3], 0.5 * eps_v[5]], [0.5 * eps_v[3], eps_v[1], 0.5 * eps_v[4]], [0.5 * eps_v[5], 0.5 * eps_v[4], eps_v[2]]])
    s = []
    a0 = sig_direction[0]
    b0 = sig_direction[1]
    c0 = sig_direction[2]
    a = a0 / np.sqrt(a0 ** 2 + b0 ** 2 + c0 ** 2)
    b = b0 / np.sqrt(a0 ** 2 + b0 ** 2 + c0 ** 2)
    c = c0 / np.sqrt(a0 ** 2 + b0 ** 2 + c0 ** 2)
    v_i = np.array([a, b, c])
    if b == 0.0 and c == 0.0:
        v_j = np.array([0, 1.0, 0])
    else:
        v_j = np.array([0, c / np.sqrt(b ** 2 + c ** 2), -b / np.sqrt(b ** 2 + c ** 2)])
    v_k = np.cross(v_i, v_j)
    v_k = normal_v(v_k)
    s.append(v_i)
    s.append(v_j)
    s.append(v_k)
    lam = np.array(s)
    # print(lam)
    eps_n = np.dot(np.dot(lam, eps_o), lam.T)
    eps = np.array([eps_n[0][0], eps_n[1][1], eps_n[2][2], 2 * eps_n[0][1], 2 * eps_n[1][2], 2 * eps_n[0][2]])
    return eps

def bcc_connection(cell):
    """
    :param cell: 包含BCC胞元所有点的集合，是一个二维数组
    :return: 起始点数组（2维），终止点数组（2维），连接对（3维）
    """
    start_point = np.array([cell[0], cell[1], cell[2], cell[3],
                          cell[4], cell[5], cell[6], cell[7]])
    end_point = np.array([cell[14], cell[14], cell[14], cell[14],
                            cell[14], cell[14], cell[14], cell[14]])
    connection_pair = np.array([])
    for i in range(0, len(start_point)):
        connection_element = np.array([start_point[i], end_point[i]])
        connection_pair = np.append(connection_pair, connection_element)
    connection_pair.resize(int(len(connection_pair) / 6.0), 2, 3)
    return connection_pair


def oct_connection(cell):
    start_point = np.array([cell[8], cell[8], cell[8], cell[8],
                            cell[9], cell[9], cell[9], cell[9],
                            cell[10], cell[10], cell[10], cell[10],
                            cell[11], cell[11], cell[11], cell[11],
                            cell[12], cell[12], cell[12], cell[12],
                            cell[13], cell[13], cell[13], cell[13],
                            cell[8], cell[8], cell[8], cell[8],
                            cell[10], cell[10], cell[10], cell[10],
                            cell[9], cell[13], cell[11], cell[12]])
    end_point = np.array([cell[0], cell[1], cell[2], cell[3],
                            cell[1], cell[2], cell[6], cell[5],
                            cell[4], cell[5], cell[6], cell[7],
                            cell[0], cell[4], cell[7], cell[3],
                            cell[3], cell[2], cell[6], cell[7],
                            cell[0], cell[1], cell[5], cell[4],
                            cell[9], cell[13], cell[11], cell[12],
                            cell[9], cell[13], cell[11], cell[12],
                            cell[13], cell[11], cell[12], cell[9]])
    connection_pair = np.array([])
    for i in range(0, len(start_point)):
        connection_element = np.array([start_point[i], end_point[i]])
        connection_pair = np.append(connection_pair, connection_element)
    connection_pair.resize(int(len(connection_pair) / 6.0), 2, 3)
    return connection_pair



def sc_connection(cell):
    start_point = np.array([cell[0], cell[1], cell[2], cell[3],
                            cell[4], cell[5], cell[6], cell[7],
                            cell[0], cell[1], cell[2], cell[3]])
    end_point = np.array([cell[1], cell[2], cell[3], cell[0],
                          cell[5], cell[6], cell[7], cell[4],
                          cell[4], cell[5], cell[6], cell[7]])
    connection_pair = np.array([])
    for i in range(0, len(start_point)):
        connection_element = np.array([start_point[i], end_point[i]])
        connection_pair = np.append(connection_pair, connection_element)
    connection_pair.resize(int(len(connection_pair) / 6.0), 2, 3)
    return connection_pair

def scoct_connection(cell):
    start_point = np.array([cell[8], cell[8], cell[8], cell[8],
                            cell[9], cell[9], cell[9], cell[9],
                            cell[10], cell[10], cell[10], cell[10],
                            cell[11], cell[11], cell[11], cell[11],
                            cell[12], cell[12], cell[12], cell[12],
                            cell[13], cell[13], cell[13], cell[13],
                            cell[8], cell[8], cell[8], cell[8],
                            cell[10], cell[10], cell[10], cell[10],
                            cell[9], cell[13], cell[11], cell[12],
                            cell[0], cell[1], cell[2], cell[3],
                            cell[4], cell[5], cell[6], cell[7],
                            cell[0], cell[1], cell[2], cell[3]
                            ])
    end_point = np.array([cell[0], cell[1], cell[2], cell[3],
                            cell[1], cell[2], cell[6], cell[5],
                            cell[4], cell[5], cell[6], cell[7],
                            cell[0], cell[4], cell[7], cell[3],
                            cell[3], cell[2], cell[6], cell[7],
                            cell[0], cell[1], cell[5], cell[4],
                            cell[9], cell[13], cell[11], cell[12],
                            cell[9], cell[13], cell[11], cell[12],
                            cell[13], cell[11], cell[12], cell[9],
                          cell[1], cell[2], cell[3], cell[0],
                          cell[5], cell[6], cell[7], cell[4],
                          cell[4], cell[5], cell[6], cell[7]
                          ])
    connection_pair = np.array([])
    for i in range(0, len(start_point)):
        connection_element = np.array([start_point[i], end_point[i]])
        connection_pair = np.append(connection_pair, connection_element)
    connection_pair.resize(int(len(connection_pair) / 6.0), 2, 3)
    return connection_pair

def dual_phase_lattice_oct_bcc(strengthen_phase_index, connection_function_base, connection_function_strengthen):
    cell_list = dot_matrix(70, 70, 70, 10)
    cell_index = np. arange(0, 343)
    base_phase_index = np.setdiff1d(cell_index, strengthen_phase_index)
    connection_pair = np.array([])
    mid_point_list = np.array([])
    for i in range(0, len(cell_index)):
        if any(i == strengthen_phase_index):
            connection_pair_element = connection_function_strengthen(cell_list[i]) # 3维数组
        else:
            connection_pair_element = connection_function_base(cell_list[i])
        connection_pair = np.append(connection_pair, connection_pair_element)
    """
    连接对的建立
    """
    connection_pair_array = connection_pair.reshape(int(len(connection_pair) / 6.0), 2, 3)
    for j in range(0, len(connection_pair_array)):
        x_mid = (connection_pair_array[j, 0, 0] + connection_pair_array[j, 1, 0]) / 2.0
        y_mid = (connection_pair_array[j, 0, 1] + connection_pair_array[j, 1, 1]) / 2.0
        z_mid = (connection_pair_array[j, 0, 2] + connection_pair_array[j, 1, 2]) / 2.0
        mid_point_element = np.array([x_mid, y_mid, z_mid])
        mid_point_list = np.append(mid_point_list, mid_point_element)

    mid_point_list.resize((int(len(mid_point_list) / 3.0), 3))
    mid_point_list_array, index_array, reverse_index_array, counts_array = np.unique(mid_point_list,
                                                                                     axis=0,
                                                                                     return_index=True,
                                                                                     return_inverse=True,
                                                                                     return_counts=True)

    # index_array_unique = np.setdiff1d(np.arange(0, len(connection_pair_array), 1), index_array)
    # connection_pair_array_1 = np.delete(connection_pair_array, index_array_unique, axis=0)
    connection_pair_array_1 = connection_pair_array[index_array]
    """
    扁平化处理
    """
    connection_pair.resize((int(len(connection_pair) / 3.0), 3))
    unique_point_array, u_index_array, u_reverse_index_array, u_counts_array = np.unique(connection_pair,
                                                                                     axis=0,
                                                                                     return_index=True,
                                                                                     return_inverse=True,
                                                                                     return_counts=True)

    """
    生成连接的编号对
    """
    start_point_array = connection_pair_array_1[:, 0]
    end_point_array = connection_pair_array_1[:, 1]
    index_start = np.array([])
    index_end = np.array([])
    for i in range(0, len(end_point_array)):
        for j in range(0, len(unique_point_array)):
            if all(end_point_array[i] == unique_point_array[j]):
                index = j
                index_end = np.append(index_end, index)
            else:
                pass

    for i in range(0, len(start_point_array)):
        for j in range(0, len(unique_point_array)):
            if all(start_point_array[i] == unique_point_array[j]):
                index = j
                index_start = np.append(index_start, index)
            else:
                pass
    index_total = np.array([index_start, index_end])
    index_total_1 = np.unique(index_total)
    return connection_pair_array_1, index_array, reverse_index_array, counts_array,\
           unique_point_array, u_index_array, u_reverse_index_array, u_counts_array,\
           index_start, index_end


def point_classification(point_array, L, connection_pair):
    d4 = np.array([])
    d1 = np.array([])
    d2 = np.array([])
    d21 = np.array([])
    d22 = np.array([])
    d23 = np.array([])
    d3 = np.array([])
    d31 = np.array([])
    d32 = np.array([])
    d33 = np.array([])
    """
    独立点的生成
    """
    for i in range(0, len(point_array)):
        if all(point_array[i] > 0) and all(point_array[i] < L):
            d4_element = point_array[i]
            d4 = np.append(d4, d4_element)
        else:
            if any(point_array[i] == 0):
                """
                判断是不是三个面上的点
                """
                if all(point_array[i] == 0):
                    """
                    判断是不是原点
                    """
                    d1_element = point_array[i]
                    d1 = np.append(d1, d1_element)
                else:
                    """
                    不是原点还是面上的点, 判断是不是棱上的点(去除顶点以外)
                    """
                    if point_array[i][0] == 0 and point_array[i][1] == 0:
                        if all(point_array[i] < L):
                            d21_element = point_array[i]
                            d21 = np.append(d21, d21_element)
                    elif point_array[i][0] == 0 and point_array[i][2] == 0:
                        if all(point_array[i] < L):
                            d22_element = point_array[i]
                            d22 = np.append(d22, d22_element)
                    elif point_array[i][1] == 0 and point_array[i][2] == 0:
                        if all(point_array[i] < L):
                            d23_element = point_array[i]
                            d23 = np.append(d23, d23_element)
                    else:
                        if all(point_array[i] < L):
                            """
                            判断是不是面上的点，去除棱上和顶点
                            """
                            if point_array[i][0] == 0:
                                d31_element = point_array[i]
                                d31 = np.append(d31, d31_element)
                            elif point_array[i][1] == 0:
                                d32_element = point_array[i]
                                d32 = np.append(d32, d32_element)
                            elif point_array[i][2] == 0:
                                d33_element = point_array[i]
                                d33 = np.append(d33, d33_element)

    d2 = np.append(d2, np.array([d21, d22, d23]))  # 棱上的独立点（排号序之后）
    d3 = np.append(d3, np.array([d31, d32, d33]))  # 面上的独立点（排号序之后）

    d1.resize((int(len(d1) / 3.0), 3))


    d2.resize((int(len(d2) / 3.0), 3))
    d21.resize((int(len(d21) / 3.0), 3))
    d22.resize((int(len(d22) / 3.0), 3))
    d23.resize((int(len(d23) / 3.0), 3))

    d3.resize((int(len(d3) / 3.0), 3))
    d31.resize((int(len(d31) / 3.0), 3))
    d32.resize((int(len(d32) / 3.0), 3))
    d33.resize((int(len(d33) / 3.0), 3))

    d4.resize((int(len(d4) / 3.0), 3))

    """
    非独立点的生成
    """
    a1 = np.array([L, 0, 0])
    a2 = np.array([0, L, 0])
    a3 = np.array([0, 0, L])
    """
    其他顶点
    """
    bar_d1_1 = d1[0] + a2
    bar_d1_2 = d1[0] + a1 + a2
    bar_d1_3 = d1[0] + a1
    bar_d1_4 = d1[0] + a3
    bar_d1_5 = d1[0] + a2 + a3
    bar_d1_6 = d1[0] + a1 + a2 + a3
    bar_d1_7 = d1[0] + a1 + a3

    bar_d1 = np.array([])
    bar_d1 = np.append(bar_d1, bar_d1_1)
    bar_d1 = np.append(bar_d1, bar_d1_2)
    bar_d1 = np.append(bar_d1, bar_d1_3)
    bar_d1 = np.append(bar_d1, bar_d1_4)
    bar_d1 = np.append(bar_d1, bar_d1_5)
    bar_d1 = np.append(bar_d1, bar_d1_6)
    bar_d1 = np.append(bar_d1, bar_d1_7)
    bar_d1.resize((int(len(bar_d1) / 3.0), 3))
    """
    第一条棱上的点-其他三条
    """
    bar_d2 = np.array([])

    bar_d21 = np.array([])
    bar_d21_1 = np.array([])
    bar_d21_2 = np.array([])
    bar_d21_3 = np.array([])
    for i in range(0, len(d21)):
        bar_d21_1_element = d21[i] + a2
        bar_d21_2_element = d21[i] + a1 + a2
        bar_d21_3_element = d21[i] + a1
        bar_d21_1 = np.append(bar_d21_1, bar_d21_1_element)
        bar_d21_2 = np.append(bar_d21_2, bar_d21_2_element)
        bar_d21_3 = np.append(bar_d21_3, bar_d21_3_element)
    bar_d21 = np.append(bar_d21, np.array([bar_d21_1, bar_d21_2, bar_d21_3]))
    bar_d21_1.resize((int(len(bar_d21_1) / 3.0), 3))
    bar_d21_2.resize((int(len(bar_d21_2) / 3.0), 3))
    bar_d21_3.resize((int(len(bar_d21_3) / 3.0), 3))
    bar_d21.resize((int(len(bar_d21) / 3.0), 3))

    """
    第二条棱上的点-其他三条
    """
    bar_d22 = np.array([])
    bar_d22_1 = np.array([])
    bar_d22_2 = np.array([])
    bar_d22_3 = np.array([])
    for i in range(0, len(d22)):
        bar_d22_1_element = d22[i] + a1
        bar_d22_2_element = d22[i] + a1 + a3
        bar_d22_3_element = d22[i] + a3
        bar_d22_1 = np.append(bar_d22_1, bar_d22_1_element)
        bar_d22_2 = np.append(bar_d22_2, bar_d22_2_element)
        bar_d22_3 = np.append(bar_d22_3, bar_d22_3_element)
    bar_d22 = np.append(bar_d22, np.array([bar_d22_1, bar_d22_2, bar_d22_3]))
    bar_d22_1.resize((int(len(bar_d22_1) / 3.0), 3))
    bar_d22_2.resize((int(len(bar_d22_2) / 3.0), 3))
    bar_d22_3.resize((int(len(bar_d22_3) / 3.0), 3))
    bar_d22.resize((int(len(bar_d22) / 3.0), 3))

    """
    第三条棱上的点-其他三条
    """
    bar_d23 = np.array([])
    bar_d23_1 = np.array([])
    bar_d23_2 = np.array([])
    bar_d23_3 = np.array([])
    for i in range(0, len(d23)):
        bar_d23_1_element = d23[i] + a2
        bar_d23_2_element = d23[i] + a2 + a3
        bar_d23_3_element = d23[i] + a3
        bar_d23_1 = np.append(bar_d23_1, bar_d23_1_element)
        bar_d23_2 = np.append(bar_d23_2, bar_d23_2_element)
        bar_d23_3 = np.append(bar_d23_3, bar_d23_3_element)
    bar_d23 = np.append(bar_d23, np.array([bar_d23_1, bar_d23_2, bar_d23_3]))
    print(len(bar_d23_1))
    bar_d23_1.resize((int(len(bar_d23_1) / 3.0), 3))
    bar_d23_2.resize((int(len(bar_d23_2) / 3.0), 3))
    bar_d23_3.resize((int(len(bar_d23_3) / 3.0), 3))
    bar_d23.resize((int(len(bar_d23) / 3.0), 3))

    """
    棱上的点总装
    """
    bar_d2 = np.append(bar_d2, np.array([bar_d21, bar_d22, bar_d23]))
    bar_d2.resize((int(len(bar_d2) / 3.0), 3))

    """
    三个面上的点-其他三个面（分别）
    """
    bar_d3 = np.array([])
    bar_d31_1 = np.array([])
    bar_d32_1 = np.array([])
    bar_d33_1 = np.array([])

    for i in range(0, len(d31)):
        bar_d31_1_element = d31[i] + a1
        bar_d32_1_element = d32[i] + a2
        bar_d33_1_element = d33[i] + a3
        bar_d31_1 = np.append(bar_d31_1, bar_d31_1_element)
        bar_d32_1 = np.append(bar_d32_1, bar_d32_1_element)
        bar_d33_1 = np.append(bar_d33_1, bar_d33_1_element)
    bar_d3 = np.append(bar_d3, np.array([bar_d31_1, bar_d32_1, bar_d33_1]))
    print(len(bar_d31_1))
    bar_d31_1.resize((int(len(bar_d31_1) / 3.0), 3))
    print(len(bar_d31_1))

    bar_d32_1.resize((int(len(bar_d32_1) / 3.0), 3))
    bar_d33_1.resize((int(len(bar_d33_1) / 3.0), 3))
    bar_d3.resize((int(len(bar_d3) / 3.0), 3))
    d = np.concatenate((d1, d2, d3, d4, bar_d1, bar_d2, bar_d3))
    """
    B矩阵的组装
    """
    I_B0 = np.eye(6, k=1)
    I0_B0 = np.zeros((6, 6))
    I_Ba = np.array([[1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0],
                   [0.0, 0.0, 1.0],
                   [0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0]])
    I0_Ba = np.array([[0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0]])
    B0 = np.zeros((len(d), len(d1) + len(d2) + len(d3) + len(d4), 6, 6))  # 4代表分块矩阵的列
    Ba = np.zeros((len(d), 3, 6, 3))  # 3代表分块矩阵的列
    c0 = len(d1) + 0

    c1 = len(d2) + c0

    c2 = len(d3) + c1

    c3 = len(d4) + c2

    c4 = len(bar_d1) + c3
    c5 = len(bar_d21_1) + c4
    c6 = len(bar_d21_2) + c5
    c7 = len(bar_d21_3) + c6
    c8 = len(bar_d22_1) + c7
    c9 = len(bar_d22_2) + c8
    c10 = len(bar_d22_3) + c9
    c11 = len(bar_d23_1) + c10
    c12 = len(bar_d23_2) + c11
    c13 = len(bar_d23_3) + c12
    c14 = len(bar_d31_1) + c13
    c15 = len(bar_d32_1) + c14
    c16 = len(bar_d33_1) + c15


    # print(len(d1),
    #     len(d2),
    #     len(d3),
    #     len(d4),
    #     len(bar_d1),
    #     len(bar_d21_1),
    #     len(bar_d21_2),
    #     len(bar_d21_3),
    #     len(bar_d22_1),
    #     len(bar_d22_2),
    #     len(bar_d22_3),
    #     len(bar_d23_1),
    #     len(bar_d23_2),
    #     len(bar_d23_3),
    #     len(bar_d31_1),
    #     len(bar_d32_1),
    #     len(bar_d33_1))

    for i in range(0, len(d)):
        if i < c0:  # 填写第一行

            B0[i, i] = I_B0


            Ba[i, 0] = I0_Ba
            Ba[i, 1] = I0_Ba
            Ba[i, 2] = I0_Ba
        elif c0 <= i < c1: # 填写第1行到第19行
            B0[i, i] = I_B0

            Ba[i, 0] = I0_Ba
            Ba[i, 1] = I0_Ba
            Ba[i, 2] = I0_Ba

        elif c1 <= i < c2:
            B0[i, i] = I_B0

            Ba[i, 0] = I0_Ba
            Ba[i, 1] = I0_Ba
            Ba[i, 2] = I0_Ba
        elif c2 <= i < c3:

            B0[i, i] = I_B0

            Ba[i, 0] = I0_Ba
            Ba[i, 1] = I0_Ba
            Ba[i, 2] = I0_Ba

        elif c3 <= i < c4:
            B0[i, 0] = I_B0
            # print(Ba[c3].shape, np.array([I0_Ba, I_Ba, I0_Ba]).shape)
            Ba[c3] = np.array([I0_Ba, I_Ba, I0_Ba])
            Ba[c3 + 1] = np.array([I_Ba, I_Ba, I0_Ba])
            Ba[c3 + 2] = np.array([I_Ba, I0_Ba, I0_Ba])
            Ba[c3 + 3] = np.array([I0_Ba, I0_Ba, I_Ba])
            Ba[c3 + 4] = np.array([I0_Ba, I_Ba, I_Ba])
            Ba[c3 + 5] = np.array([I_Ba, I_Ba, I_Ba])
            Ba[c3 + 6] = np.array([I_Ba, I0_Ba, I_Ba])

        elif c4 <= i < c13:
            if c4 <= i < c5:   #bar_d21_1里的点
                j = i - c4  # 局部坐标， 从0开始，一直到6
                B0[i, c0 + j] = I_B0
                Ba[i] = np.array([I0_Ba, I_Ba, I0_Ba])

            elif c5 <= i < c6: #bar_d21_2里的点
                j = i - c5  # 局部坐标， 从0开始，一直到6
                B0[i, c0 + j] = I_B0
                Ba[i] = np.array([I_Ba, I_Ba, I0_Ba])

            elif c6 <= i < c7: #bar_d21_3里的点
                j = i - c6  # 局部坐标， 从0开始，一直到6
                B0[i, c0 + j] = I_B0
                Ba[i] = np.array([I_Ba, I0_Ba, I0_Ba])

            elif c7 <= i < c8: #bar_d22_1里的点
                j = i - c7  # 局部坐标， 从0开始，一直到6
                B0[i, c0 + len(d21) + j] = I_B0
                Ba[i] = np.array([I_Ba, I0_Ba, I0_Ba])

            elif c8 <= i < c9: #bar_d22_2里的点
                j = i - c8  # 局部坐标， 从0开始，一直到6
                B0[i, c0 + len(d21) + j] = I_B0
                Ba[i] = np.array([I_Ba, I0_Ba, I_Ba])

            elif c9 <= i < c10: #bar_d22_3里的点
                j = i - c9  # 局部坐标， 从0开始，一直到6
                B0[i, c0 + len(d21) + j] = I_B0
                Ba[i] = np.array([I0_Ba, I0_Ba, I_Ba])

            elif c10 <= i < c11:  #bar_d22_3里的点
                j = i - c10  # 局部坐标， 从0开始，一直到6
                B0[i, c0 + len(d21) + len(d22) + j] = I_B0
                Ba[i] = np.array([I0_Ba, I_Ba, I0_Ba])

            elif c11 <= i < c12:  #bar_d22_3里的点
                j = i - c11  # 局部坐标， 从0开始，一直到6
                B0[i, c0 + len(d21) + len(d22) + j] = I_B0
                Ba[i] = np.array([I0_Ba, I_Ba, I_Ba])
            elif c12 <= i < c13:  #bar_d22_3里的点
                j = i - c12  # 局部坐标， 从0开始，一直到6
                B0[i, c0 + len(d21) + len(d22) + j] = I_B0
                Ba[i] = np.array([I0_Ba, I0_Ba, I_Ba])

        elif c13 <= i < c16:


            if c13 <= i < c14:
                j = i - c13  # 局部坐标， 从0开始，一直到6
                B0[i, c0 + len(d21) + len(d22) + len(d23) + j] = I_B0
                Ba[i] = np.array([I_Ba, I0_Ba, I0_Ba])
            elif c14 <= i < c15:
                j = i - c14  # 局部坐标， 从0开始，一直到6
                B0[i, c0 + len(d21) + len(d22) + len(d23) + len(d31) + j] = I_B0
                Ba[i] = np.array([I0_Ba, I_Ba, I0_Ba])
            elif c15 <= i < c16:
                j = i - c15  # 局部坐标， 从0开始，一直到6
                B0[i, c0 + len(d21) + len(d22) + len(d23) + + len(d31) + len(d32) + j] = I_B0
                Ba[i] = np.array([I0_Ba, I0_Ba, I_Ba])


    d1_index = np.array([])
    for i in range(0, len(d1)):
        for j in range(0, len(point_array)):
            if all(d1[i] == point_array[j]):
                index = j
                d1_index = np.append(d1_index, index)
            else:
                pass

    d2_index = np.array([])
    for i in range(0, len(d2)):
        for j in range(0, len(point_array)):
            if all(d2[i] == point_array[j]):
                index = j
                d2_index = np.append(d2_index, index)
            else:
                pass

    d3_index = np.array([])
    for i in range(0, len(d3)):
        for j in range(0, len(point_array)):
            if all(d3[i] == point_array[j]):
                index = j
                d3_index = np.append(d3_index, index)
            else:
                pass
    d4_index = np.array([])
    for i in range(0, len(d4)):
        for j in range(0, len(point_array)):
            if all(d4[i] == point_array[j]):
                index = j
                d4_index = np.append(d4_index, index)
            else:
                pass

    """
    输出新起始连接点和终止连接点，进行点集的二次排序，方便结构总体刚度矩阵的装配
    """
    start_point_array = connection_pair[:, 0]
    end_point_array = connection_pair[:, 1]
    index_start = np.array([])
    index_end = np.array([])
    for i in range(0, len(end_point_array)):
        for j in range(0, len(d)):
            if all(end_point_array[i] == d[j]):
                index = j
                index_end = np.append(index_end, index)
            else:
                pass

    for i in range(0, len(start_point_array)):
        for j in range(0, len(d)):
            if all(start_point_array[i] == d[j]):
                index = j
                index_start = np.append(index_start, index)
            else:
                pass
    # k_list = []
    # k = 0
    # for i in range(0, len(d)):
    #     for j in range(0, len(point_array)):
    #         if all(d[i] == point_array[j]):
    #             k_list.append(k)
    #             k += 1
    # print(len(k_list))

    return d1, d2, d3, d4, d, B0, Ba, index_start, index_end


def unfold_tensor(array):
    shape = array.shape
    Ke = np.zeros((shape[0] * shape[2], shape[1] * shape[3]))
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            for k in range(0, shape[2]):
                for l in range(0, shape[3]):
                     Ke[i * shape[2] + k, j * shape[3] + l] = array[i, j, k, l]
    return Ke



def Ke11(l, D):
    A = (np.pi * (D ** 2)) / 4.0  # 杆件的横截面积
    I_y = (np.pi * (D ** 4)) / 64.0  # 圆截面的惯性矩
    I_z = I_y  # 圆截面的惯性矩
    J = (np.pi * (D ** 4)) / 32.0  # 圆截面的极惯性矩
    k11 = [[(E * A / l), 0, 0, 0, 0, 0],
           [0, (12 * E * I_z / (l ** 3)), 0, 0, 0, (6 * E * I_z / (l ** 2))],
           [0, 0, (12 * E * I_y / (l ** 3)), 0, - (6 * E * I_y / (l ** 2)), 0],
           [0, 0, 0, ((G * J) / l), 0, 0],
           [0, 0, - (6 * E * I_y / (l ** 2)), 0, ((4 * E * I_y) / l), 0],
           [0, (6 * E * I_z / (l ** 2)), 0, 0, 0, ((4 * E * I_z) / l)]]
    return k11


def Ke12(l, D):
    A = (np.pi * (D ** 2)) / 4.0  # 杆件的横截面积
    I_y = (np.pi * (D ** 4)) / 64.0  # 圆截面的惯性矩
    I_z = I_y  # 圆截面的惯性矩
    J = (np.pi * (D ** 4)) / 32.0  # 圆截面的极惯性矩
    k12 = [[- (E * A / l), 0, 0, 0, 0, 0],
           [0, - (12 * E * I_z / (l ** 3)), 0, 0, 0, (6 * E * I_z / (l ** 2))],
           [0, 0, - (12 * E * I_y / (l ** 3)), 0, - (6 * E * I_y / (l ** 2)), 0],
           [0, 0, 0, -((G * J) / l), 0, 0],
           [0, 0, (6 * E * I_y / (l ** 2)), 0, (2 * E * I_y / l), 0],
           [0, - (6 * E * I_z / (l ** 2)), 0, 0, 0, (2 * E * I_z / l)]]
    return k12


def Ke21(l, D):
    A = (np.pi * (D ** 2)) / 4.0  # 杆件的横截面积
    I_y = (np.pi * (D ** 4)) / 64.0  # 圆截面的惯性矩
    I_z = I_y  # 圆截面的惯性矩
    J = (np.pi * (D ** 4)) / 32.0  # 圆截面的极惯性矩
    k21 = [[- (E * A / l), 0, 0, 0, 0, 0],
           [0, - (12 * E * I_z / (l ** 3)), 0, 0, 0, - (6 * E * I_z / (l ** 2))],
           [0, 0, - (12 * E * I_y / (l ** 3)), 0, (6 * E * I_y / (l ** 2)), 0],
           [0, 0, 0, - ((G * J) / l), 0, 0],
           [0, 0, - (6 * E * I_y / (l ** 2)), 0, (2 * E * I_y / l), 0],
           [0, (6 * E * I_z / (l ** 2)), 0, 0, 0, ((2 * E * I_z) / l)]]
    return k21


def Ke22(l, D):
    A = (np.pi * (D ** 2)) / 4.0  # 杆件的横截面积
    I_y = (np.pi * (D ** 4)) / 64.0  # 圆截面的惯性矩
    I_z = I_y  # 圆截面的惯性矩
    J = (np.pi * (D ** 4)) / 32.0  # 圆截面的极惯性矩
    k22 = [[(E * A / l), 0, 0, 0, 0, 0],
           [0, (12 * E * I_z / (l ** 3)), 0, 0, 0, - (6 * E * I_z / (l ** 2))],
           [0, 0, (12 * E * I_y / (l ** 3)), 0, (6 * E * I_y / (l ** 2)), 0],
           [0, 0, 0, ((G * J) / l), 0, 0],
           [0, 0, (6 * E * I_y / (l ** 2)), 0, ((4 * E * I_y) / l), 0],
           [0, - (6 * E * I_z / (l ** 2)), 0, 0, 0, ((4 * E * I_z) / l)]]
    return k22


def rotation_m(p1, p2):
    s = []
    a0 = p2[0]-p1[0]
    b0 = p2[1]-p1[1]
    c0 = p2[2]-p1[2]
    a = a0 / np.sqrt(a0 ** 2 + b0 ** 2 + c0 ** 2)
    b = b0 / np.sqrt(a0 ** 2 + b0 ** 2 + c0 ** 2)
    c = c0 / np.sqrt(a0 ** 2 + b0 ** 2 + c0 ** 2)
    v_i = np.array([a, b, c])
    if b == 0.0 and c == 0.0:
        v_j = np.array([0, 1.0, 0])
    else:
        v_j = np.array([0, c / np.sqrt(b ** 2 + c ** 2), -b / np.sqrt(b ** 2 + c ** 2)])
    v_k = np.cross(v_i, v_j)
    v_k = normal_v(v_k)
    s.append(v_i)
    s.append(v_j)
    s.append(v_k)
    lam = np.array(s)
    lam2 = np.block([[lam, np.zeros((3, 3))],
                     [np.zeros((3, 3)), lam]])
    return lam2


def dis(p1, p2):
    a0 = p2[0]-p1[0]
    b0 = p2[1]-p1[1]
    c0 = p2[2]-p1[2]
    d = np.sqrt(a0 ** 2 + b0 ** 2 + c0 ** 2)
    return d


def normal_v(v):
    m = 1 / (np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2))
    v1 = np.array([m * v[0],  m * v[1], m * v[2]])
    return v1


"""
不同相的位置
"""
s_phase_index = np.array([0, 1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 35, 42, 43, 44, 45, 46, 47, 48, 13, 20, 27, 34, 41,
          49, 55, 91, 97,
          98, 104, 140, 146,
          147, 153, 189, 195,
          196, 202, 238, 244,
          245, 251, 287, 293,
          294, 295, 296, 297, 298, 299, 300, 301, 308, 315, 322, 329, 336, 337, 338, 339, 340, 341, 342, 307, 314, 321, 328, 335])
b_phase_index = np.array([0, 6, 42, 48,
          294, 300, 336, 342,
          57, 61, 85, 89,
          253, 257, 281, 285,
          114, 116, 128, 130,
          212, 214, 226, 228,
          171])
o_phase_index = np.array([0, 6, 8, 12, 16, 18, 24, 30, 32, 36, 40, 42, 48,
          294, 300, 302, 306, 310, 312, 318, 324, 326, 330, 334, 336, 342,
          50, 54, 56, 62, 66, 72, 74, 80, 84, 90, 92, 96,
          246, 250, 252, 258, 262, 268, 270, 276, 280, 286, 288, 292,
          100, 102, 108, 112, 118, 120, 124, 126, 132, 136, 142, 144,
          198, 200, 206, 210, 216, 218, 222, 224, 230, 234, 240, 242,
          150, 156, 158, 162, 166, 168, 174, 176, 180, 184, 186, 192])

sb_phase_index_pre = np.array([0, 1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 35, 42, 43, 44, 45, 46, 47, 48, 13, 20, 27, 34, 41,
          49, 55, 91, 97,
          98, 104, 140, 146,
          147, 153, 189, 195,
          196, 202, 238, 244,
          245, 251, 287, 293,
          294, 295, 296, 297, 298, 299, 300, 301, 308, 315, 322, 329, 336, 337, 338, 339, 340, 341, 342, 307, 314, 321, 328, 335,
          0, 6, 42, 48,
          294, 300, 336, 342,
          57, 61, 85, 89,
          253, 257, 281, 285,
          114, 116, 128, 130,
          212, 214, 226, 228,
          171])

sb_phase_index = np.unique(sb_phase_index_pre)

so_phase_index_pre = np.array([0, 1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 35, 42, 43, 44, 45, 46, 47, 48, 13, 20, 27, 34, 41,
          49, 55, 91, 97,
          98, 104, 140, 146,
          147, 153, 189, 195,
          196, 202, 238, 244,
          245, 251, 287, 293,
          294, 295, 296, 297, 298, 299, 300, 301, 308, 315, 322, 329, 336, 337, 338, 339, 340, 341, 342, 307, 314, 321, 328, 335,
          0, 6, 8, 12, 16, 18, 24, 30, 32, 36, 40, 42, 48,
          294, 300, 302, 306, 310, 312, 318, 324, 326, 330, 334, 336, 342,
          50, 54, 56, 62, 66, 72, 74, 80, 84, 90, 92, 96,
          246, 250, 252, 258, 262, 268, 270, 276, 280, 286, 288, 292,
          100, 102, 108, 112, 118, 120, 124, 126, 132, 136, 142, 144,
          198, 200, 206, 210, 216, 218, 222, 224, 230, 234, 240, 242,
          150, 156, 158, 162, 166, 168, 174, 176, 180, 184, 186, 192])

so_phase_index = np.unique(so_phase_index_pre)

bo_phase_index_pre = np.array([0, 6, 42, 48,
          294, 300, 336, 342,
          57, 61, 85, 89,
          253, 257, 281, 285,
          114, 116, 128, 130,
          212, 214, 226, 228,
          171,
          0, 6, 8, 12, 16, 18, 24, 30, 32, 36, 40, 42, 48,
          294, 300, 302, 306, 310, 312, 318, 324, 326, 330, 334, 336, 342,
          50, 54, 56, 62, 66, 72, 74, 80, 84, 90, 92, 96,
          246, 250, 252, 258, 262, 268, 270, 276, 280, 286, 288, 292,
          100, 102, 108, 112, 118, 120, 124, 126, 132, 136, 142, 144,
          198, 200, 206, 210, 216, 218, 222, 224, 230, 234, 240, 242,
          150, 156, 158, 162, 166, 168, 174, 176, 180, 184, 186, 192])

bo_phase_index = np.unique(bo_phase_index_pre)

sbo_phase_index_pre = np.array([0, 1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 35, 42, 43, 44, 45, 46, 47, 48, 13, 20, 27, 34, 41,
          49, 55, 91, 97,
          98, 104, 140, 146,
          147, 153, 189, 195,
          196, 202, 238, 244,
          245, 251, 287, 293,
          294, 295, 296, 297, 298, 299, 300, 301, 308, 315, 322, 329, 336, 337, 338, 339, 340, 341, 342, 307, 314, 321, 328, 335,
          0, 6, 42, 48,
          294, 300, 336, 342,
          57, 61, 85, 89,
          253, 257, 281, 285,
          114, 116, 128, 130,
          212, 214, 226, 228,
          171,
          0, 6, 8, 12, 16, 18, 24, 30, 32, 36, 40, 42, 48,
          294, 300, 302, 306, 310, 312, 318, 324, 326, 330, 334, 336, 342,
          50, 54, 56, 62, 66, 72, 74, 80, 84, 90, 92, 96,
          246, 250, 252, 258, 262, 268, 270, 276, 280, 286, 288, 292,
          100, 102, 108, 112, 118, 120, 124, 126, 132, 136, 142, 144,
          198, 200, 206, 210, 216, 218, 222, 224, 230, 234, 240, 242,
          150, 156, 158, 162, 166, 168, 174, 176, 180, 184, 186, 192])

sbo_phase_index = np.unique(sbo_phase_index_pre)

bos_phase_index_pre = np.array([0, 6, 42, 48,
          294, 300, 336, 342,
          57, 61, 85, 89,
          253, 257, 281, 285,
          114, 116, 128, 130,
          212, 214, 226, 228,
          171,
          0, 6, 8, 12, 16, 18, 24, 30, 32, 36, 40, 42, 48,
          294, 300, 302, 306, 310, 312, 318, 324, 326, 330, 334, 336, 342,
          50, 54, 56, 62, 66, 72, 74, 80, 84, 90, 92, 96,
          246, 250, 252, 258, 262, 268, 270, 276, 280, 286, 288, 292,
          100, 102, 108, 112, 118, 120, 124, 126, 132, 136, 142, 144,
          198, 200, 206, 210, 216, 218, 222, 224, 230, 234, 240, 242,
          150, 156, 158, 162, 166, 168, 174, 176, 180, 184, 186, 192,
          122, 73,  220, 269, 172, 173, 169, 170, 157, 164, 178, 185])

bos_phase_index = np.unique(bos_phase_index_pre)

iso_phase_index_pre = np.array([0, 1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 35, 42, 43, 44, 45, 46, 47, 48, 13, 20, 27, 34, 41,
          49, 55, 91, 97,
          98, 104, 140, 146,
          147, 153, 189, 195,
          196, 202, 238, 244,
          245, 251, 287, 293,
          294, 295, 296, 297, 298, 299, 300, 301, 308, 315, 322, 329, 336, 337, 338, 339, 340, 341, 342, 307, 314, 321, 328, 335,
          0, 6, 42, 48,
          294, 300, 336, 342,
          57, 61, 85, 89,
          253, 257, 281, 285,
          114, 116, 128, 130,
          212, 214, 226, 228,
          171,122, 73, 220, 269, 172, 173, 169, 170, 157, 164, 178, 185])
iso_phase_index = np.unique(iso_phase_index_pre)

all_phase_index_pre = np.array([0, 1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 35, 42, 43, 44, 45, 46, 47, 48, 13, 20, 27, 34, 41,
          49, 55, 91, 97,
          98, 104, 140, 146,
          147, 153, 189, 195,
          196, 202, 238, 244,
          245, 251, 287, 293,
          294, 295, 296, 297, 298, 299, 300, 301, 308, 315, 322, 329, 336, 337, 338, 339, 340, 341, 342, 307, 314, 321, 328, 335,
          0, 6, 42, 48,
          294, 300, 336, 342,
          57, 61, 85, 89,
          253, 257, 281, 285,
          114, 116, 128, 130,
          212, 214, 226, 228,
          171,122, 73, 220, 269, 172, 173, 169, 170, 157, 164, 178, 185,
          0, 6, 8, 12, 16, 18, 24, 30, 32, 36, 40, 42, 48,
          294, 300, 302, 306, 310, 312, 318, 324, 326, 330, 334, 336, 342,
          50, 54, 56, 62, 66, 72, 74, 80, 84, 90, 92, 96,
          246, 250, 252, 258, 262, 268, 270, 276, 280, 286, 288, 292,
          100, 102, 108, 112, 118, 120, 124, 126, 132, 136, 142, 144,
          198, 200, 206, 210, 216, 218, 222, 224, 230, 234, 240, 242,
          150, 156, 158, 162, 166, 168, 174, 176, 180, 184, 186, 192
                                ])
all_phase_index = np.unique(all_phase_index_pre)

bos_1_phase_index_pre = np.array([0, 6, 42, 48,
          294, 300, 336, 342,
          57, 61, 85, 89,
          253, 257, 281, 285,
          114, 116, 128, 130,
          212, 214, 226, 228,
          171,
          0, 6, 8, 12, 16, 18, 24, 30, 32, 36, 40, 42, 48,
          294, 300, 302, 306, 310, 312, 318, 324, 326, 330, 334, 336, 342,
          50, 54, 56, 62, 66, 72, 74, 80, 84, 90, 92, 96,
          246, 250, 252, 258, 262, 268, 270, 276, 280, 286, 288, 292,
          100, 102, 108, 112, 118, 120, 124, 126, 132, 136, 142, 144,
          198, 200, 206, 210, 216, 218, 222, 224, 230, 234, 240, 242,
          150, 156, 158, 162, 166, 168, 174, 176, 180, 184, 186, 192,
          122, 73, 220, 269, 172, 173, 169, 170, 157, 164, 178, 185,
          114, 116, 128, 130,
          212, 214, 226, 228])

bos_1_phase_index = np.unique(bos_1_phase_index_pre)

rad = 1.0
E = 70e3
mu = 0.3
G = E / (2 * (1 + mu))
#
p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = dual_phase_lattice_oct_bcc(o_phase_index, oct_connection, sc_connection)
#
dd1, dd2, dd3, dd4, dd, b0, ba, start_point_index, end_point_index = point_classification(p5, 70.0, p1)
b0_2, ba_2 = unfold_tensor(b0), unfold_tensor(ba)

"""
总体刚度矩阵的装配
"""

kuc = np.zeros((len(dd), len(dd), 6, 6))
for e in range(0, len(start_point_index)):
    point1 = dd[int(start_point_index[e])]
    point2 = dd[int(end_point_index[e])]
    cT = rotation_m(point1, point2)
    length = dis(point1, point2)
    k11 = np.dot(np.dot(cT.T, Ke11(length, rad)), cT)
    k12 = np.dot(np.dot(cT.T, Ke12(length, rad)), cT)
    k21 = np.dot(np.dot(cT.T, Ke21(length, rad)), cT)
    k22 = np.dot(np.dot(cT.T, Ke22(length, rad)), cT)
    kuc[int(start_point_index[e])][int(start_point_index[e])] += k11
    kuc[int(start_point_index[e])][int(end_point_index[e])] += k12
    kuc[int(end_point_index[e])][int(start_point_index[e])] += k21
    kuc[int(end_point_index[e])][int(end_point_index[e])] += k22
k_uc = unfold_tensor(kuc)

a = 70
Beps = np.array([[a, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.5 * a, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.5 * a],
                 [0.0, 0.0, 0.0, 0.5 * a, 0.0, 0.0],
                 [0.0, a, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 0.5 * a, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.5 * a],
                 [0.0, 0.0, 0.0, 0.0, 0.5 * a, 0.0],
                 [0.0, 0.0, a, 0.0, 0.0, 0.0]])
M0 = -1.0 * (np.linalg.pinv(np.dot(np.dot(b0_2.T, k_uc), b0_2)))
M1 = np.dot(np.dot(M0, b0_2.T), k_uc)
D0 = np.dot(M1, ba_2)
Da = np.dot(b0_2, D0) + ba_2
Ka = np.dot(np.dot(Da.T, k_uc), Da)
K = np.dot(np.dot(Beps.T, Ka), Beps) / (a ** 3)
print(np.linalg.matrix_rank(b0_2), np.linalg.matrix_rank(ba_2))
# print(K)
# print(dd.ndim)
# print(dd[19: 127])
# print(start_point_index)
# print(end_point_index)
# print(p1[:,0])
# print(p1[:,1])
# print(len(start_point_index))
# print(len(p10))
# print(len(p1))
# print(dd4)
#
#
# test2 = np.unique(dd, axis=0)
# print(test2, len(test2))
#
# print(np.linalg.matrix_rank(k_uc), k_uc.shape)
#
#
#
"""
空间刚度图
"""
def yansuan(phi, tht):
    sig_dir = np.array([np.cos(phi) * np.sin(tht), np.sin(phi), np.cos(phi) * np.cos(tht)])
    sig = stress_rotation(sig_dir)
    eps_v = np.linalg.solve(K, sig)
    eps = strain_rotation(eps_v, sig_dir)
    Ee = 1.0 / eps[0]
    return Ee


print(yansuan(0, 0))
print(yansuan(0, 3.1415926/2))
print(yansuan(3.1415926/4, math.atan(math.sqrt(2))))
fig = plt.figure()
ax = fig.gca(projection='3d')

tht1 = np.linspace(0.01, 2 * 3.14159, 50)
phi1 = np.linspace(0.01, 2 * 3.14159, 50)
tht, phi = np.meshgrid(tht1, phi1)
rol = []
for ppp in range(0, len(tht1)):
    rol.append([])
    for ppp2 in range(0, len(phi1)):
        rol[ppp].append(yansuan(tht1[ppp], phi1[ppp2]))
#print(x, y)
X = rol * np.cos(phi) * np.sin(tht)
Y = rol * np.sin(phi)
norm = Normalize()
colors = norm(rol)
cmap = plt.cm.get_cmap('coolwarm')
my_font = {'family': 'Times New Roman', 'style': 'italic', 'size': 10}


Z = rol * np.cos(phi) * np.cos(tht)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=cmap(colors))
ax.set_xlabel('x', fontdict=my_font)
ax.set_ylabel('y', fontdict=my_font)
ax.set_zlabel('z', fontdict=my_font)
ax.elev = -90
ax.azim = -90
plt.show()

fig = plt.figure()
ax2 = fig.gca(fc='whitesmoke', projection='3d')

for j in range(0, len(p1)):
    ax2.plot([p1[j, 0, 0], p1[j, 1, 0]], [p1[j, 0, 1], p1[j, 1, 1]], [p1[j, 0, 2], p1[j, 1, 2]],
              ls='-',
              c='k')   # size)


plt.show()







