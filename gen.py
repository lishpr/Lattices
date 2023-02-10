import choose
import main
import numpy as np
import math


rad = 1.0
E = 70e3
mu = 0.3
G = E / (2 * (1 + mu))
#
for i in range(10):
    phase_index = np.unique(np.array(choose.select_lattice(7,128)))
    print(phase_index)
    p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = main.dual_phase_lattice_oct_bcc(phase_index, main.oct_connection, main.sc_connection)
    #
    dd1, dd2, dd3, dd4, dd, b0, ba, start_point_index, end_point_index = main.point_classification(p5, 70.0, p1)
    b0_2, ba_2 = main.unfold_tensor(b0), main.unfold_tensor(ba)

    """
    总体刚度矩阵的装配
    """

    kuc = np.zeros((len(dd), len(dd), 6, 6))
    for e in range(0, len(start_point_index)):
        point1 = dd[int(start_point_index[e])]
        point2 = dd[int(end_point_index[e])]
        cT = main.rotation_m(point1, point2)
        length = main.dis(point1, point2)
        k11 = np.dot(np.dot(cT.T, main.Ke11(length, rad)), cT)
        k12 = np.dot(np.dot(cT.T, main.Ke12(length, rad)), cT)
        k21 = np.dot(np.dot(cT.T, main.Ke21(length, rad)), cT)
        k22 = np.dot(np.dot(cT.T, main.Ke22(length, rad)), cT)
        kuc[int(start_point_index[e])][int(start_point_index[e])] += k11
        kuc[int(start_point_index[e])][int(end_point_index[e])] += k12
        kuc[int(end_point_index[e])][int(start_point_index[e])] += k21
        kuc[int(end_point_index[e])][int(end_point_index[e])] += k22
    k_uc = main.unfold_tensor(kuc)

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
    # print(np.linalg.matrix_rank(b0_2), np.linalg.matrix_rank(ba_2))


    print(main.yansuan(0, 0), main.yansuan(0, 3.1415926/2), main.yansuan(3.1415926/4, math.atan(math.sqrt(2))))
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
