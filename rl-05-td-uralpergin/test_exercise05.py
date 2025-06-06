
import numpy as np
import pytest
from cliff_walking import CliffWalkingEnv
from q_learning import q_learning



def assert_float_dict_almost_equal(a, b, decimal=6):
    for key_pair in zip(sorted(a), sorted(b)):
        assert key_pair[0] == key_pair[1]
        np.testing.assert_array_almost_equal(a[key_pair[0]], b[key_pair[1]], decimal=decimal)


def test_q_values():
    env = pytest.global_env
    np.random.seed(0)
    Q, stats = q_learning(env, 500)

    expected_q_values = {36: [-13., -99.99847412, -13.99793134, -13.99989125],
                             24: [-13.49710026, -12., -13.99820799, -12.99943967],
                             12: [-12.82379343, -12.78845206, -12.6809052, -12.68432215],
                             0: [-12.25, -12.35247495, -12.39078253, -12.24448329],
                             1: [-11.71176481, -11.74629913, -11.93450181, -11.8367309],
                             2: [-11.17147528, -10.92768896, -10.98834139, -11.82520464],
                             14: [-11.11142202, -10.96755917, -10.97793159, -11.41046726],
                             3: [-10.25, -10.08161693, -10.12084676, -11.12942917],
                             4: [-9.32727051, -9.2432254, -9.74520199, -9.26034445],
                             5: [-8.5, -8.35266845, -8.70257811, -8.77540643],
                             6: [-7.5, -7.50886764, -7.60559975, -7.5479126],
                             7: [-6.77734375, -6.64107635, -6.78943835, -6.79408813],
                             8: [-6.30754702, -5.78547857, -5.74104174, -6.24649891],
                             19: [-6.3800348, -5.99989625, -5.99990623, -6.63451042],
                             18: [-7.21072567, -6.99974138, -6.99986636, -8.55963877],
                             17: [-8.96059073, -7.9994099, -7.99944799, -9.67197862],
                             16: [-9.62802058, -8.99848076, -8.9990922, -9.75750722],
                             15: [-10.16274317, -9.99715028, -9.99795646, -10.80202542],
                             13: [-12.42137411, -11.9335771, -11.95402925, -12.81399528],
                             20: [-5.96095161, -4.99996096, -4.99998729, -6.23703834],
                             9: [-5., -4.92796804, -4.90053431, -5.54233545],
                             10: [-4.25, -3.96801244, -3.98320146, -4.98549087],
                             11: [-3.44148111, -3., -2.99619046, -3.76515198],
                             23: [-3.06648111, -2.87414416, -2., -2.3046875],
                             22: [-4.47421227, -2.99999853, -2.9999976, -3.03710938],
                             21: [-5.41569181, -3.9999867, -3.99998555, -5.71363213],
                             32: [-5.99980367, -4., -99.21875, -5.99784381],
                             31: [-6.99183671, -5., -99.8046875, -6.99960543],
                             33: [-4.98881985, -3., -99.8046875, -4.97478665],
                             35: [-2.9981689, -1.99804688, -1., -2.99197388],
                             34: [-3.98397119, -2., -99.8046875, -3.98498535],
                             47: [0., 0., 0., 0.],
                             37: [0., 0., 0., 0.],
                             25: [-12.90564936, -11., -99.99389648, -12.97025546],
                             26: [-11.86995709, -10., -99.99694824, -11.94398283],
                             38: [0., 0., 0., 0.],
                             28: [-9.89046974, -8., -99.99990463, -9.99936445],
                             27: [-10.99496568, -9., -99.99847412, -10.99690954],
                             29: [-8.98680985, -7., -99.609375, -8.96609374],
                             30: [-7.99175931, -6., -99.97558594, -7.96041614],
                             43: [0., 0., 0., 0.],
                             40: [0., 0., 0., 0.],
                             39: [0., 0., 0., 0.],
                             41: [0., 0., 0., 0.],
                             42: [0., 0., 0., 0.],
                             45: [0., 0., 0., 0.],
                             44: [0., 0., 0., 0.],
                             46: [0., 0., 0., 0.]}


    assert_float_dict_almost_equal(Q, expected_q_values, decimal=2)


def test_rewards():
    env = pytest.global_env
    np.random.seed(0)
    Q, stats = q_learning(env, 500)
    expected_reward = [-112.0,
                           -100.0,
                           -109.0,
                           -109.0,
                           -164.0,
                           -113.0,
                           -122.0,
                           -123.0,
                           -114.0,
                           -74.0,
                           -111.0,
                           -135.0,
                           -112.0,
                           -148.0,
                           -130.0,
                           -134.0,
                           -180.0,
                           -159.0,
                           -123.0,
                           -66.0,
                           -75.0,
                           -106.0,
                           -41.0,
                           -103.0,
                           -127.0,
                           -114.0,
                           -61.0,
                           -23.0,
                           -74.0,
                           -68.0,
                           -38.0,
                           -40.0,
                           -102.0,
                           -24.0,
                           -129.0,
                           -62.0,
                           -54.0,
                           -26.0,
                           -119.0,
                           -29.0,
                           -34.0,
                           -20.0,
                           -61.0,
                           -23.0,
                           -128.0,
                           -52.0,
                           -17.0,
                           -40.0,
                           -110.0,
                           -19.0,
                           -36.0,
                           -32.0,
                           -38.0,
                           -17.0,
                           -21.0,
                           -33.0,
                           -120.0,
                           -113.0,
                           -119.0,
                           -111.0,
                           -32.0,
                           -34.0,
                           -29.0,
                           -19.0,
                           -25.0,
                           -17.0,
                           -25.0,
                           -15.0,
                           -16.0,
                           -126.0,
                           -25.0,
                           -19.0,
                           -16.0,
                           -17.0,
                           -102.0,
                           -21.0,
                           -15.0,
                           -29.0,
                           -19.0,
                           -13.0,
                           -20.0,
                           -112.0,
                           -102.0,
                           -104.0,
                           -29.0,
                           -100.0,
                           -13.0,
                           -13.0,
                           -17.0,
                           -13.0,
                           -13.0,
                           -13.0,
                           -25.0,
                           -109.0,
                           -17.0,
                           -105.0,
                           -13.0,
                           -13.0,
                           -108.0,
                           -100.0,
                           -104.0,
                           -111.0,
                           -15.0,
                           -101.0,
                           -123.0,
                           -104.0,
                           -13.0,
                           -13.0,
                           -102.0,
                           -15.0,
                           -100.0,
                           -13.0,
                           -108.0,
                           -102.0,
                           -103.0,
                           -13.0,
                           -13.0,
                           -13.0,
                           -14.0,
                           -13.0,
                           -17.0,
                           -14.0,
                           -13.0,
                           -103.0,
                           -13.0,
                           -107.0,
                           -13.0,
                           -13.0,
                           -110.0,
                           -15.0,
                           -13.0,
                           -100.0,
                           -110.0,
                           -21.0,
                           -104.0,
                           -106.0,
                           -15.0,
                           -18.0,
                           -109.0,
                           -18.0,
                           -109.0,
                           -104.0,
                           -16.0,
                           -105.0,
                           -13.0,
                           -13.0,
                           -17.0,
                           -21.0,
                           -13.0,
                           -17.0,
                           -13.0,
                           -21.0,
                           -106.0,
                           -17.0,
                           -13.0,
                           -18.0,
                           -13.0,
                           -105.0,
                           -14.0,
                           -113.0,
                           -103.0,
                           -13.0,
                           -15.0,
                           -13.0,
                           -15.0,
                           -13.0,
                           -22.0,
                           -14.0,
                           -13.0,
                           -13.0,
                           -17.0,
                           -109.0,
                           -15.0,
                           -111.0,
                           -13.0,
                           -16.0,
                           -13.0,
                           -100.0,
                           -13.0,
                           -13.0,
                           -13.0,
                           -13.0,
                           -13.0,
                           -13.0,
                           -13.0,
                           -15.0,
                           -15.0,
                           -13.0,
                           -107.0,
                           -103.0,
                           -13.0,
                           -13.0,
                           -15.0,
                           -15.0,
                           -17.0,
                           -110.0,
                           -13.0,
                           -17.0,
                           -13.0,
                           -13.0,
                           -108.0,
                           -13.0,
                           -15.0,
                           -13.0,
                           -110.0,
                           -13.0,
                           -13.0,
                           -108.0,
                           -111.0,
                           -13.0,
                           -103.0,
                           -13.0,
                           -100.0,
                           -13.0,
                           -13.0,
                           -13.0,
                           -104.0,
                           -19.0,
                           -13.0,
                           -13.0,
                           -102.0,
                           -13.0,
                           -15.0,
                           -15.0,
                           -18.0,
                           -13.0,
                           -13.0,
                           -113.0,
                           -107.0,
                           -13.0,
                           -13.0,
                           -13.0,
                           -13.0,
                           -15.0,
                           -13.0,
                           -13.0,
                           -15.0,
                           -105.0,
                           -13.0,
                           -13.0,
                           -111.0,
                           -15.0,
                           -13.0,
                           -105.0,
                           -15.0,
                           -105.0,
                           -13.0,
                           -13.0,
                           -13.0,
                           -13.0,
                           -105.0,
                           -14.0,
                           -13.0,
                           -14.0,
                           -15.0,
                           -19.0,
                           -105.0,
                           -102.0,
                           -13.0,
                           -15.0,
                           -15.0,
                           -108.0,
                           -13.0,
                           -15.0,
                           -16.0,
                           -13.0,
                           -103.0,
                           -110.0,
                           -13.0,
                           -25.0,
                           -18.0,
                           -13.0,
                           -15.0,
                           -13.0,
                           -13.0,
                           -13.0,
                           -13.0,
                           -13.0,
                           -21.0,
                           -15.0,
                           -13.0,
                           -15.0,
                           -102.0,
                           -17.0,
                           -15.0,
                           -111.0,
                           -15.0,
                           -15.0,
                           -15.0,
                           -14.0,
                           -113.0,
                           -104.0,
                           -103.0,
                           -106.0,
                           -14.0,
                           -13.0,
                           -107.0,
                           -109.0,
                           -13.0,
                           -15.0,
                           -13.0,
                           -13.0,
                           -17.0,
                           -19.0,
                           -100.0,
                           -111.0,
                           -15.0,
                           -13.0,
                           -13.0,
                           -15.0,
                           -13.0,
                           -13.0,
                           -16.0,
                           -14.0,
                           -14.0,
                           -13.0,
                           -107.0,
                           -104.0,
                           -15.0,
                           -13.0,
                           -107.0,
                           -17.0,
                           -13.0,
                           -103.0,
                           -100.0,
                           -13.0,
                           -15.0,
                           -13.0,
                           -100.0,
                           -13.0,
                           -19.0,
                           -15.0,
                           -16.0,
                           -17.0,
                           -107.0,
                           -13.0,
                           -15.0,
                           -13.0,
                           -13.0,
                           -15.0,
                           -13.0,
                           -15.0,
                           -16.0,
                           -13.0,
                           -106.0,
                           -17.0,
                           -13.0,
                           -13.0,
                           -13.0,
                           -104.0,
                           -13.0,
                           -13.0,
                           -13.0,
                           -103.0,
                           -15.0,
                           -13.0,
                           -13.0,
                           -23.0,
                           -17.0,
                           -103.0,
                           -105.0,
                           -14.0,
                           -13.0,
                           -13.0,
                           -15.0,
                           -13.0,
                           -13.0,
                           -17.0,
                           -17.0,
                           -17.0,
                           -15.0,
                           -13.0,
                           -109.0,
                           -106.0,
                           -13.0,
                           -13.0,
                           -15.0,
                           -13.0,
                           -107.0,
                           -13.0,
                           -13.0,
                           -15.0,
                           -15.0,
                           -13.0,
                           -15.0,
                           -13.0,
                           -15.0,
                           -105.0,
                           -105.0,
                           -14.0,
                           -17.0,
                           -13.0,
                           -13.0,
                           -106.0,
                           -13.0,
                           -14.0,
                           -15.0,
                           -13.0,
                           -15.0,
                           -16.0,
                           -19.0,
                           -20.0,
                           -15.0,
                           -13.0,
                           -103.0,
                           -13.0,
                           -13.0,
                           -16.0,
                           -13.0,
                           -13.0,
                           -15.0,
                           -106.0,
                           -13.0,
                           -15.0,
                           -13.0,
                           -17.0,
                           -16.0,
                           -15.0,
                           -15.0,
                           -19.0,
                           -106.0,
                           -15.0,
                           -13.0,
                           -13.0,
                           -13.0,
                           -13.0,
                           -111.0,
                           -14.0,
                           -17.0,
                           -14.0,
                           -13.0,
                           -15.0,
                           -13.0,
                           -104.0,
                           -13.0,
                           -13.0,
                           -15.0,
                           -13.0,
                           -100.0,
                           -15.0,
                           -17.0,
                           -107.0,
                           -104.0,
                           -13.0,
                           -107.0,
                           -13.0,
                           -13.0,
                           -13.0,
                           -15.0,
                           -15.0,
                           -15.0,
                           -100.0,
                           -17.0,
                           -102.0,
                           -13.0,
                           -13.0,
                           -105.0,
                           -100.0,
                           -111.0,
                           -15.0,
                           -17.0,
                           -15.0,
                           -13.0,
                           -108.0,
                           -17.0,
                           -13.0,
                           -19.0,
                           -13.0,
                           -17.0,
                           -19.0,
                           -17.0,
                           -13.0,
                           -15.0,
                           -105.0,
                           -13.0,
                           -13.0,
                           -15.0,
                           -13.0,
                           -100.0,
                           -16.0,
                           -15.0,
                           -13.0,
                           -13.0,
                           -13.0,
                           -15.0,
                           -15.0,
                           -26.0,
                           -15.0,
                           -17.0,
                           -13.0,
                           -13.0,
                           -15.0,
                           -13.0,
                           -13.0,
                           -114.0,
                           -23.0,
                           -17.0,
                           -17.0,
                           -17.0,
                           -15.0]

    np.testing.assert_array_almost_equal(stats[1], expected_reward, decimal=2)


