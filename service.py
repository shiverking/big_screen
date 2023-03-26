from model.DBN import DBN_RES
from model.LSTM import LSTM_RES
import math

def get_dbnModel_res(scenario, testName):
    upper_left_corner, data_pie, time_sum, x, y, upper_right_corner, lower_right_corner = DBN_RES(scenario, testName)
    time_line_dbn = [{'name':'模型预测时间-DBN', 'type': 'line', 'data': y}]
    return upper_left_corner, data_pie, time_sum, x, time_line_dbn, upper_right_corner, lower_right_corner

def get_lstmModel_res(scenario, testName):
    bottom_left_corner, time_sum, x, y, upper_right_corner, lower_right_corner = LSTM_RES(scenario, testName)
    time_line_lstm = [{'name':'模型预测时间-LSTM', 'type': 'line', 'data': y}]
    return bottom_left_corner, time_sum, x, time_line_lstm, upper_right_corner, lower_right_corner