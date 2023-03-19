from model.DBN import DBN_RES
from model.LSTM import LSTM_RES
import math

def get_echart1_data():
    plt_data, legend_line, legend_pie, perc_pie, title_pie, time_sum_DBN, _, times, bar_legend, bar_data = DBN_RES()
    xAxis_line = [x for x in range(200)]
    series_line = list()
    for data_list in plt_data:
        series_line.append({'name': data_list['name'],'type': 'line', 'data': data_list['data'], 'smooth': 'true'})
    data_pie = list()
    for leg, per in list(zip(legend_pie, perc_pie)):
        data_pie.append({'value': per, 'name': leg})
    time_dbn = [{'name':'模型预测时间-DBN', 'type': 'line', 'data': times}]
    bar_dbn = [{'name': '预测精度-DBN', 'type': 'bar', 'data': bar_data}]
    return xAxis_line, series_line, legend_line, data_pie, title_pie, time_sum_DBN, time_dbn, bar_legend, bar_dbn

def get_echart2_data():
    plot_print, legend, time_sum, time_steps, times, bar_data = LSTM_RES()
    print(bar_data)
    series_line = list()
    for data_list in plot_print:
        series_line.append({'name': data_list['name'],'type': 'line', 'data': data_list['data'], 'smooth': 'true'})
    time_lstm = [{'name':'模型预测时间-LSTM', 'type': 'line', 'data': times}]
    bar_lstm = [{'name': '预测精度-LSTM', 'type': 'bar', 'data': [0 if math.isnan(x) else x for x in bar_data]}]
    print(bar_lstm)
    return series_line, legend, time_sum, time_steps, time_lstm, bar_lstm

if __name__ == '__main__':
    get_echart1_data()