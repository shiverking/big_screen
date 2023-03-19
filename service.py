from model.DBN import DBN_RES
from model.LSTM import LSTM_RES

def get_echart1_data():
    plt_data, legend_line, legend_pie, perc_pie, title_pie, time_sum_DBN, _, times = DBN_RES()
    xAxis_line = [x for x in range(200)]
    series_line = list()
    for data_list in plt_data:
        series_line.append({'name': data_list['name'],'type': 'line', 'data': data_list['data'], 'smooth': 'true'})
    data_pie = list()
    for leg, per in list(zip(legend_pie, perc_pie)):
        data_pie.append({'value': per, 'name': leg})
    time_dbn = [{'name':'模型预测时间-DBN', 'type': 'line', 'data': times}]
    return xAxis_line, series_line, legend_line, data_pie, title_pie, time_sum_DBN, time_dbn

def get_echart2_data():
    plot_print, legend, time_sum, time_steps, times = LSTM_RES()
    series_line = list()
    for data_list in plot_print:
        series_line.append({'name': data_list['name'],'type': 'line', 'data': data_list['data'], 'smooth': 'true'})
    time_lstm = [{'name':'模型预测时间-LSTM', 'type': 'line', 'data': times}]
    return series_line, legend, time_sum, time_steps, time_lstm



if __name__ == '__main__':
    get_echart1_data()