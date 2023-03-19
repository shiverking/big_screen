from model.DBN import DBN_RES
from model.LSTM import LSTM_RES
import math

def get_echart1_data():
    plt_data, legend_line, legend_pie, perc_pie, title_pie, time_sum_DBN, _, times, bar_legend, bar_data, dbn_table = DBN_RES()
    xAxis_line = [x for x in range(200)]
    series_line = list()
    for data_list in plt_data:
        series_line.append({'name': data_list['name'],'type': 'line', 'data': data_list['data'], 'smooth': 'true'})
    data_pie = list()
    for leg, per in list(zip(legend_pie, perc_pie)):
        data_pie.append({'value': per, 'name': leg})
    time_dbn = [{'name':'模型预测时间-DBN', 'type': 'line', 'data': times}]
    bar_dbn = [{'name': '预测精度-DBN', 'type': 'bar', 'data': bar_data}]
    field_list = ['zero', 'one', 'five', 'ten', 'fifteen', 'twenty']
    cols = [{"field":'zero', "title": ''},
            {"field":'one', "title": 'KF-DBN(t=1)'},
            {"field":'five', "title": 'KF-DBN(t=5)'},
            {"field":'ten', "title": 'KF-DBN(t=10)'},
            {"field":'fifteen', "title": 'KF-DBN(t=15)'},
            {"field":'twenty', "title": 'KF-DBN(t=20)'},]
    data = []
    for i in range(len(dbn_table['data'])):
        col_dict = dict()
        col_dict[field_list[0]] = dbn_table['rowLabels'][i]
        for j in range(1, len(field_list)):
            col_dict[field_list[j]] = dbn_table['data'][i][j-1]
        data.append(col_dict)
    dbn_table = {"cols":cols,"data":data}
    return xAxis_line, series_line, legend_line, data_pie, title_pie, time_sum_DBN, time_dbn, bar_legend, bar_dbn, dbn_table

def get_echart2_data():
    plot_print, legend, time_sum, time_steps, times, bar_data, lstm_table = LSTM_RES()
    series_line = list()
    for data_list in plot_print:
        series_line.append({'name': data_list['name'],'type': 'line', 'data': data_list['data'], 'smooth': 'true'})
    time_lstm = [{'name':'模型预测时间-LSTM', 'type': 'line', 'data': times}]
    bar_lstm = [{'name': '预测精度-LSTM', 'type': 'bar', 'data': [0 if math.isnan(x) else x for x in bar_data]}]
    field_list = ['zero2', 'one2', 'five2', 'ten2', 'fifteen2', 'twenty2']
    cols = [{"field":'zero2', "title": ''},
            {"field":'one2', "title": 'LSTM(t=1)'},
            {"field":'five2', "title": 'LSTM(t=5)'},
            {"field":'ten2', "title": 'LSTM(t=10)'},
            {"field":'fifteen2', "title": 'LSTM(t=15)'},
            {"field":'twenty2', "title": 'LSTM(t=20)'},]
    data = []
    for i in range(len(lstm_table['data'])):
        col_dict = dict()
        col_dict[field_list[0]] = lstm_table['rowLabels'][i]
        for j in range(1, len(field_list)):
            col_dict[field_list[j]] = 0 if math.isnan(lstm_table['data'][i][j-1]) else lstm_table['data'][i][j-1]
        data.append(col_dict)
    lstm_table = {"cols":cols,"data":data}
    return series_line, legend, time_sum, time_steps, time_lstm, bar_lstm, lstm_table

if __name__ == '__main__':
    get_echart2_data()