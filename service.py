from model.DBN import DBN_RES

def get_echart1_data():
    plt_data, legend_line, legend_pie, perc_pie, title_pie = DBN_RES()
    xAxis_line = [x for x in range(200)]
    series_line = list()
    for data_list in plt_data:
        series_line.append({'name': data_list['name'],'type': 'line', 'data': data_list['data'], 'smooth': 'true'})
    data_pie = list()
    for leg, per in list(zip(legend_pie, perc_pie)):
        data_pie.append({'value': leg, 'name': per})
    return xAxis_line, series_line, legend_line, data_pie, title_pie