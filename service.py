from model.DBN import DBN_RES

def get_echart1_data():
    plt_data, legend = DBN_RES()
    print(plt_data)
    xAxis = [x for x in range(200)]
    series = list()
    for data_list in plt_data:
        print(data_list)
        series.append({'name': data_list['name'],'type': 'line', 'data': data_list['data'], 'smooth': True})
    return xAxis, series, legend

if __name__ == '__main__':
    print(get_echart1_data())