#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2020/8/26 14:48
# @Author : way
# @Site : 
# @Describe:

from service import get_dbnModel_res, get_lstmModel_res

import json

class SourceDataDemo:

    def __init__(self):
        self.title = '智能方法性能边界对比分析原理验证系统'
        self.counter = {'name': '2018年总收入情况', 'value': 12581189}
        self.counter2 = {'name': '2018年总支出情况', 'value': 3912410}
        self.echart1_data = {
            'title': '行业分布',
            'data': [ 
                {"name": "商超门店", "value": 47},
                {"name": "教育培训", "value": 52},
                {"name": "房地产", "value": 90},
                {"name": "生活服务", "value": 84},
                {"name": "汽车销售", "value": 99},
                {"name": "旅游酒店", "value": 37},
                {"name": "五金建材", "value": 2},
            ]
        }
        self.echart2_data = {
            'title': '省份分布',
            'data': [
                {"name": "浙江", "value": 47},
                {"name": "上海", "value": 52},
                {"name": "江苏", "value": 90},
                {"name": "广东", "value": 84},
                {"name": "北京", "value": 99},
                {"name": "深圳", "value": 37},
                {"name": "安徽", "value": 150},
            ]
        }
        self.echarts3_1_data = {
            'title': '年龄分布',
            'data': [
                {"name": "0岁以下", "value": 47},
                {"name": "20-29岁", "value": 52},
                {"name": "30-39岁", "value": 90},
                {"name": "40-49岁", "value": 84},
                {"name": "50岁以上", "value": 99},
            ]
        }
        self.echarts3_2_data = {
            'title': '职业分布',
            'data': [
                {"name": "电子商务", "value": 10},
                {"name": "教育", "value": 20},
                {"name": "IT/互联网", "value": 20},
                {"name": "金融", "value": 30},
                {"name": "学生", "value": 40},
                {"name": "其他", "value": 50},
            ]
        }
        self.echarts3_3_data = {
            'title': '兴趣分布',
            'data': [
                {"name": "汽车", "value": 4},
                {"name": "旅游", "value": 5},
                {"name": "财经", "value": 9},
                {"name": "教育", "value": 8},
                {"name": "软件", "value": 9},
                {"name": "其他", "value": 9},
            ]
        }
        self.echart4_data = {
            'title': '时间趋势',
            'data': [
                {"name": "安卓", "value": [3, 4, 3, 4, 3, 4, 3, 6, 2, 4, 2, 4, 3, 4, 3, 4, 3, 4, 3, 6, 2, 4, 4]},
                {"name": "IOS", "value": [5, 3, 5, 6, 1, 5, 3, 5, 6, 4, 6, 4, 8, 3, 5, 6, 1, 5, 3, 7, 2, 5, 8]},
            ],
            'xAxis': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '11', '12', '13', '14', '15', '16', '17',
                      '18', '19', '20', '21', '22', '23', '24'],
        }
        self.echart5_data = {
            'title': '省份TOP',
            'data': [
                {"name": "浙江", "value": 2},
                {"name": "上海", "value": 3},
                {"name": "江苏", "value": 3},
                {"name": "广东", "value": 9},
                {"name": "北京", "value": 15},
                {"name": "深圳", "value": 18},
                {"name": "安徽", "value": 20},
                {"name": "四川", "value": 13},
            ]
        }
        self.echart6_data = {
            'title': '一线城市情况',
            'data': [
                {"name": "浙江", "value": 80, "value2": 20, "color": "01", "radius": ['59%', '70%']},
                {"name": "上海", "value": 70, "value2": 30, "color": "02", "radius": ['49%', '60%']},
                {"name": "广东", "value": 65, "value2": 35, "color": "03", "radius": ['39%', '50%']},
                {"name": "北京", "value": 60, "value2": 40, "color": "04", "radius": ['29%', '40%']},
                {"name": "深圳", "value": 50, "value2": 50, "color": "05", "radius": ['20%', '30%']},
            ]
        }
        self.map_1_data = {
            'symbolSize': 100,
            'data': [
                {'name': '海门', 'value': 239},
                {'name': '鄂尔多斯', 'value': 231},
                {'name': '招远', 'value': 203},
            ]
        }
        self.echart7_data = {
            'title': '一线城市情况',
            'data': [
                {"name": "浙江", "value": 80, "value2": 20, "color": "01", "radius": ['59%', '70%']},
                {"name": "上海", "value": 70, "value2": 30, "color": "02", "radius": ['49%', '60%']},
                {"name": "广东", "value": 65, "value2": 35, "color": "03", "radius": ['39%', '50%']},
                {"name": "北京", "value": 60, "value2": 40, "color": "04", "radius": ['29%', '40%']},
                {"name": "深圳", "value": 50, "value2": 50, "color": "05", "radius": ['20%', '30%']},
            ]
        }

class SourceData(SourceDataDemo):

    def __init__(self):
        """
        按照 SourceDataDemo 的格式覆盖数据即可
        """
        super().__init__()
        upper_left_corner, data_pie, time_sum_dbn, x, time_line_dbn, upper_right_corner_dbn, lower_right_corner_dbn = get_dbnModel_res()
        bottom_left_corner, time_sum_lstm, x, time_line_lstm, upper_right_corner_lstm, lower_right_cornerlstm = get_lstmModel_res()

        self.echart1_data = {
            'title': 'DBN意图识别结果图',
            'xAxis': upper_left_corner.xAxis,
            'series': upper_left_corner.series,
            'legend': upper_left_corner.legend
        }
        self.echart6_data = {
            'title': data_pie.title,
            'series': [
                        {
                        'type': 'pie',
                        'radius': '60%',
                        'data': data_pie.series,
                        'emphasis': {
                            'itemStyle': {
                            'shadowBlur': 10,
                            'shadowOffsetX': 0,
                            'shadowColor': 'rgba(0, 0, 0, 0.5)'
                            }
                        },
                        'label': {
                            'normal': {
                                'show': 'true',
                                'formatter': '{b}: {c}({d}%)' #自定义显示格式(b:name, c:value, d:百分比)
                            }
                        },
                        }
                    ]
        }
        self.counter = {'value':f'{time_sum_dbn:.2f}', 'name': 'DBN运行时间(单位：秒)'}
        self.counter2 = {'value':f'{time_sum_lstm: .2f}', 'name': 'LSTM运行时间(单位：秒)'}
        self.echart2_data = {
            'title': 'LSTM意图识别结果图',
            'xAxis': bottom_left_corner['time_step=1'].xAxis,
            'series': bottom_left_corner['time_step=1'].series,
            'legend': bottom_left_corner['time_step=1'].legend
        }
        self.echart7_data = {
            'title': '不同模型的预测时间',
            'xAxis': x,
            'series': time_line_dbn + time_line_lstm,
            'legend': ['模型预测时间-DBN', '模型预测时间-LSTM']
        }
        self.echart5_data = {
            'title': '不同模型的预测精度',
            'yAxis': upper_right_corner_dbn['time_step=1'].yAxis,
            'series': upper_right_corner_dbn['time_step=1'].series
        }
        self.echart4_data = {
            'title': '不同模型的预测精度',
            'yAxis': lower_right_corner_dbn['precision']['time_step=1'].yAxis,
            'series': upper_right_corner_dbn['precision']['time_step=1'].series
        }