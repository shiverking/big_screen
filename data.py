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
        self.counter = {}
        self.counter2 = {}
        self.echart1_data = {}
        self.echart2_data = {}
        self.echarts3_1_data = {}
        self.echarts3_2_data = {}
        self.echarts3_3_data = {}
        self.echart4_data = {}
        self.echart5_data = {}
        self.echart6_data = {}
        self.echart7_data = {}

class SourceData(SourceDataDemo):

    def __init__(self, scenario, testName, typeName):
        """
        按照 SourceDataDemo 的格式覆盖数据即可
        """
        super().__init__()
        import copy
        upper_left_corner, data_pie, time_sum_dbn, x, time_line_dbn, upper_right_corner_dbn, lower_right_corner_dbn = get_dbnModel_res(scenario, testName, typeName)
        bottom_left_corner, time_sum_lstm, x, time_line_lstm, upper_right_corner_lstm, lower_right_corner_lstm = get_lstmModel_res(scenario, testName, typeName)
        upper_right_corner = dict()
        upper_right_corner['dbn'] = copy.deepcopy(upper_right_corner_dbn)
        upper_right_corner['lstm'] = copy.deepcopy(upper_right_corner_lstm)
        for key in upper_right_corner_dbn.keys():
            upper_right_corner_dbn[key]['series'] += upper_right_corner_lstm[key]['series']
        upper_right_corner['dbn_and_lstm'] = copy.deepcopy(upper_right_corner_dbn)
        lower_right_corner = dict()
        lower_right_corner['dbn'] = copy.deepcopy(lower_right_corner_dbn)
        lower_right_corner['lstm'] = copy.deepcopy(lower_right_corner_lstm)
        for key in lower_right_corner_dbn.keys():
            for key2 in lower_right_corner_dbn[key].keys():
                lower_right_corner_dbn[key][key2]['series'] += lower_right_corner_lstm[key][key2]['series']
        lower_right_corner['dbn_and_lstm'] = copy.deepcopy(lower_right_corner_dbn)

        self.echart1_data = {
            'title': 'DBN意图识别结果图',
            'xAxis': upper_left_corner['xAxis'],
            'series': upper_left_corner['series'],
            'legend': upper_left_corner['legend']
        }
        self.echart6_data = {
            'title': data_pie['title'],
            'series': [
                        {
                        'type': 'pie',
                        'radius': '60%',
                        'data': data_pie['series'],
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
            'data': bottom_left_corner
        }
        self.echart7_data = {
            'title': '不同模型的预测时间',
            'xAxis': x,
            'dbnSeries':time_line_dbn,
            'lstmSeries': time_line_lstm,
            'dbnLegend': ['模型预测时间-DBN'],
            'series': time_line_dbn + time_line_lstm,
            'legend': ['模型预测时间-DBN','模型预测时间-LSTM' ],
            'lstmLegend': ['模型预测时间-LSTM'],
        }
        self.echart5_data = {
            'title': '不同模型的整体性能指标分析',
            'data': upper_right_corner
        }
        self.echart4_data = {
            'title': '不同模型的各类别性能指标分析',
            'data': lower_right_corner
        }