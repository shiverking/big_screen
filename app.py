#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2020/8/26 14:48
# @Author : way
# @Site :
# @Describe:

from flask import Flask, render_template
from data import *
from flask import Flask,jsonify,request

app = Flask(__name__)
#热更新
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True
#处理乱码
app.config['JSON_AS_ASCII']=False

@app.route('/index',methods=['post'])
def index():
    inputs = request.json.get('data')
    scenario_dict = {'0': '正南打击利佩茨克机场数据', '1': '正西无人机打击圣彼得堡数据', '2':'无人机侦察数据', '3': '正西轰炸机打击莫斯科数据'}
    type_name = {'0': '战斗机', '1': '无人机', '2':'无人机', '3':'轰炸机'}
    test_name_dict = {
            '1': 'F-22 科加尔尼西亚 #1',
            '2': 'F-22 科加尔尼西亚 #2',
            '3': 'F-22 科加尔尼西亚 #3',
            '4': 'F-22 科加尔尼西亚 #4',
            '5': 'F-22 科加尔尼西亚 #5',
            '6': 'F-22 科加尔尼西亚 #6',
            '7': 'F-22 科加尔尼西亚 #7',
            '8': 'F-22 科加尔尼西亚 #8',
            '9': 'F-22 科加尔尼西亚 #9',
            '10': 'F-22 科加尔尼西亚 #10',
            '11': '罗斯福 #42',
            '12': '罗斯福 #43',
            '13': '罗斯福 #44',
            '14': '罗斯福 #45',
            '15': '罗斯福 #46',
            '16': '罗斯福 #47',
            '17': '罗斯福 #48',
            '18': 'B-52 RAF Fairford #1',
            '19': 'B-52 RAF Fairford #2',
            '20': 'B-52 RAF Fairford #3',
            '21': 'B-2 RAF Fairford #1',
            '22': 'B-2 RAF Fairford #2',
            '23': 'B-21 RAF Fairford #1',
            '24': 'B-21 RAF Fairford #1'
        }
    data = SourceData(scenario_dict[inputs['situation']], test_name_dict[inputs['target']], type_name[inputs['situation']])
    return render_template('index.html', form=data, title=data.title)

@app.route('/')
def home():
    return render_template('home.html', title='智能方法性能边界对比分析原理验证系统')

if __name__ == "__main__":
    app.run(host='127.0.0.1', debug=False)
