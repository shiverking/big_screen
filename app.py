#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2020/8/26 14:48
# @Author : way
# @Site :
# @Describe:

from flask import Flask, render_template
from data import *

app = Flask(__name__)
#热更新
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True
#处理乱码
app.config['JSON_AS_ASCII']=False

@app.route('/index')
def index():
    data = SourceData()
    return render_template('index.html', form=data, title=data.title)

@app.route('/')
def home():
    return render_template('home.html', title='智能方法性能边界对比分析原理验证系统')

@app.route('/getTableData')
def getTableData():
    cols = []
    data = []
    data.append({"id":1,"username":"user1","sex":"男"})
    cols.append({"field":'id', "title": 'ID'})
    cols.append({"field":'username', "title": '用户名'})
    cols.append({"field":'sex', "title": '性别'})
    return {"cols":cols,"data":data}

if __name__ == "__main__":
    app.run(host='127.0.0.1', debug=False)
