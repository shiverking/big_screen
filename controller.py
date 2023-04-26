from flask import Flask,jsonify,request
from flask_cors import CORS
from data import *

server = Flask(__name__)
server.config['JSON_AS_ASCII']=False

@server.route('/intentRecognition',methods=['post'])
def intentRecognition():
    situation = request.json.get('situation')
    target = request.json.get('target')

    if situation !=None and target !=None:
        scenario_dict = {'0': '正南打击利佩茨克机场数据', '1': '正西无人机打击圣彼得堡数据', '2':'无人机侦察数据', '3': '正西轰炸机打击莫斯科数据'}
        type_name = {'0': '战斗机', '1': '无人机', '2':'侦察机', '3':'轰炸机'}
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
                '24': 'B-21 RAF Fairford #1',
                '25': 'MQ-4 RAF Mildenhall #1',
                '26': 'MQ-4 RAF Mildenhall #2',
            }
        data = SourceData(scenario_dict[situation], test_name_dict[target], type_name[situation])
    dict = {}
    dict['data'] = data
    return jsonify(dict)

if __name__ == "__main__":
    ip="0.0.0.0"
    port = 3389
    CORS(server, supports_credentials=True)
    server.run(host = ip, port = port, use_reloader=False)