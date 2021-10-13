from flask import Flask, render_template, send_from_directory, request, redirect, session, send_file
import os, time
import zipfile, json

from onnx_distinguish_run import Distinguish_onnx

# Flask 객체 생성
app = Flask(__name__)

import netron

server = 'localhost'
netron_port = 8080

# folder name
onnx_folder_path = 'onnx_folder/'
log_file_path = 'log_folder/'


# main page
@app.route('/', methods=['GET', 'POST'])
def main_page():
    # 파일 기본 정보들 출력
    zip_file_list = get_filelist()  # 파일 이름 리스트 반환
    zip_file_latest_time = get_file_latest_time(zip_file_list)  # 파일 마지막 수정시간 반환
    zip_file_size = get_filesize(zip_file_list)  # 파일 크기 반환
    # -------------------------------------------------------
    # upload
    if request.method == 'POST':
        upload_process()
    # -------------------------------------------------------

    # log 데이터
    fr = open(log_file_path + "log.txt", 'r+')
    log_list = fr.readlines()
    #print(type(log_list), log_list)
    fr.close()

    # -----------------------------
    keywords_list = []
    author_list = []
    for i in zip_file_list:
        file_name = onnx_folder_path + i
        ext = zipfile.ZipFile(file_name).extract('package.json')

        with open(ext, "r", encoding="utf8") as f:
            contents = f.read()  # string 타입
            json_data = json.loads(contents)


        keywords = json_data["keywords"]
        keywords = ', '.join(keywords)
        print(keywords)
        keywords_list.append(keywords)
        print(keywords_list)
        author = json_data["author"]
        author_list.append(author)

    return render_template('mainpage.html',
                           zip_file_list=zip_file_list,
                           zip_file_latest_time=zip_file_latest_time,
                           zip_file_size=zip_file_size,
                           log_list=log_list,
                           len=len(keywords_list),
                           keywords_list=keywords_list,
                           author_list=author_list,
                           zip=zip,  # zip이란 python 함수도 jinja에 같이 넘긴 것
                           )


# upload 기능 + log write 기능
def upload_process():
    f = request.files['file']
    print(f)
    file_name = f.filename
    file_path = os.getcwd() + '\\' + onnx_folder_path[:-1] + '\\'  # 파일 저장 위치 경로
    f.save(file_path + file_name)  # 저장할 경로 + 파일명
    print(file_path + file_name + '에 파일 저장 완료')

    user_ip = request.remote_addr
    upload_time = time.strftime('%y-%m-%d %H:%M:%S')

    # txt 파일 저장
    f = open(log_file_path + "log.txt", 'a+')
    f.write(
        '(' + upload_time + ') - ' + 'Upload_Success     ' + ' - ' + file_name + ' - ' + user_ip + ' - ' + file_path + '\n')
    f.close()
    return redirect('/')

@app.route('/post', methods=['POST'])
def post():
    value = request.form['author']
    board = request.form['description']
    model = request.form['model']

    f = open(model + ".json", 'a+')
    f.write('{\"author\":\"' + value + '\",\"description\":\"' + board + '\"}')
    f.close()

    return redirect('/')

@app.route('/upload')
def upload():
    return render_template('send.html')

# download 기능
@app.route('/download/<filename>', methods=['GET', 'POST'])
def download(filename):
    download_time = time.strftime('%y-%m-%d %H:%M:%S')
    # log txt 파일 저장
    f = open(log_file_path + "log.txt", 'a+')
    f.write(
        '(' + download_time + ') - ' + 'Download_Success ' + ' - ' + filename + ' - \n')
    f.close()

    return send_from_directory('onnx_folder', filename)  # directory 명, 파일이름


# Visualization
@app.route('/visualizations/<filename>', methods=['GET', 'POST'])
def visualization(filename):

    file_name = onnx_folder_path + filename
    ext = zipfile.ZipFile(file_name).extract('package.json')

    with open(ext, "r", encoding="utf8") as f:
        contents = f.read()  # string 타입
        json_data = json.loads(contents)


    print(file_name)
    model_name = str(os.path.basename(file_name))
    name = json_data["name"]
    version = json_data["version"]
    description = json_data["description"]
    keywords = json_data["keywords"]
    keywords = ', '.join(keywords)
    author = json_data["author"]
    license = json_data["license"]
    #exc_file = json_data["nodes"[""]]
    #print('model_type 정보는=', model_type)

    #netron.start(file=file_name, browse=False, port=netron_port, host=server)
    return render_template('netron_wrapper.html',
                           model_name=model_name,
                           name=name,
                           version=version,
                           description=description,
                           keywords=keywords,
                           author=author,
                           license=license
                           )


## 기능함수들
# 1. 파일 이름 리스트 반환
def get_filelist():
    # 파일 리스트 만들기 - onnx 파일만 출력
    dir_list = os.listdir(onnx_folder_path)
    zip_file_list = []
    json_file_list = []
    for x in dir_list:
        if '.zip' in x:
            zip_file_list.append(x)
    return zip_file_list



def get_file_latest_time(file_list):
    # 파일 최근 수정날짜 정보
    latest_time = []
    latest_ls = latest_time.append
    for i in file_list:
        latest_ls(time.ctime(os.path.getmtime(onnx_folder_path + i)))  ## 최근 수정날짜 정보
    return latest_time


def get_filesize(file_list):
    # 파일 크기
    file_size = []
    size_ls = file_size.append

    # 리스트 정보들 만들어서 넘겨주기
    for i in file_list:
        size_ls(str(os.path.getsize(onnx_folder_path + i)) + ' B')  ## 파일크기
    return file_size


import onnx


def onnxruntime_imformation(onnx_file_name):
    model1 = onnx.load(onnx_file_name)

    # input_type 정보 출력
    s = model1.graph.input[0].type.tensor_type.shape.dim
    ls = list(map(lambda x: refunc(x), s))
    input_type = 'tensor(' + ','.join(ls) + ')'
    print(input_type)

    # output_type 정보 출력
    o = model1.graph.output[0].type.tensor_type.shape.dim
    ls = list(map(lambda x: refunc(x), o))
    output_type = 'tensor(' + ','.join(ls) + ')'
    print(output_type)

    return input_type, output_type


import re


def refunc(x):
    try:
        k = re.sub(r'^"|"$', '', str(x).split(' ')[1].strip())
    except:
        k = str(x)
    return k


# onnx 판별식 코드(ml,dl,snn인지 판별)
def onnx_type(onnx_file_name):
    distinguish_onnx = Distinguish_onnx(onnx_file_name)  # 모델경로
    # snn인지 아닌지 먼저 판별
    a = Distinguish_onnx.getModelOperator(distinguish_onnx)
    print('Operator=', a)
    model_framework = distinguish_onnx.getModelFramework()
    print('Framework=', model_framework)

    if model_framework == 'SNNnengo':
        return 'SNN'
    else:
        model_domain = distinguish_onnx.getModeldomain()
        if model_domain == 'ai.onnx':
            # ml의 경우
            # print("ml입니다")
            return 'ml'
        else:  # onnx 일경우
            # dl의 경우
            # print("dl입니다")
            return 'dl'


'''
def getModelOperator(onnx_file_name):
    #for i in range(len(onnx_file_name.onnx_load_model.graph.node)):
    division=str(onnx_file_name.producer_name)
    print(division)
'''

# 서버 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5065, debug=True)
