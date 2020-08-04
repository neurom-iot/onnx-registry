from flask import Flask, render_template, request, redirect,send_file , send_from_directory
import os, time

# Flask 객체 생성
app = Flask(__name__)

########################
###### HTML 렌더링 ########
########################
@app.route('/', methods=['GET','POST'])
def main_page():
	# form 태그 (upload)
	if request.method == 'POST':
		f = request.files['file']
		file_name = f.filename
		f.save(file_name) #저장할 경로 + 파일명
		user_ip = request.remote_addr
		file_path = os.getcwd()#os.path.abspath(__file__) # 파일 저장 위치 경로

		upload_time = time.strftime('%y-%m-%d %H:%M:%S')

		# txt 파일 저장
		f = open("log.txt", 'a+')
		f.write('(' + upload_time + ') | ' + 'Upload_Success     '+' | ' + file_name+' | '+user_ip+' | ' + file_path + '\n')
		f.close()
		return redirect('/')

	# 파일 리스트 만들기 - onnx 파일만 출력
	dir_list = os.listdir()
	file_list = []
	for x in dir_list:
		if '.onnx' in x or '.txt' in x:
			file_list.append(x)

	# 파일 최근 수정날짜 정보
	latest_time = []
	latest_ls = latest_time.append

	# 파일 크기
	file_size = []
	size_ls = file_size.append

	# 리스트 정보들 만들어서 넘겨주기
	for i in file_list:
		latest_ls(time.ctime(os.path.getmtime(i))) ## 최근 수정날짜 정보
		size_ls(str(os.path.getsize(i))+' B') ## 파일크기

	# Log 데이터 넘겨주기

	fr = open("log.txt", 'r+')
	log_list = fr.readlines()
	print(type(log_list), log_list)
	fr.close()

	# file path 정보
	path_list = list(map(lambda x: os.getcwd()+"'\'" +x, file_list))
	return render_template('onnx_manager.html', data_list=file_list,
						   latest_time=latest_time, file_size = file_size,
						   log_list = log_list, path_list=path_list)

# 시각화 python cmd 명령어 실행으로 구현
@app.route('/visualization', methods=['GET','POST'])
def visualization():
	if request.method == 'POST':
		files = os.listdir("./") #지정된 디렉토리의 전체 파일 목록 가져오기

		for x in files:
			if(x == request.form['file']): # 파일이름이면
				path = "./"
				os.system('cd {}'.format(os.getcwd()))
				os.system('start chrome http://localhost:8080') # chrome 으로 웹 페이지 띄우기
				os.system('Netron {}'.format(x)) # netron 으로 파일 실행
				return

#파일 다운로드 처리
@app.route('/filedown', methods = ['GET', 'POST'
										  ''])
def down_file():
	if request.method == 'POST':
		files = os.listdir("./") #지정된 디렉토리의 전체 파일 목록 가져오기
		file_path = os.getcwd()  # os.path.abspath(__file__) # 파일 저장 위치 경로
		Download_time = time.strftime('%y-%m-%d %H:%M:%S')
		user_ip = request.remote_addr
		for x in files:
			if(x == request.form['file']): # 파일이름이면
				path = "./"

				# txt 파일 저장
				with open("log.txt", 'a+') as f:
					f = open("log.txt", 'a+')
					f.write('(' + Download_time + ') | ' + 'Download_Success' + ' | ' + x + ' | ' + user_ip + ' | ' + file_path + '\n')

				return send_file(path + request.form['file'],
						attachment_filename = request.form['file'], #
						as_attachment=True)

#서버 실행
if __name__ == '__main__':
	app.run(host='127.0.0.1', port=5050, debug = True)