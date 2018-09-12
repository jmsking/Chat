#! /user/bin/python3

from flask import Flask,render_template,request
from ChatBot import ChatBot

app = Flask(__name__)		#创建一个wsgi应用

@app.route('/')
def chat_page():
	return render_template("chat.html")

@app.route('/obtain_answer')
def obtain_answer():
	question = request.args.get('question')
	chatBot = ChatBot()
	answer = chatBot.infer(question)
	return answer

if __name__ == '__main__':
	app.run(debug=True)		#启动app的调试模式