import threading
from threading import Thread
from flask import Flask, url_for, request, render_template, send_from_directory, flash, Response, make_response,redirect
from datetime import timedelta
from NER.predit import NER_Predit
from NER.train import model_train
from NER.utils import showDatas
import NER.globalVar as g
from NER.addTrain import addTrainTxt
import jinja2

app  =  Flask(__name__)
app.config['DEBUG']=True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)
app.config['SECRET_KEY'] = '123456'
msg=""
addS=""

def render_without_request(template_name, **context):
    """
    无效代码
    """
    env = jinja2.Environment(
        loader=jinja2.PackageLoader('NER-Project','templates')
    )
    template = env.get_template(template_name)
    return template.render(**context)



@app.route('/')
def mainpage():
    return render_template('mainpage.html')

@app.route('/predit')
def Predit():
    print("进入预测页面")
    return render_template('extendPredit.html')

@app.route('/predit', methods=['POST', 'GET'])
def Resquest():
    print("识别文本")
    if request.method == 'POST':
        msg = request.form['word']
        print("收到识别请求")
        lemmatization_mode = "false"
        if request.form.get('Re')== "Yes":#开启词性还原
            lemmatization_mode="true"

        # t1 = threading.Thread(target=NER_Predit,args=(msg, lemmatization_mode,))
        # t1.start()
        #flash("正在进行识别")
        msg=NER_Predit(msg,lemmatization_mode)
        # #msg=t1.get_result()
        # g._init()
        # msg=g.get_value("Preditres")
        #
        ss=""
        # for i in msg:
        #     ss=ss+str(i)+"\n"

        ss=ss+msg['人名'] +"\n\n"+msg['地名'] +"\n\n"+msg['组织名'] +"\n\n"+msg['专有名词'] +"\n"

        if ss=="":
            ss="未识别到任何命名实体"
        create_txt(ss)
        # t1 = threading.Thread(target=Predit_file, args=(msg, lemmatization_mode))
        # t1.start()
        # ss=g.get_value("Preditres")
        # if ss == "":
        #     print("ss NULL")
        #     ss="正在预测"
        return render_template('extendPredit.html',output=ss)

# @app.route('/predit')
# def Predit_file(msg,lemmatization_mode):
#     print("进入子线程predit")
#     msg = NER_Predit(msg, lemmatization_mode)
#     ss = ""
#     for i in msg:
#         ss = ss + str(i) + "\n"
#
#
#     # create_txt(ss)
#     # respon=make_response()
#     # respon.response=render_template('extendPredit.html',output="正在预测")
#     #app.jinja_env.globals.update(output=ss)
#     print("re")
#     # app.app_context().push()
#     # app.config['SERVER_NAME'] = '1234'
#     with app.app_context():  # 借助with语句使用app_context创建应用上下文
#         print(render_template('extendPredit.html',output="sadsd"))
#     #return render_template('extendPredit.html',output="sadsd")
#     #return redirect(url_for('refresh'))
#
#
# @app.route('/predit')
# def refresh():
#     global ss
#     print("跳转？")
#     app.app_context().push()
#     return render_template('downloadFile.html')


@app.route("/download")
def download():
    print("下载")

    return send_from_directory(r".\\static\\resource", filename="识别结果.txt", as_attachment=True)


#将结果保存到txt
def create_txt(msg):
    path = ".\\static\\resource\\"  # 创建的txt文件的存放路径
    name="识别结果"
    full_path = path + name + '.txt'
    file = open(full_path, 'w')
    file.write(msg)
    file.close()
    print(msg)
    print("保存成功")

@app.route("/downloadFile")
def downloadFile():
    print("下载文件")
    return render_template('downloadFile.html')

@app.route("/download1")
def download1():
    print("下载Bi-LSTM-Model.h5")
    return send_from_directory(r".\\NER\\data", filename="Bi-LSTM-Model.h5", as_attachment=True)

@app.route("/download2")
def download2():
    print("下载inverse_word_dictionary.pk")
    return send_from_directory(r".\\NER\\data", filename="inverse_word_dictionary.pk", as_attachment=True)

@app.route("/download3")
def download3():
    print("下载label_dictionary.pk")
    return send_from_directory(r".\\NER\\data", filename="label_dictionary.pk", as_attachment=True)

@app.route("/download4")
def download4():
    print("下载output_dictionary.pk")
    return send_from_directory(r".\\NER\\data", filename="output_dictionary.pk", as_attachment=True)

@app.route("/download5")
def download5():
    print("下载word_dictionary.pk")
    return send_from_directory(r".\\NER\\data", filename="word_dictionary.pk", as_attachment=True)

@app.route("/download6")
def download6():
    print("下载train.txt")
    return send_from_directory(r".\\NER\\data", filename="train.txt", as_attachment=True)

@app.route("/download7")
def download7():
    print("下载识别结果.txt")
    return send_from_directory(r".\\static\\resource", filename="识别结果.txt", as_attachment=True)

@app.route("/train")
def trainModel():
    print("训练模型1")
    return render_template('modelTrain.html',Accuracy="0%",train_code="model = Sequential()\nmodel.add(Embedding("
                                                                      "input_dim=vocab_size + 1, "
                                                                      "output_dim=output_dim, "
                                                                      "input_length=input_shape, "
                                                                      "mask_zero=True))\nmodel.add(Bidirectional("
                                                                      "LSTM(units=n_units, activation='selu',"
                                                                      "return_sequences=True)))\nmodel.add("
                                                                      "TimeDistributed(Dense(label_size + 1, "
                                                                      "activation='sigmoid')))\nmodel.compile("
                                                                      "optimizer='adam', loss='binary_crossentropy', "
                                                                      "metrics=['accuracy'])")

@app.route('/train', methods=['POST', 'GET'])
def Resquest1():
    print("训练模型2")
    if request.method == 'POST':
        msg = request.form['train_model']
        print("收到训练请求")
        acc=""
        try:
            if request.form.get('mode') == "aauto":  # 默认
                #flash('正在训练默认模型....')
                msg=""


                acc=str(model_train(msg))+"%"
            else:
                #flash('正在训练自定义模型....')
                acc=str(model_train(msg))+"%"
            print("执行成功")

        except Exception:
            print("出现错误,输入的代码有问题")
            acc="Err：代码有误，请检查!!"
        return render_template('modelTrain.html',Accuracy=acc,train_code="训练成功!!!")


@app.route('/add')
def addTrain():
    print("进入添加页面")
    return render_template('extendAdd.html')

@app.route('/add',methods=['POST', 'GET'])
def addTrain2():
    print("提交")
    if request.method == 'POST':
        msg1 = request.form['add_word']
        msg2 =request.form['word_res']
        print("msg1： ",msg1)
        print("msg2： ",msg2)
        if msg1 !="":
            global addS
            addS=addTrainTxt(msg1)
        elif msg2 !="":
            print("加入")
            insertTrain(msg2)
            llist = showDatas()
            return render_template('addRes.html', num_word=llist[0], num_dir=llist[1])


    return render_template('extendAdd.html',word_res=addS)


def insertTrain(sss):
    print("插入成功")

    #print(sss)

    gea=1
    s1 = ""
    s2=""
    s3=""
    index1 = range(0, len(sss), 1)
    i=0
    while i<len(sss):

        temp = sss[i]
        #print(i)
        if sss[i] != "\r" and gea == 1:
            s1 = s1 + sss[i]
        elif sss[i] == "\r" and gea == 1:
            i = i + 1
            s1 = s1 + "\t"
            gea = 2
        elif sss[i] != "\r" and gea == 2:
            s2 = s2 + sss[i]
        elif sss[i] == "\r" and gea == 2:
            i = i + 1
            s2 = s2 + "\t"
            gea = 3
        elif sss[i] != "\r" and gea == 3:

            s3 = s3 + sss[i]
        elif sss[i] == "\r" and gea == 3:
            i = i + 1
            s3 = s3 + "\t"
            gea = 1

        i+=1
        # elif i==(len(sss)-1):
        #     s3 = s3 + "\t"
        #     s3 = s3 + sss[i]

    print("s1: ",s1)
    print("s2: ", s2)
    print("s3: ", s3)

    s1=s1+"\n"
    s2 = s2 + "\n"
    s3 = s3 + "\n"

    path = "NER/DATA/"  # 创建的txt文件的存放路径
    name = "train"
    full_path = path + name + '.txt'
    file = open(full_path, 'a')
    file.write(s1 + s2 + s3)
    file.close()


if __name__ == "__main__":
    app.run(debug=False,threaded=True)