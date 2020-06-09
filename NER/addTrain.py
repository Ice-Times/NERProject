#训练集添加程序
import nltk

def addTrainTxt(document):

    sentences = nltk.sent_tokenize(document)

    data = []
    temp = ""
    for sent in sentences:
        temp = nltk.pos_tag(nltk.word_tokenize(sent))

    s1 = ""
    for i in temp:
        s1 = s1 + i[0] + "\n"+i[1]+"\n"+"O"+"\n"

    return s1

