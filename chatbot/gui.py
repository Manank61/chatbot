import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
model = load_model('F:\python project\chatbot\model')
import json
import random
intents = json.loads(open('F:\python project\chatbot\intents.json').read())
words = pickle.load(open('F:\python project\chatbot\words.pkl','rb'))
classes = pickle.load(open('F:\python project\chatbot\classes.pkl','rb'))

def clean_up_sentence(sentence):
    #tokenize sentence
    sentence_words=nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words
def bow(sentence,words,show_details=True):
    #tokenize the sentence
    clean_up_sentence(sentence)
    bag=[0]*len(words)
    for s in clean_up_sentence(sentence):
        for i,w in enumerate(words):
            if(w==s):
                bag[i]=1
    return(np.array(bag))  
def predict_classes(sentence,model):
    p=bow(sentence,words,show_details=False)
    pre=model.predict(np.array([p]))[0]
    error_threshold=0.25
    results=[[i,r] for i,r in enumerate(pre) if r>error_threshold]
    results.sort(key=lambda x : x[1], reverse=True)
    return_list=[]
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in intents['intents']:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result
def chatbot_response(text):
    ints = predict_classes(text,model)
    res = getResponse(ints, intents)
    return res

#tkinter
import tkinter
from tkinter import *
root=Tk()

def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)
    if msg!='':
        chatlog.config(state=NORMAL)   
        chatlog.insert(END,"YOU :"+msg+'\n\n')
        chatlog.config(foreground='#1fe026',font=('arial',12))

        res=chatbot_response(msg) 
        chatlog.insert(END,'NARUTO : '+res + '\n\n')
        chatlog.config(state=DISABLED)
        chatlog.yview(END)

root.title('chatguy')
root.geometry('400x500')
root.resizable(width=False, height=False)

#create chat window
chatlog=Text(root,bd=0,bg='white',height='8',width='50',font='arial')

chatlog.config(state=DISABLED)

#bind scrollbar to chat window
scrollbar=Scrollbar(root,command=chatlog.yview,cursor='heart')
chatlog['yscrollcommand']=scrollbar.set

#create button(send)
sendbutton = Button(root, font=("Verdana",12,'bold'), text="Send", width="12", height=5,bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',command= send )
EntryBox = Text(root, bd=0, bg="white",width="29", height="5", font="Arial")

#components on the screen
scrollbar.place(x=376,y=6, height=386)
chatlog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
sendbutton.place(x=6, y=401, height=90)

root.mainloop()