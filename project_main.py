import cv2
import numpy as np
import pika
import json
import pickle
from sklearn.externals import joblib
from model import Model
import project_1
import project_2



def load_model():
    global clf
    global model
    clf = joblib.load("/diskb/DH/recognition/project-3/train_model.m")
    model = Model().cuda()
    #model.load("/diskb/DH/recognition/project-3/model.pth")
    model.restore("/diskb/DH/recognition/project-3/model.pth")

def producer(server, qu_name, value):
    value = json.dumps(value)
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=server))
    channel = connection.channel()
    #channel.queue_declare(queue=qu_name)#
    channel.basic_publish(exchange='machineRoomExchange',
                          routing_key='detectResponseQueueKey',
                          body=value)
def get_position(body):
    position = []
    element = body.get('item_list')
    for c in element:
        position.append(c.get('box'))
    return position

def callback(ch, method, properties, body):
    body = json.loads(body)
    img_path = body.get('path')
    position = get_position(body)
    result = project_1.First_method(img_path, clf, model, position)
    for num in range(len(result)):
        body.get('item_list')[num]['position'] = result[num]
    print(body)
    producer(server='localhost', qu_name='lmachineRoomExchange', value=body)
    ch.basic_ack(delivery_tag=method.delivery_tag)

def consumer(server, qu_name):
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=server))
    channel = connection.channel()
    channel.basic_consume(qu_name, callback, False)
    channel.start_consuming()


clf = ''
model = ''
load_model()
print('-----waiting!-----')
consumer(server='localhost', qu_name='detectMainResponseQueue')
