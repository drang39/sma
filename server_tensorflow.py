from flask import request, Flask
import json
import numpy as np
import io
from PIL import Image
# from keras.models import load_model
import tensorflow as tf
import cv2
import time
app = Flask(__name__)

def load_graph(frozen_graph_filename):
    # We parse the graph_def file
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
 
    # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def, 
            input_map=None, 
            return_elements=None, 
            name="prefix", 
            op_dict=None, 
            producer_op_list=None
        )
    return graph



def loading_model():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    model_fn = 'k_model.h5.pb'
    global model,graph,input_shape,sess
    graph = load_graph(model_fn)
    sess = tf.Session(graph=graph,config=config)
    input_shape = (150,150)


@app.route("/predict_file", methods=['POST'])
def predict_file():
    if request.method =='POST':
        if request.files.get('img'):
            img = request.files['img'].read()
            img = Image.open(io.BytesIO(img))
            img = img.resize(input_shape)
            img = np.expand_dims(img,axis=0)
            img = img/255.
            with graph.as_default():
                x = graph.get_tensor_by_name('prefix/input_5:0')
                y = graph.get_tensor_by_name('prefix/output_node0:0')
                prediction = sess.run(y, feed_dict={x: img})
            prediction = str(prediction.argmax(1)[0])
        return prediction

@app.route("/test", methods=['POST'])
def test():
    print(123)
    return 'received'



if __name__ == "__main__":
    s = time.time()
    loading_model()
    print(time.time()-s)
    app.run()  