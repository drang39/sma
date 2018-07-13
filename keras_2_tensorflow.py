
# coding: utf-8

# In[1]:


from keras.models import load_model
import tensorflow as tf
from pathlib import Path
from keras import backend as K


# In[2]:


#load keras model
model_fn = 'cu_inceptionv3_150e6400.h5'
net_model = load_model(model_fn)


# In[3]:


#get keras session
sess = K.get_session()


# In[4]:


#tf.train.write_graph(graph_or_graph_def,logdir,name,as_text=True)
#model weight and name
f = 'model.ascii'
#output_folder
output_fld=''
#write graph
tf.train.write_graph(sess.graph.as_graph_def(), output_fld, f, as_text=True)


# In[5]:


from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from tensorflow.tools.graph_transforms import TransformGraph


# In[6]:


#number of output
num_output=1
#output prefix
output_node_prefix='final_output'
#final output name set to tf.identity
pred = [None]*num_output
pred_node_names = [None]*num_output
for i in range(num_output):
    pred_node_names[i] = output_node_prefix+str(i)
    pred[i] = tf.identity(net_model.outputs[i], name=pred_node_names[i])


# In[7]:


#set transfer object
transforms = ["quantize_weights", "quantize_nodes"]
transformed_graph_def = TransformGraph(sess.graph.as_graph_def(), [], pred_node_names, transforms)
# constant_graph = graph_util.convert_variables_to_constants(sess, transformed_graph_def, pred_node_names)


# In[8]:


#set output model path
output_model_file='tensorflow_model.pb'


# In[9]:


#set constant_graph
constant_graph = graph_util.convert_variables_to_constants(sess, transformed_graph_def, pred_node_names)


# In[11]:


#write gragh
graph_io.write_graph(constant_graph, output_fld,output_model_file, as_text=False)
#print model saving and the path of model
print('saved the freezed graph (ready for inference) at: ', str(Path(output_fld) / output_model_file))


# In[12]:


#load tensorflow model function 
#load gragh


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

