import tensorflow as tf, sys



# change this as you see fitg'

t=""
try:
	t = sys.argv[1]
except Exception:
	print('bad')

if t=="":
	image_path = '/Programming/Python/NeuralNetwork ImageRec/test1.jpg'
else:
	image_path = t

# Read in the image_data
image_data = tf.gfile.FastGFile(image_path, 'rb').read()

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("/Programming/Python/NeuralNetwork ImageRec/trained_data/output_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("/Programming/Python/NeuralNetwork ImageRec/trained_data/output_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})
    
    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        score = score * 100
        print('%s  %.5f;' % (human_string, score))
        
    