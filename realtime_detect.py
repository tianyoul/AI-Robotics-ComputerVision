import tensorflow as tf
import numpy as np
import cv2

classes = ['fist', 'hand', 'left', 'right', 'up', 'down']

def find_gesture(result):
    result = result[0]
    max_val = 0.0
    max_ind = 0
    for i in range(len(result)):
        if result[i] > max_val:
            max_val = result[i]
            max_ind = i

    return max_ind

## Let us restore the saved model
sess = tf.Session()
# Step-1: Recreate the network graph. At this step only graph is created.
saver = tf.train.import_meta_graph('model.meta')
# Step-2: Now let's load the weights saved using the restore method.
saver.restore(sess, tf.train.latest_checkpoint('./'))

# Accessing the default graph which we have restored
graph = tf.get_default_graph()

# Now, let's get hold of the op that we can be processed to get the output.
# In the original network y_pred is the tensor that is the prediction of the network
y_pred = graph.get_tensor_by_name("y_pred:0")




vidCap = cv2.VideoCapture(0)

top_left = (50, 100)
bottom_right = (500, 600)  # In opencv, need to be reversed in np
green = (100, 180, 50)
num_channels=3
image_size = 500

while True:
    x = cv2.waitKey(10)
    char = chr(x & 0xFF)
    if char == 'q':
        break
    ret, img = vidCap.read()

    cv2.rectangle(img, top_left, bottom_right, green)
    gesture = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    images = []
    # Resizing the image to our desired size and preprocessing will be done exactly as done during training
    image = cv2.resize(gesture, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0 / 255.0)
    x_batch = images.reshape(1, image_size, image_size, num_channels)

    ## Let's feed the images to the input placeholders
    x = graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    y_test_images = np.zeros((1, 4))

    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result = sess.run(y_pred, feed_dict=feed_dict_testing)
    print(classes[find_gesture(result)])
    cv2.imshow("Camera", img)

cv2.destroyAllWindows()
vidCap.release()