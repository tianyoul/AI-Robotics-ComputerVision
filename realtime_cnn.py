import tensorflow as tf
import numpy as np
import cv2
from global_variables import*
from preprocessing import*

# Code based on https://github.com/sankit1/cv-tricks.com/tree/master/Tensorflow-tutorials/tutorial-2-image-classifier
from pynput.mouse import Button, Controller
mouse = Controller()


def find_gesture(result):
    '''Given the output of the model, find the category with the highest probability. If it's less than 80%,
    we consider it to be none since it is not confident enough. Note: we used softmax so the probability adds up to 1.
    We also used the none category.'''
    result = result[0]
    max_val = 0.0
    max_ind = 0
    for i in range(len(result)):
        if result[i] > max_val:
            max_val = result[i]
            max_ind = i
    if max_val > 0.8:
        return max_ind
    else:
        return 6

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
    gesture = preprocess(gesture)
    cv2.imshow("Processed", cv2.flip(gesture, 1))
    gesture = cv2.cvtColor(gesture, cv2.COLOR_GRAY2BGR)

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
    y_test_images = np.zeros((1, classes_num))

    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result = sess.run(y_pred, feed_dict=feed_dict_testing)
    print(classes[find_gesture(result)])
    ind = find_gesture(result)
    # Move the mouse according to the result of our model.
    if ind == 2:
        mouse.move(-20, 0)
    elif ind == 3:
        mouse.move(20, 0)
    elif ind == 4:
        mouse.move(0, -20)
    elif ind == 5:
        mouse.move(0, 20)
    elif ind == 2:
        mouse.click(Button.left)
    elif ind == 1:
        mouse.click(Button.left, 2)
    #print(result)

    cv2.imshow("Camera", cv2.flip(img, 1))

cv2.destroyAllWindows()
vidCap.release()