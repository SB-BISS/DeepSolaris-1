'''Visualization of the filters of VGG16, via gradient ascent in input space.
This script can run on CPU in a few minutes.
Results example: http://i.imgur.com/4nj4KjN.jpg
'''
import numpy as np
import time
import sys
import argparse
from keras.preprocessing.image import save_img
from keras.models import load_model
from keras import backend as K


def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())


def visualize_filters(model, layer_name, img_height, img_width):
    # this is the placeholder for the input images
    input_img = model.input
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    kept_filters = []
    for filter_index in range(200):
        # we only scan through the first 200 filters,
        # but there are actually 512 of them
        print('Processing filter %d' % filter_index)
        start_time = time.time()

        # we build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        layer_output = layer_dict[layer_name].output
        if K.image_data_format() == 'channels_first':
            loss = K.mean(layer_output[:, filter_index, :, :])
        else:
            loss = K.mean(layer_output[:, :, :, filter_index])

        # we compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        grads = normalize(grads)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # step size for gradient ascent
        step = 1.

        # we start from a gray image with some random noise
        if K.image_data_format() == 'channels_first':
            input_img_data = np.random.random((1, 3, img_width, img_height))
        else:
            input_img_data = np.random.random((1, img_width, img_height, 3))
        input_img_data = (input_img_data - 0.5) * 20 + 128

        # we run gradient ascent for 20 steps
        for i in range(20):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step

            print('Current loss value:', loss_value)
            if loss_value <= 0.:
                # some filters get stuck to 0, we can skip them
                break

        # decode the resulting input image
        if loss_value > 0:
            img = deprocess_image(input_img_data[0])
            kept_filters.append((img, loss_value))
        end_time = time.time()
        print('Filter %d processed in %ds' % (filter_index, end_time - start_time))
    return kept_filters


def load_model_from_file(filename):
    model = load_model(filename)
    model.summary()
    return model


def create_filter_visualization(img_height, img_width, kept_filters, n, layer_name):
    # the filters that have the highest loss are assumed to be better-looking.
    # we will only keep the top 64 filters.
    kept_filters.sort(key=lambda x: x[1], reverse=True)
    kept_filters = kept_filters[:n * n]
    # build a black picture with enough space for
    # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
    margin = 5
    width = n * img_width + (n - 1) * margin
    height = n * img_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, 3))
    # fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            img, loss = kept_filters[i * n + j]
            stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
            (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

    # save the result to disk
    save_img(layer_name + '_stitched_filters_%dx%d.png' % (n, n), stitched_filters)


def main(argv):
    parser = argparse.ArgumentParser(description='Visualizes Keras neural network models')
    parser.add_argument("filename", type=str, help="The model filename")
    parser.add_argument("--width", type=int, help="The width of the generated pictures for each filter", default=128)
    parser.add_argument("--height", type=int, help="The height of the generated pictures for each filter", default=128)
    parser.add_argument("--number_of_filters", type=int, help="The number of filter visualization to keep (n*n)", default=8)

    args = parser.parse_args()
    model = load_model_from_file(args.filename)
    for layer in model.layers:
        # we will stitch the best filters on a number_of_filters x number_of_filters grid.
        kept_filters = visualize_filters(model, layer.name, args.height, args.width)
        create_filter_visualization(args.height, args.width, kept_filters, args.number_of_filters, layer.name)


if __name__ == "__main__":
    main(sys.argv)