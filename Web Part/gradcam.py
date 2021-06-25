import tensorflow as tf
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt


class Gradcam:
    def __init__(self):
        MODEL_PATH = '/home/dsail/codeeplearning/models/oct_model.h5'
        self.model = tf.keras.models.load_model(MODEL_PATH)
    
    def get_gradcam(self, image_url, save_url):
        array = get_img_array(img_path=image_url, size=(256, 256, 3))
        heatmap = make_gradcam_heatmap(array, self.model)
        save_and_display_gradcam(image_url, heatmap, save_url)


def get_img_array(img_path, size):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    array = tf.keras.preprocessing.image.img_to_array(img)/255

    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer("block5_pool").output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.2):
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    superimposed_img.save(cam_path)


if __name__ == "__main__":
    gc = Gradcam() # 객체로 gradcam선언
    gc.get_gradcam("/home/dsail/kojunseo/codeep/OCT2017 /test/DME/DME-30521-2.jpeg", "./gradcam1.jpg")
    # get_gradcam부분에 파라미터로 (이미지 위치, 생성하고 싶은 gradcam이름) 넣으면 된다.
    # gc.get_gradcam("./../kojunseo/codeep/chest_xray/test/NORMAL/NORMAL2-IM-0374-0001.jpeg", "./gradcam2.jpg")