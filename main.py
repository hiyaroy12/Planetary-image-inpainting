from __future__ import print_function, division
import os
from keras.datasets import cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K
from read_cut_img import detect_lines, add_lines
from train_keras import get_unet
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tqdm import tqdm
import math

def get_psnr(img1, img2):

    img1 = img1.astype('float32')
    img2 = img2.astype('float32')
    mse = np.mean((img1 - img2) ** 2)
    PIXEL_MAX = 1.0
    psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return(psnr)

#get direction of the mask
def get_direction(mask):
    M = cv2.moments(mask)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    w, h = mask.shape
    cX = cX - w//2
    cY = -(cY - h//2)
    if abs(cX)>abs(cY):
        if cX > 0:
            return 0
        else:
            return 1
    else:
        if cY > 0:
            return 2
        else:
            return 3
    # 0-> right, 1-> left, 2-> up, 3-> down
    
#get mirror image
def get_mirror_img(img, dirn):
    # 0-> right, 1-> left, 2-> up, 3-> down
    if dirn==0:
        rev_img = img.copy()
        rev_img = rev_img[:,::-1]
        new_img = np.concatenate((img, rev_img), axis=1)
    elif dirn==1:
        rev_img = img.copy()
        rev_img = rev_img[:, ::-1]
        new_img = np.concatenate(( rev_img, img), axis=1)
    elif dirn==2:
        rev_img = img.copy()
        rev_img = rev_img[::-1, :]
        new_img = np.concatenate((rev_img, img), axis=0)
        new_img = new_img.T
    elif dirn==3:
        rev_img = img.copy()
        rev_img = rev_img[::-1, :]
        new_img = np.concatenate((img, rev_img), axis=0)
        new_img = new_img.T
    
    return new_img
    
class ContextEncoder():
    def __init__(self, mirror=True):
        self.img_rows = 256
        self.img_cols = 256 * (2 if mirror else 1)
        self.mask_height = 256
        self.mask_width = 256 * (2 if mirror else 1)
        self.channels = 1
        self.num_classes = 2
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.missing_shape = (self.img_rows, self.img_cols, self.channels)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates the missing part of the image
        masked_img = Input(shape=self.img_shape)
        gen_missing = self.generator(masked_img)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines if it is generated or if it is a real image
        valid = self.discriminator(gen_missing)

        # Trains generator to fool discriminator
        self.combined = Model(masked_img , [gen_missing, valid])
        self.combined.compile(loss=['mse', 'binary_crossentropy'],
            loss_weights=[0.999, 0.001],
            optimizer=optimizer)

    def build_generator(self, unet =True):

        if unet:
            return get_unet((self.img_rows, self.img_cols, 1))
        else:
            model = Sequential()

            # Encoder
            model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
            model.add(LeakyReLU(alpha=0.2))
            #model.add(BatchNormalization(momentum=0.8))
            model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
            model.add(LeakyReLU(alpha=0.2))
            #model.add(BatchNormalization(momentum=0.8))
            model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
            model.add(LeakyReLU(alpha=0.2))
            #model.add(BatchNormalization(momentum=0.8))

            model.add(Conv2D(512, kernel_size=1, strides=2, padding="same"))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.5))

            # Decoder
            model.add(UpSampling2D())
            model.add(Conv2D(128, kernel_size=3, padding="same"))
            model.add(Activation('relu'))
            #model.add(BatchNormalization(momentum=0.8))
            model.add(UpSampling2D())
            model.add(Conv2D(64, kernel_size=3, padding="same"))
            model.add(Activation('relu'))
            #model.add(BatchNormalization(momentum=0.8))
            model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
            model.add(Activation('sigmoid'))

            model.summary()

            masked_img = Input(shape=self.img_shape)
            gen_missing = model(masked_img)

            return Model(masked_img, gen_missing)

    def build_discriminator(self):

        model = Sequential()
        model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=self.missing_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(256, kernel_size=3, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(BatchNormalization(momentum=0.8))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.missing_shape)
        # mask = Input(shape=self.missing_shape)
        # masked_input = img * mask
        validity = model(img)
        # op = validity
        # return Model(inputs=[img, mask], outputs=op)
        return Model(inputs=img, outputs=validity)

    def mask_randomly(self, list_imgs_lines, list_imgs_no_lines, batchsize, mirror=True):
        rand_idx_lines = np.random.choice(len(list_imgs_lines), batchsize, replace=False).tolist()
        rand_idx_no_lines = np.random.choice(len(list_imgs_no_lines), batchsize, replace=False).tolist()

        missing_parts = np.empty((batchsize, self.img_rows, self.img_cols, self.channels))
        masked_imgs = np.empty((batchsize, self.img_rows, self.img_cols, self.channels))
        imgs = np.empty((batchsize, self.img_rows, self.img_cols, self.channels))
        masks = np.empty((batchsize, self.img_rows, self.img_cols, self.channels))
        i=0
        for lineidx, nolineidx in tqdm(zip(rand_idx_lines, rand_idx_no_lines)):
            img_line = cv2.imread(list_imgs_lines[lineidx], 0)
            img_line = cv2.resize(img_line, (256, 256))
            img_no_line = cv2.imread(list_imgs_no_lines[nolineidx], 0)
            img_no_line = cv2.resize(img_no_line, (256, 256))
            thres_img, binary_i = detect_lines(img_line, thres=10)

            img_superimposed_lines = add_lines(img_no_line.copy(), thres_img)
            if mirror:
                mask = cv2.resize(thres_img, (256, 256)).copy()
                dirn = get_direction(mask)
                thres_img = get_mirror_img(thres_img, dirn)
                img_superimposed_lines = get_mirror_img(img_superimposed_lines, dirn)
                img_no_line = get_mirror_img(img_no_line, dirn)
            img_missing = add_lines(img_no_line.copy(), 1 - thres_img)
            
            if mirror:
                missing_parts[i, :,:, 0] = cv2.resize(img_missing, (256*2, 256)).copy()/255.
                masked_imgs[i, :,:, 0] = cv2.resize(img_superimposed_lines, (256*2, 256)).copy()/255.
                imgs[i, :,:, 0] = cv2.resize(img_no_line, (256*2, 256)).copy()/255.
                masks[i, :,:, 0] = cv2.resize(thres_img, (256*2, 256)).copy()
            else:
                missing_parts[i, :,:, 0] = cv2.resize(img_missing, (256, 256)).copy()/255.
                masked_imgs[i, :,:, 0] = cv2.resize(img_superimposed_lines, (256, 256)).copy()/255.
                imgs[i, :,:, 0] = cv2.resize(img_no_line, (256, 256)).copy()/255.
                masks[i, :,:, 0] = cv2.resize(thres_img, (256, 256)).copy()
            i+=1
        return masked_imgs, missing_parts, imgs, masks

    def train(self, list_imgs_no_lines, list_imgs_lines, epochs, batch_size=128, sample_interval=50):
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            masked_imgs, missing_parts, imgs, masks = self.mask_randomly(list_imgs_lines,
                                                                         list_imgs_no_lines, batch_size)
            d_loss_list = [[], []]
            g_loss_list = [[], []]
            for idx in range(3):
                # Generate a batch of new images
                if idx==0:
                    gen_missing = self.generator.predict(masked_imgs)
                    # Train the discriminator
                    d_loss_real = self.discriminator.train_on_batch(missing_parts, valid)
                    d_loss_fake = self.discriminator.train_on_batch(gen_missing * masks, fake)
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                # ---------------------
                #  Train Generator
                # ---------------------
                g_loss = self.combined.train_on_batch(masked_imgs, [missing_parts, valid])
                for kk in range(2):
                    d_loss_list[kk].append(d_loss[kk])
                    g_loss_list[kk].append(g_loss[kk])

            # Plot the progress
            print("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" %
                   (epoch, np.mean(d_loss_list[0]), 100*np.mean(d_loss_list[1]),
                    np.mean(g_loss_list[0]), np.mean(g_loss_list[1])))

            # # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.generator.save("generator_{}.h5f".format(cluster_num))
                self.discriminator.save("discriminator_{}.h5f".format(cluster_num))
                self.sample_images(epoch, batch_size)

    def sample_images(self, epoch, batch_size):
        r, c = 3, 6
        psnr_imgs_masked_imgs = []
        psnr_imgs_filled_in = []

        masked_imgs, missing_parts, imgs, masks = self.mask_randomly(list_imgs_lines,
                                                                     list_imgs_no_lines, batch_size)
        gen_missing = self.generator.predict(masked_imgs)

        filled_in = masked_imgs + masks * gen_missing
        fig, axs = plt.subplots(r, c)
        for i in range(c):
            axs[0,i].imshow(imgs[i, :,:, 0], cmap='gray')
            axs[0,i].axis('off')
            axs[1,i].imshow(masked_imgs[i, :, :, 0], cmap='gray')
            axs[1,i].axis('off')
            psnr1 = get_psnr(masked_imgs[i, :, :, 0], imgs[i, :, :, 0])
            axs[1,i].set_title('{:0.2f}'.format(psnr1))
            psnr_imgs_masked_imgs.append(psnr1)

            axs[2,i].imshow(filled_in[i, :, :, 0], cmap='gray')
            axs[2,i].axis('off')
            psnr2 = get_psnr(filled_in[i, :, :, 0], imgs[i, :, :, 0])
            axs[2, i].set_title('{:0.2f}'.format(psnr2))
            psnr_imgs_filled_in.append(psnr2)
        fig.savefig("result/new_context_encoder/images_cluster_%d/%d.png" % (cluster_num, epoch))
        plt.close()

        mean_psnr1 = np.mean(psnr_imgs_masked_imgs)
        mean_psnr2 = np.mean(psnr_imgs_filled_in)
        print(mean_psnr1)
        print(mean_psnr2)

    def save_model(self):
        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "generator")
        save(self.discriminator, "discriminator")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--cluster_num', type=int, default=None)
    args = parser.parse_args()
    cluster_num = args.cluster_num

    no_lines_file = 'data/clean_files_cluster_{}.txt'.format(cluster_num)
    lines_file = 'data/files_cluster_{}.txt'.format(cluster_num)

    list_imgs_no_lines = [line.rstrip('\n') for line in open(no_lines_file)]
    list_imgs_lines = [line.rstrip('\n') for line in open(lines_file)]

    context_encoder = ContextEncoder()
    context_encoder.train(list_imgs_no_lines, list_imgs_lines, epochs=30000, batch_size=8, sample_interval=50)
