import argparse
import numpy as np
from numpy.random import randint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model,load_model,Input
from keras.layers import Conv2D,Conv2DTranspose,LeakyReLU,Activation,Concatenate,Dropout,BatchNormalization
from keras.layers import LeakyReLU
from matplotlib import pyplot
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization


def parse_arguments():
    parser = argparse.ArgumentParser()
	parser.add_argument('--load_mask_dataset', '-m', type=str, help='Loading the dataset with masked covered faces')
	parser.add_argument('--load_face_dataset', '-f', type=str, help='Loading the dataset with faces without face masks')
	parser.add_argument('--continues', '-c', type=bool, default=False, help='To continue training from a given checkpoint')
    parser.add_argument('--load_generator_checkpoint', '-g', type=str, help='Loading the generator model')
	parser.add_argument('--load_discriminator_checkpoint', '-d', type=str, help='Loading the discriminator model')

    args = parser.parse_args()
    return args


# Discriminator
def define_discriminator(image_shape):

	init = RandomNormal(stddev=0.02)
	in_src_image = Input(shape=image_shape)
	in_target_image = Input(shape=image_shape)
	merged = Concatenate()([in_src_image, in_target_image])
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	patch_out = Activation('sigmoid')(d)
	model = Model([in_src_image, in_target_image], patch_out)
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
	return model

# Encoder
def define_encoder_block(layer_in, n_filters, batchnorm=True):

	init = RandomNormal(stddev=0.02)
	g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	g = LeakyReLU(alpha=0.2)(g)
	return g

# Decoder
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
	init = RandomNormal(stddev=0.02)
	g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	g = BatchNormalization()(g, training=True)
	if dropout:
		g = Dropout(0.5)(g, training=True)
	g = Concatenate()([g, skip_in])
	g = Activation('relu')(g)
	return g

# Generator
def define_generator(image_shape=(256,256,3)):
	init = RandomNormal(stddev=0.02)
	in_image = Input(shape=image_shape)
	e1 = define_encoder_block(in_image, 64, batchnorm=False)
	e2 = define_encoder_block(e1, 128)
	e3 = define_encoder_block(e2, 256)
	e4 = define_encoder_block(e3, 512)
	e5 = define_encoder_block(e4, 512)
	e6 = define_encoder_block(e5, 512)
	e7 = define_encoder_block(e6, 512)
	
	b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
	b = Activation('relu')(b)

	d1 = decoder_block(b, e7, 512)
	d2 = decoder_block(d1, e6, 512)
	d3 = decoder_block(d2, e5, 512)
	d4 = decoder_block(d3, e4, 512, dropout=False)
	d5 = decoder_block(d4, e3, 256, dropout=False)
	d6 = decoder_block(d5, e2, 128, dropout=False)
	d7 = decoder_block(d6, e1, 64, dropout=False)

	g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
	out_image = Activation('tanh')(g)

	model = Model(in_image, out_image)
	return model


def define_gan(g_model, d_model, image_shape):

	for layer in d_model.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False

	in_src = Input(shape=image_shape)
	gen_out = g_model(in_src)
	dis_out = d_model([in_src, gen_out])
	model = Model(in_src, [dis_out, gen_out])
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
	return model

def generate_real_samples(n_samples, patch_shape):

	ix = np.random.randint(0, X1.shape[0], n_samples)
	x1, x2 = X1[ix], X2[ix]
	y = np.ones((n_samples, patch_shape, patch_shape, 1))
	return [x1, x2], y

def generate_fake_samples(g_model, samples, patch_shape):
	X = g_model.predict(samples)
	y = np.zeros((len(X), patch_shape, patch_shape, 1))
	return X, y

def summarize_performance(step, g_model, d_model, n_samples=3):
	[X_realA, X_realB], _ = generate_real_samples(n_samples, 1)
	X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
	X_realA = (X_realA + 1) / 2.0
	X_realB = (X_realB + 1) / 2.0
	X_fakeB = (X_fakeB + 1) / 2.0

	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realA[i])

	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples + i)
		pyplot.axis('off')
		pyplot.imshow(X_fakeB[i])

	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realB[i])

	filename1 = '/content/gdrive/MyDrive/plot_celeba_fin_%06d.png' % (step+1)
	pyplot.savefig(filename1)
	pyplot.close()

	filename2 = '/content/gdrive/MyDrive/gen_celeba_fin_model_%06d.h5' % (step+1)
	g_model.save(filename2)
	filename3 = '/content/gdrive/MyDrive/dis_celeba_fin_model_%06d.h5' % (step+1)
	d_model.save(filename3)
	print('>Saved: %s, %s and %s' % (filename1, filename2,filename3))


def train(d_model, g_model, gan_model, n_epochs=100, n_batch=8):

	n_patch = d_model.output_shape[1]
	bat_per_epo = int(len(X1) / n_batch)

	n_steps = bat_per_epo * n_epochs

	for i in range(296200,n_steps):

		[X_realA, X_realB], y_real = generate_real_samples(n_batch, n_patch)
		X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
		d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
		d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
		g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
		print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))

		if (i+1) % (bat_per_epo * 10) == 0:
			summarize_performance(i, g_model,d_model)


def main(args):

	# Load the datasets
	X1 = load(args.load_mask_dataset)['arr_0']
	X2 = load(args.load_face_dataset)['arr_0']
	print('Loaded', X1.shape, X2.shape)

	# define input shape based on the loaded dataset
	image_shape = X1.shape[1:]
	# define the models

	if arg.continues:
	
		d_model = load_model(args.discriminator, compile = False)
		g_model = load_model(args.generator, compile = False)
	else:
		d_model = define_discriminator(image_shape)
		g_model = define_generator(image_shape)
	
	# define the composite model
	gan_model = define_gan(g_model, d_model, image_shape)
	# train model
	train(d_model, g_model, gan_model)


if __name__ == '__main__':
    args = parse_arguments()
    save_config(args)
    main(args)