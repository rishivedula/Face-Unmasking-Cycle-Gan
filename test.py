import argparse
from keras.models import load_model
from matplotlib import pyplot

def parse_arguments():
    parser = argparse.ArgumentParser()
	  parser.add_argument('--load_test_dataset', '-', type=str, help='Loading the dataset with masked covered faces')
    parser.add_argument('--load_test_generator_checkpoint', '-g', type=str, help='Loading the generator model')

    args = parser.parse_args()
    return args

def load_real_samples(filename):

	data = np.load(filename)

	X1, X2 = data['arr_0'], data['arr_1']

	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]
 
def plot_images(src_img, gen_img, tar_img):
  images = np.vstack((src_img, gen_img, tar_img))

  images = (images + 1) / 2.0
  titles = ['Source', 'Generated', 'Expected']

  for i in range(len(images)):

    pyplot.plot()

    pyplot.axis('off')

    pyplot.imshow(images[i])

    pyplot.title(titles[i])
    filename1 = titles[i]+'.png'
    pyplot.savefig(filename1)
  pyplot.show()
 

def main(args):

    [X1, X2] = load_real_samples(args.load_test_dataset)
    print('Loaded', X1.shape, X2.shape)

    model = load_model(args.load_test_generator_checkpoint)

    ix = np.random.randint(0, len(X1), 1)
    src_image, tar_image = X1[ix], X2[ix]

    gen_image = model.predict(src_image)

    plot_images(src_image, gen_image, tar_image)
    filename1 = 'image.png'
    pyplot.savefig(filename1)
    pyplot.close()



if __name__ == '__main__':
    args = parse_arguments()
    save_config(args)
    main(args)