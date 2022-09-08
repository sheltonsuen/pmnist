import matplotlib.pyplot as plt
import numpy as np
import paddle
from PIL import Image

img_path = "./example.jpg"
im = Image.open(img_path)

# plt.imshow(im)
# plt.show()

im = im.convert("L")
print(np.array(im).shape)

im = im.resize((28, 28), Image.ANTIALIAS)

plt.imshow(im)
plt.show()


def load_img(path):
    im = Image.open(path).convert("L")
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).reshape(1, -1).astype(np.float32)
    im = im / 255
    return im


img = load_img(img_path)


class MNIST(paddle.nn.Layer):
    def __init__(self):
        super(MNIST, self).__init__()

        self.fc = paddle.nn.Linear(in_features=784, out_features=1)

    def forward(self, inputs):
        return self.fc(inputs)


model = MNIST()

param_dict = paddle.load("./mnist.pdparams")
model.load_dict(param_dict)

model.eval()

result = model(paddle.to_tensor(img))

print("result: {}, {}".format(result, result.numpy().astype("int32")))
