import paddle
from paddle.nn import Linear
import paddle.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt

paddle.vision.set_image_backend("cv2")

train_dataset = paddle.vision.datasets.MNIST(mode="train")

train_data0 = np.array(train_dataset[0][0])
train_label0 = np.array(train_dataset[0][1])

# plt.figure("Image")
# plt.figure(figsize=(2, 2))
# plt.imshow(train_data0, cmap=plt.cm.binary)
# plt.axis("on")
# plt.title("image")
# plt.show()

print(train_data0.shape)
print(train_label0.shape, train_label0)


class MNIST(paddle.nn.Layer):
    def __init__(self):
        super(MNIST, self).__init__()

        self.fc = paddle.nn.Linear(in_features=784, out_features=1)

    def forward(self, inputs):
        return self.fc(inputs)


def norm_img(img):
    assert len(img.shape) == 3

    batch_size, img_h, img_w = img.shape[0], img.shape[1], img.shape[2]

    img = img / 255

    img = paddle.reshape(img, [batch_size, img_h * img_w])

    return img


model = MNIST()


def train(model):

    model.train()

    train_loader = paddle.io.DataLoader(train_dataset, batch_size=16, shuffle=True)

    opt = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())

    EPOCH_NUM = 10

    for epoch in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            images = norm_img(data[0]).astype("float32")
            labels = data[1].astype("float32")

            predicts = model(images)

            loss = F.square_error_cost(predicts, labels)
            avg_loss = paddle.mean(loss)

            if batch_id % 100 == 0:
                print(
                    "epoch: {}, batch: {} loss: {}".format(
                        epoch, batch_id, avg_loss.numpy()
                    )
                )

            avg_loss.backward()
            opt.step()
            opt.clear_grad()


train(model)

paddle.save(model.state_dict(), "foo.txt")
