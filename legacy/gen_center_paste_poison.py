from PIL import Image
import torchvision.transforms as transforms

resize = transforms.Resize((196, 196))

# 加载32x32图像
for i in range(0, 5000):
    big_image = Image.open('/home/cwh/Workspace/post-hoc-cbm-main/data/cifar100AdversarialDataset/class_boy/image_{}.png'.format(i))
    small_image = Image.open('/home/cwh/Workspace/post-hoc-cbm-main/data/cifar100AdversarialDataset/class_baby/image_{}.png'.format(i))
    small_image = resize(small_image)

    # 计算小图像的粘贴位置
    x_pos = int((224 - 96) / 2)
    y_pos = int((224 - 96) / 2)

    # 在大图像中央插入小图像
    big_image.paste(small_image, (x_pos, y_pos))

    big_image.save('/home/cwh/Workspace/post-hoc-cbm-main/data/cifar100AdversarialDataset/class_full_mask/class_full_mask/poison{}.jpg'.format(i))

