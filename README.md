# Comparing Image-to-Image translation algorithms

## CycleGAN

### horse ⇔ zebra
![](https://github.com/kiyohiro8/pytorch-unpaired-image-to-image-translation/blob/master/sample/horse2zebra_cyclegan_epoch_200.png)

Horse → zebra works, but zebra → horse is relatively difficult.

difficult samples
![](https://github.com/kiyohiro8/pytorch-unpaired-image-to-image-translation/blob/master/sample/horse2zebra_cyclegan_epoch_200_difficult.png)

Because CycleGAN mainly distinguishes horse objects by color, it converts earth (top), brick (middle) and human skin (bottom) into a zebra pattern.

### apple ⇔ orange
![](https://github.com/kiyohiro8/pytorch-unpaired-image-to-image-translation/blob/master/sample/apple2orange_cyclegan_epoch_200.png)

difficult samples
![](https://github.com/kiyohiro8/pytorch-unpaired-image-to-image-translation/blob/master/sample/apple2orange_cyclegan_epoch_200_difficult.png)

CycleGAN does a good job of converting colors and textures, but not shapes (top). Also, this model doesn't seem to care about the differences between the contents of apples and oranges (bottom).

### cheese cake ⇔ chocolate cake
![](https://github.com/kiyohiro8/pytorch-unpaired-image-to-image-translation/blob/master/sample/cheesecake_chocolatecake_epoch100.png)
