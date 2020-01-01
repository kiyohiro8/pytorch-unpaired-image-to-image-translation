# Comparing Image-to-Image translation algorithms

## CycleGAN

### horse ⇔ zebra
![](https://github.com/kiyohiro8/pytorch-unpaired-image-to-image-translation/blob/master/sample/horse2zebra_cyclegan_epoch_200.png)

Horse → zebra works, but zebra → horse is difficult.

![](https://github.com/kiyohiro8/pytorch-unpaired-image-to-image-translation/blob/master/sample/horse2zebra_cyclegan_epoch_200_difficult.png)

Because CycleGAN mainly distinguishes horse objects by color, it converts earth (top), brick (middle) and human skin (bottom) into a zebra pattern.

### cheese cake ⇔ chocolate cake
![](https://github.com/kiyohiro8/pytorch-unpaired-image-to-image-translation/blob/master/sample/cheesecake_chocolatecake_epoch100.png)
