The Transformer architecture has been the foundational cornerstone for the recent AI revolution, serving as the backbone of Large Language Models, which have demonstrated impressive language understanding and reasoning capabilities. When pre-trained on large amounts of data, Transformers have also shown to be highly effective in image classification via the advent of the Vision Transformer. However, it still lags in vision application performance compared to Convolutional Neural Networks (CNNs), which offer translational invariance, whereas Transformers lack inductive bias. Further, the Transformer relies on the attention mechanism, which despite increasing the receptive field, makes it computationally inefficient due to its quadratic time complexity. In this paper, we enhance the Transformer architecture, focusing on its above two shortcomings. We propose two efficient Vision Transformer architectures that significantly reduce the computational complexity without sacrificing classification performance. Our first enhanced architecture is the EEViT-PAR, which combines features from two recently proposed designs of PerceiverAR and CaiT. This enhancement leads to our second architecture, EEViT-IP, which provides implicit windowing capabilities akin to the SWIN Transformer and implicitly improves the inductive bias, while being extremely memory and computationally efficient. We perform detailed experiments on multiple image datasets to show the effectiveness of our architectures. Our best performing EEViT outperforms existing SOTA ViT models in terms of execution efficiency and matches or surpasses their classification accuracy on different benchmarks.

EEViT: main file

Requirements.txt: packages required for Python environment

![Table 3](https://github.com/user-attachments/assets/e44ce37b-11a4-43dd-a814-1a6eb1fddc33)

![Table 4](https://github.com/user-attachments/assets/1313409d-3203-4760-b0d0-d3d9ba6969c7)

![Table 5](https://github.com/user-attachments/assets/d9fc9edd-e34d-4b01-9bc2-39b51ea9c071)

![Figure 11](https://github.com/user-attachments/assets/774e5cc0-2be0-4439-9d02-8151fdc45285)
