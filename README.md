The Transformer architecture has been the foundational cornerstone for the recent AI revolution, serving as the backbone of Large Language Models, which have demonstrated impressive language understanding and reasoning capabilities. When pre-trained on large amounts of data, Transformers have also shown to be highly effective in image classification via the advent of the Vision Transformer. However, it still lags in vision application performance compared to Convolutional Neural Networks (CNNs), which offer translational invariance, whereas Transformers lack inductive bias. Further, the Transformer relies on the attention mechanism, which despite increasing the receptive field, makes it computationally inefficient due to its quadratic time complexity. In this paper, we enhance the Transformer architecture, focusing on its above two shortcomings. We propose two efficient Vision Transformer architectures that significantly reduce the computational complexity without sacrificing classification performance. Our first enhanced architecture is the EEViT-PAR, which combines features from two recently proposed designs of PerceiverAR and CaiT. This enhancement leads to our second architecture, EEViT-IP, which provides implicit windowing capabilities akin to the SWIN Transformer and implicitly improves the inductive bias, while being extremely memory and computationally efficient. We perform detailed experiments on multiple image datasets to show the effectiveness of our architectures. Our best performing EEViT outperforms existing SOTA ViT models in terms of execution efficiency and matches or surpasses their classification accuracy on different benchmarks.

EEViT: main file

Requirements.txt: packages required for Python environment

![fig1](https://github.com/user-attachments/assets/ed0a22c6-fc56-4b41-9d67-f7f968dae6aa)
![fig2](https://github.com/user-attachments/assets/c865bfde-736b-4048-b466-35e792c96118)
![fig3](https://github.com/user-attachments/assets/50ba222a-1356-4bd9-92ba-6280ea1b7318)
![fig4](https://github.com/user-attachments/assets/ab9ed84b-88c3-41f3-9fe9-9f07448c85c0)
![fig5](https://github.com/user-attachments/assets/e3a5525c-aa64-427f-a2b9-622f05b2cd3a)
![fig6](https://github.com/user-attachments/assets/5535f212-5428-4457-bc34-e225bcd6628c)
![fig7](https://github.com/user-attachments/assets/983cf2d7-5789-4035-bef1-90a94e223ccd)
![fig8](https://github.com/user-attachments/assets/d9d192ca-29f4-4083-a859-ffbca8aa8f38)
![fig9](https://github.com/user-attachments/assets/cb19129a-2e5c-47c0-bb3e-1eb5e911b78a)
![fig10](https://github.com/user-attachments/assets/a1e47d06-fda9-4aef-84da-bf445bc89aa4)
![fig11](https://github.com/user-attachments/assets/ba095c4c-dff5-48a1-ac6a-cc4e76c2fc2c)
![fig12](https://github.com/user-attachments/assets/3335a805-120d-4874-960e-91a6d115b527)
![fig13](https://github.com/user-attachments/assets/99920dc7-79ba-45cc-b078-db3f989b7dec)
