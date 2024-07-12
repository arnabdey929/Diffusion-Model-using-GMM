# Diffusion-Model-using-GMM
Learning a one-dimensional arbitrary distribution using Diffusion Models

In this project we Learn about a Diffusion Model using a GMM(Gaussian Mixture Model).

We know that GMMs can approximate any probability distribution to an arbitrary precision.
That is, given that GMMs are universal density estimators, if we can show that a Diffusion Model can learn the
distribution of any GMM, then it can be proven that Diffusion Models can learn any distribution.

For this example, I have used a 1-Dimensional feature for visual verification, and a very simple MLP with 4
fully connected layers with GELU activation at each layer to learn the following distribution : 

X ~ 0.3N(-5, 1) + 0.5N(1, 2) + 0.2N(3, 1)

I used 1000 data points sampled from the above distribution.

The diffusion process is exactly the same as given in the paper DDPM.

**Important Note**

While copying and running the code make sure that "torch.compile()" is available for your python version.
If not, just comment the line and the rest should work just fine.

------------------------------------------------------------------------------------------------------------------------------
If you want to load my pretrained model, change PATH to the actual path to myModel.pt
