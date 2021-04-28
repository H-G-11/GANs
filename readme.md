# Generative Adversarial Networks

## What is this project about ?

 Generative Adversarial Networks (GANs) are neural network architectures trained to produce a generative model faking real data. 

In this project, I implement several GANs architecture:
- Deep Convolutional GANs (DC-GAN)
- Wassertstein GAN (WGAN)
- Optimal Transport GAN (OT-GAN)
- Cycle-GAN

and tested it on MNIST. I present a summary of all my readings on GANs in the joined PDF with the images generated, and also give a quick overview of it in the next section.

## More detailed description

The idea of GANs goes back to (Goodfellow et al. 2014), and consists in the training of two neural networks at the same time in order to generate data (mostly images) according to the distribution of a real dataset (for instance, MNIST). 

GANs consist of a generator, _g_, which has to capture the data distribution, and a discriminator (or critic) _d_, which must estimate the probability that the data at hand comes from the training dataset (real images) or has been simulated by the generator.

In Deep Convolutional GANs (Radford, Metz, 2016), the main point concerns the type of architectures used for _g_ and _d_. In fact, the authors advise taking Deep Convolutional Networks with a few tricks that they describe in their papers, and that I recall in the joined PDF.

In the article _Wasserstein GAN_ (Arjovsky et al. 2017), the authors focus on the various ways to define a distance or divergence on distributions. They provide a comprehensive theoretical analysis of how the Earth Mover (EM) distance behaves in comparison to popular probability distances and divergences. They also define a form of GAN called Wasserstein-GAN that minimizes a reasonable and efficient approximation of the EM distance. Finally They empirically show that Wasserstein-GANs cure the main training problem of GANs : the training instability.

In Optimal Transport GANs (Salimans et al., 2018), this idea goes further and another loss function inspired from Optimal Transport is used. The Sinkhorn algorithm (Cuturi, 2013) will also be useful to approximate the distance at hand. Even if the models are longer to train, it still provides some very good results.

In Cycle-GAN, the goal is different. In fact, (Zhu et al., 2020) adress the problem of Image-to-Image translation through the use of two GANs trained at the same time. They introduce new losses assuring a cycle consistency, and show very good results.

## How to test it ?

You can find my code in [this colab notebook](https://colab.research.google.com/drive/1qCF3gWdJQcGuFJFmwK8UEwbIer75FeXW?usp=sharing)