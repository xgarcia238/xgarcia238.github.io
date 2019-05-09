#  Projects

## 8-bit VAE: A latent variable model for NES music

I delved into generating multi-instrumental music using the synthesizer from the Nintendo Entertainment System (NES) through the `nesmdb` library introduced in the work of [Donahue et. al](https://arxiv.org/abs/1806.04278). A clever data representation allows us to formulate the problem as a sequence-to-sequence task. In this project, I built a variational autoencoder to create short NES snippets of music by interpolating in the latent space. You can read more details in my blog [post]({% post_url /blog/2018-03-18-8bitvae %}) or check out the GitHub [repo.](https://github.com/xgarcia238/8bit-VAE)