---
layout: post
mathjax: true
comments: true
title:  "Welcome to Jekyll!"
date:   2016-02-12 17:50:00
categories: main
---


The purpose of these notes is to explain a few of the confusing parts of the paper ``Attention is all you need". We begin by discussing the simplest setup and expanding, drawing analogies between the Transformer model and our intermediate models.

#eq2Seq Framework

We begin by establishing the setting. We have an input sequence $\textbf{x} = (x_1,... ,x_T)$, where each $x_i \in \mathbb{R}^{d_{\text{input}}}$ as well as a target sequence $\textbf{y} = (y_1,..., y_{T'})$ where we'll think of the $y_i$ as being integer-valued, representing the index of some word in some dictionary of size $d_{\text{output}}$.

 For simplicity, consider the simplest RNN encoder-decoder scheme where both the encoder and the decoder are vanilla RNN. The encoder consumes the sequence $\textbf{x}$, produces a sequence of hidden states $\textbf{h} = (h_1, ... ,h_T)$ where $h_i \in \mathbb{R}^{d_{\text{hidden}}}$. We then produce a new hidden state $q_0 = q_0(\textbf{h})$, depending on the previous the hidden states. Traditionally, $q_0 = h_T$, but we don't need to make this assumption. We use this as the initial hidden state for another RNN, termed the decoder, which will autoregressively produce $\textbf{y}$. More precisely, the input to the decoder will be the sequence given by $(\langle \text{SOS} \rangle, y_1,... y_{T'-1})$, where the $\langle \text{SOS} \rangle$ token stands for ``start of sentence" with some predetermined index in our dictionary. With this input, the decoder then produces its own set of hidden states $\textbf{Q} = (q_1, ..., q_{T'})$, and we use this to compute the logits for our predictions. Explicitly, we have the formula: $$\Pa(y_i = \cdot) = \text{softmax}(F(q_i))$$ where $F : \mathbb{R}^{d_{\text{hidden}}} \rightarrow \mathbb{R}^{d_{\text{output}}}$ is usually a simply fully-connected layer i.e. an affine transformation. Notice that by our choice of input for the decoder, the logits of $y_i$ only depend on $\textbf{h}$ and $y_1, ... , y_{i-1}$ i.e. we never peek into the future.

You'll find this post in your `_posts` directory - edit this post and re-build (or run with the `-w` switch) to see your changes!
To add new posts, simply add a file in the `_posts` directory that follows the convention: YYYY-MM-DD-name-of-post.ext.

Jekyll also offers powerful support for code snippets:

{% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
{% endhighlight %}

Check out the [Jekyll docs][jekyll] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll's GitHub repo][jekyll-gh].

[jekyll-gh]: https://github.com/mojombo/jekyll
[jekyll]:    http://jekyllrb.com
