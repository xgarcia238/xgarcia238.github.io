---
layout: post
title: "Self-Attention in the Transformer"
author: Xavier Garcia
category: misc
mathjax_support: true
mathjax: 1
visible: 1
---

The purpose of these notes is to explain the self-attention mechanism in the paper "Attention is all you need". We begin by discussing the simplest setup and expanding, drawing analogies between the Transformer model and intermediate models.

# Seq2Seq Framework

We begin by establishing the setting. We have an input sequence $\textbf{x} = (x_1,... ,x_L)$, where each $x_i \in \mathbb{R}^{d_{\text{input}}}$ will represent some vector representation of a word as well as a target sequence $\textbf{y} = (y_1,..., y_{L'})$ where we'll think of the $y_i$ as being integer-valued, representing the index of some word in some dictionary of size $d_{\text{output}}$.

For simplicity, consider the simplest RNN encoder-decoder scheme where both the encoder and the decoder are vanilla RNN. The encoder consumes the sequence $\textbf{x}$ and produces a sequence of hidden states $\textbf{h} = (h_1, ... ,h_L)$ where $h_i \in \mathbb{R}^{d_{\text{hidden}}}$. These hidden states serve as representations for the words at each time, endowed with the context from the words that came before it. We then produce a new hidden state $q_0 = q_0(\textbf{h})$, depending on the previous the hidden states. Traditionally, $q_0 = h_L$, but we don't need to make this assumption. We use this as the initial hidden state for another RNN, termed the decoder, which will autoregressively produce $\textbf{y}$. More precisely, the input to the decoder will be the sequence given by $(\texttt{sos} , y_1,... y_{L'-1})$, where the $\texttt{sos}$ token stands for start of sentence with some predetermined index in our dictionary. With this input, the decoder then produces its own set of hidden states $\textbf{Q} = (q_1, ..., q_{L'})$, and we use this to compute the logits for our predictions. Explicitly, we have the formula: 

$$\mathbb{P}(y_i = \cdot) = \text{softmax}(F(q_i))$$ 

where $F : \mathbb{R}^{d_{\text{hidden}}} \rightarrow \mathbb{R}^{d_{\text{output}}}$ is usually a simply fully-connected layer i.e. an affine transformation. Notice that by our choice of input for the decoder, the logits of $y_i$ only depend on $\textbf{h}$ and $y_1, ... , y_{i-1}$ i.e. we never peek into the future. 


## Attention for RNNs
In this section, I'll explain first what attention is in the RNN case, and use that as a motivation for the self-attention used in the paper. 

One of the problems with the formulation stated above is that the entire essence of the input sequence $\textbf{x}$ must be captured within $q_0$. While this seems reasonable for a small sequence, this could quickly become out of hand for very long sequences, especially since all such vector representations must be the same dimension, independent of $L$. To make matters worse, it may be the case that in our decoding, we may not be interested in all parts of the input sequence during every prediction. For example, if we were translating from English to Spanish, it is reasonable to expect that to produce the first word in the translation, we only need the first word or two from the English sentence.  Unfortunately, translation is not as simple as pairing the words one by one, especially since the sentences may have different lengths. Nevertheless, the idea of focusing on certain key words from the source as one produces the next word in a translation seems useful. The idea of attention is to endow our decoder RNN with this mechanism directly. More explicitly, to compute each $y_j$, we first compute an $\textit{alignment score}$ $\alpha_{ij}$ between the decoder's hidden state $q_j$ and the encoder's hidden state $h_i$ for each $i$.  In particular, 
 
\begin{equation}
\alpha_{ij} := \text{softmax}( \langle h_i , q_j \rangle) := \frac{  e^{\langle h_i, q_j \rangle} }{\sum^{L}_{k=1} e^{\langle h_k , q_j \rangle}}
\end{equation}

We use these alignment scores to produce an attention vector 

$$A_j = \sum_{i=1}^L \alpha_{ij} h_i.$$

We should think that $y_j$ is using its key $q_j$ to query each of the previous $x_i$ through their keys $h_i$ by computing their inner product then normalizing the values through a softmax. For future convenience, we rewrite this equation in matrix form. We view the sequence of hidden states $\textbf{h}$ and $\textbf{Q} := (q_{j})$ and the alignment scores $\alpha_{ij}$ as matrices i.e. $\textbf{h} \in \mathbb{R}^{L \times d_{\text{hidden}}}$, $\textbf{Q} \in \mathbb{R}^{L' \times d_{\text{hidden}}}$ and $\alpha := (\alpha_{ij}) \in \mathbb{R}^{L \times L'}$, which are connected by the following equation:



$$\alpha = \text{softmax}\left(\textbf{h} Q^T \right).$$

In particular, $\alpha^T = \text{softmax} \left( Q\textbf{h}^T \right)$ where the softmax is now along the other axis and we can compute the attention vectors as follows:

$$\textbf{A} = \alpha^T \textbf{h} = \text{softmax} \left( Q \textbf{h}^T \right) \textbf{h}$$

where the $j$th column of $\textbf{A}$ is $A_j$ for all $j$. We call these values the \textit{attention values}, which are then passed through a fully-connected layer followed by a softmax to produce the probabilities for $\textbf{y}$. We can think of the $A_j$ as being analogous to the hidden states of the decoder in the RNN case. The main difference is that each $A_j$ can directly connect to the representation of each word in the input, as opposed to being limited to only using the last hidden state as a summary for the entire sequence.

## Self-Attention and Multiple Heads


Before defining self-attention, we first summarize the computations we performed in the previous subsections. To each element of the output, we assigned to it a query vector (the hidden states), which we used it to ``query" the words in the input by computing the inner product with the input's keys (the hidden states of the input), and after normalization, these became the alignment scores. We then used these alignment scores as weights to sum each element's contribution (henceforth called values), to obtain attention vector for that element of the output. We group these up into matrices $\textbf{Q} \in \mathbb{R}^{L' \times  d}$, $\textbf{K} \in \mathbb{R}^{L \times d}$, and $\textbf{V} \in \mathbb{R}^{L \times d}$ respectively, and use these to create attention values in an analogous way as before i.e. 
$$
\textbf{A} = \text{softmax}(\textbf{Q} \textbf{K}^T)\textbf{V}.
$$

As a sanity check, the reader can check that if we set $\textbf{K} = \textbf{V} = \textbf{h}$ and $d = d_{\text{hidden}}$, then this is exactly the definition of attention described in the previous subsection.

This construction makes it so that each entry of the ouput attends to all entries in the inputs at once to make a representation of the data specific to that context. We could also have output attend to itself as well as the input attend to itself. Rather than passing the input along an RNN, we could have each word attend to itself and the other input words in order to produce a contextualized representation for each word. Therefore, we can endow the input words with a query, key, and value vector and produce contextualized representation that way as substitutes for the hidden states. As before, we can package these into matrices $\textbf{Q} \in \mathbb{R}^{L \times d}, \textbf{K} \in \mathbb{R}^{L \times d} , \textbf{V} \in \mathbb{R}^{L \times d}$ and define attention vectors as before. We could then use these as substitutes for the hidden states of an encoder RNN. Moreover, these representations would be richer than the RNN's representation since they can not only peek into the future, but they also have direct access to each word's representation, as opposed to only the last word's representation. We can employ the analogous transformations on the decoder side as well.

It thus remains to devise a way to compute these matrices. We'll write out the computations for the self-attention in the encoder side and omit the decoder ones since they are analogous. By viewing $\textbf{x}$ as a matrix $\textbf{X} \in \mathbb{R}^{L \times d_{\text{input}}}$, we can construct these matrices by a linear transformation i.e. we define $\textbf{Q} = \textbf{X} W_Q$ for some matrix $W_Q$ and similarly for $V$ and $K$. We choose multiplication on the right because we want to act on the columns, as that will allow us to examine every timestep at once. For example, if we chose $W_Q$ as a block matrix, where the first $m$ rows consisted of ones in the diagonal and zero elsewhere, then $\textbf{Q} = \textbf{X} W_{Q}$ would consist of the the $m$ dimensions of the all features across time. 

The obvious problem is that any such restriction of features would be unable to capture global behavior. It would be fantastic if instead of relying on one set of matrices $(W_Q, W_V, W_K)$, we could rely on multiple matrices, say $(W_{Q_1},W_{V_1}, W_{K_1}), ..., (W_{Q_h}, W_{V_h}, W_{K_h})$ and combine them in a meaningful way so as to capture various of types of phenomena. This is exactly what Vaswani does with his idea of multiple heads. Moreover, to reduce computation constraints, the matrices have scaled their dimensions by a factor of $h$. In particular, instead of using $W_Q \in \mathbb{R}^{d_{\text{input}}\times d_{\text{input}}}$, Vaswani uses $W_{Q_i} \in \mathbb{R}^{d_{\text{input}} \times d_{\text{input}}/h}$ for $i=1,..., h$ and uses this to create attention vectors $\textbf{A}(1),...,\textbf{A}(h)$ then concatenates them to form the final $\textbf{A}$.

## Masked Self-Attention

In our traditional encoder-decoder scheme, it was clear that that during the prediction of the output, we were not peeking into the future. However, the contextualized representations we've made for each word in the decoder's input inadvertantly peek into the future due to the self-attention mechanism. To see this more clearly, notice that by construction, the $j$th row of $\textbf{Q}$ and $\textbf{K}$ only depends on the $j$th word $x_j$. In particular, this means that the $ij$th entry of $\textbf{Q}\textbf{K}^T$ only depends on the $i$th and $j$th words. Once we take a softmax in the $j$ variable, we'll be introducing information from every word into each entry of $\text{softmax}(\textbf{Q}\textbf{K}^T)$. Since these weights are the ones that determine which value vectors we use in the final representation of the words in our decoder's input, we need to prevent this peeking. The simplest way would be to mask the entries which cause the information leak i.e. set $$(\textbf{QK}^T)_{ij} = -\infty$$ for $i < j$. After this modification, the resulting entries $\text{softmax}(\textbf{QK}^T)_{ij}$ from the softmax operation depend only on the words up to $i$.

[jekyll-gh]: https://github.com/mojombo/jekyll
[jekyll]:    http://jekyllrb.com
