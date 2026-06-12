# Detailed Summary: *Neural Factorization Machines for Sparse Predictive Analytics*

**Paper:** Xiangnan He and Tat-Seng Chua. *Neural Factorization Machines for Sparse Predictive Analytics*. SIGIR 2017 / arXiv:1708.05027.

## 1. High-level overview

This paper proposes **Neural Factorization Machines (NFM)**, a model for prediction with high-dimensional sparse features, especially categorical features converted by one-hot encoding. The core problem is that many web, recommendation, advertising, and retrieval tasks use sparse categorical variables such as user ID, item ID, tag ID, context, occupation, gender, location, or time. In these settings, good performance depends heavily on modeling **feature interactions**.

Traditional **Factorization Machines (FM)** are strong for sparse prediction because they model all pairwise feature interactions using latent embeddings. However, the paper argues that FM is still fundamentally a **linear model over interaction terms** and only directly captures **second-order interactions**. This limits its ability to model complex real-world data where the relationship among features may be nonlinear and higher-order.

Recent deep models such as **Wide&Deep** and **DeepCross** can model nonlinear feature interactions, but they usually start by simply concatenating feature embeddings. The authors argue that this low-level concatenation does not explicitly encode interactions, so the deep layers must learn interactions from scratch. This makes such models harder to optimize and more prone to overfitting.

NFM is proposed as a bridge between these two worlds:

- It keeps the FM-style idea of modeling pairwise feature interactions.
- It introduces a neural operation called **Bi-Interaction pooling** to encode second-order interactions as a dense vector.
- It stacks nonlinear hidden layers on top of this interaction vector to learn higher-order and nonlinear interactions.
- It can recover FM as a special case when no hidden layer is used.

The paper’s main empirical claim is that **a shallow NFM with only one hidden layer can outperform FM, higher-order FM, Wide&Deep, and DeepCross** on two sparse regression tasks.

---

## 2. Motivation and problem setting

The paper focuses on **sparse predictive analytics**. In many IR and data mining tasks, input variables are categorical. After one-hot encoding, each example becomes a very high-dimensional sparse vector:

\[
\mathbf{x} \in \mathbb{R}^n
\]

where most entries are zero and nonzero entries indicate active categorical features. For example, in a recommendation setting, an instance may include:

- user ID,
- item ID,
- tag ID,
- time,
- location,
- device type,
- context features.

A simple linear model can assign weights to individual features, but it cannot capture that certain combinations of features matter. For example, a user may like a movie only under a specific genre-tag-context combination. Therefore, the model must account for **interactions between features**.

Manual feature crossing can create interaction features, but it requires domain knowledge and heavy feature engineering. FM solves part of this by learning interactions automatically, but FM is limited in expressiveness. NFM is designed to improve FM without relying on very deep architectures.

---

## 3. Background: Factorization Machines

Given an input feature vector \(\mathbf{x} \in \mathbb{R}^n\), a standard second-order FM predicts:

\[
\hat{y}_{FM}(\mathbf{x}) = w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n}\sum_{j=i+1}^{n} \mathbf{v}_i^T \mathbf{v}_j x_i x_j
\]

where:

- \(w_0\) is the global bias,
- \(w_i\) is the first-order weight for feature \(i\),
- \(\mathbf{v}_i \in \mathbb{R}^k\) is the embedding of feature \(i\),
- \(\mathbf{v}_i^T \mathbf{v}_j\) measures the interaction between features \(i\) and \(j\),
- \(x_i x_j\) ensures that only active feature pairs contribute.

FM is powerful because it can estimate interactions even when a feature pair rarely or never appears together in training, since the interaction is factorized through embeddings.

However, the paper emphasizes two limitations:

1. **Linearity:** FM is linear with respect to its parameters. It models interaction strength using an inner product, but it does not apply nonlinear transformations to interaction representations.
2. **Second-order restriction:** FM directly models only pairwise interactions. Higher-order FMs exist, but they are still linear and harder to estimate.

---

## 4. Background: Deep neural models for sparse data

The paper discusses deep models such as **Wide&Deep** and **DeepCross**. These methods also embed sparse features into dense vectors, but they usually concatenate the embeddings and pass the concatenated vector into an MLP or residual network.

The authors identify a key weakness:

> Concatenation itself does not model interactions.

If the model starts from concatenated embeddings, the hidden layers must discover useful feature interactions entirely by nonlinear transformations. In sparse data, this can require a deeper network and can cause optimization problems such as:

- vanishing or exploding gradients,
- degradation,
- overfitting,
- high sensitivity to initialization.

The paper shows empirically that Wide&Deep and DeepCross trained from random initialization perform poorly on the Frappe dataset. When initialized with FM embeddings, their performance improves greatly, suggesting that the deep models need good low-level interaction representations.

This motivates the main idea of NFM: instead of giving the neural network raw concatenated embeddings, give it an explicit interaction representation first.

---

# 5. Method section in detail: Neural Factorization Machines

## 5.1 Overall NFM formulation

NFM keeps the linear part of FM and replaces the FM pairwise scalar interaction term with a neural interaction function:

\[
\hat{y}_{NFM}(\mathbf{x}) = w_0 + \sum_{i=1}^{n} w_i x_i + f(\mathbf{x})
\]

The first two terms model the bias and first-order feature effects. The function \(f(\mathbf{x})\) is the neural component that models feature interactions.

The neural part contains four main stages:

1. **Embedding layer**
2. **Bi-Interaction pooling layer**
3. **Hidden layers**
4. **Prediction layer**

The model architecture shown in Figure 2 of the paper follows this flow:

\[
\text{sparse input} \rightarrow \text{embedding lookup} \rightarrow \text{Bi-Interaction pooling} \rightarrow \text{hidden layers} \rightarrow \text{prediction score}
\]

The linear first-order regression part is added separately to the final prediction.

---

## 5.2 Embedding layer

Each feature \(i\) has an embedding vector:

\[
\mathbf{v}_i \in \mathbb{R}^k
\]

For an input vector \(\mathbf{x}\), NFM constructs a set of scaled embeddings:

\[
\mathcal{V}_x = \{x_1\mathbf{v}_1, x_2\mathbf{v}_2, \ldots, x_n\mathbf{v}_n\}
\]

Because \(\mathbf{x}\) is sparse, only nonzero features need to be included:

\[
\mathcal{V}_x = \{x_i\mathbf{v}_i \mid x_i \neq 0\}
\]

This is important because NFM works not only with binary one-hot features but also with real-valued sparse features. Scaling the embedding by \(x_i\) allows the feature value to affect the interaction representation.

---

## 5.3 Bi-Interaction pooling layer

This is the central contribution of the paper.

Instead of concatenating embeddings, NFM computes the element-wise product for every pair of active feature embeddings and sums them:

\[
f_{BI}(\mathcal{V}_x) = \sum_{i=1}^{n}\sum_{j=i+1}^{n} x_i\mathbf{v}_i \odot x_j\mathbf{v}_j
\]

where \(\odot\) is element-wise multiplication.

The output is a \(k\)-dimensional vector. Each dimension aggregates pairwise interactions between all active features in that latent dimension. This differs from FM in an important way:

- FM sums pairwise interactions into a **single scalar**.
- NFM keeps the pairwise interaction information as a **vector**, then passes it to nonlinear layers.

So Bi-Interaction pooling is like a vectorized version of FM’s pairwise interaction term.

### Why this is useful

Bi-Interaction pooling gives the neural network a more informative low-level input than concatenation. It explicitly encodes second-order interactions before the hidden layers. Therefore, the hidden layers do not need to learn pairwise interaction structure from scratch; they can focus on nonlinear transformations and higher-order patterns.

### Efficient computation

A naive implementation would require all feature pairs, which seems expensive. But the paper shows that Bi-Interaction pooling can be rewritten as:

\[
f_{BI}(\mathcal{V}_x) = \frac{1}{2}\left[\left(\sum_{i=1}^{n} x_i\mathbf{v}_i\right)^2 - \sum_{i=1}^{n}(x_i\mathbf{v}_i)^2\right]
\]

where the square is element-wise.

This makes the operation efficient. If \(N_x\) is the number of nonzero features in \(\mathbf{x}\), the complexity is:

\[
O(kN_x)
\]

This is the same order as standard FM and does not require enumerating all pairs explicitly.

---

## 5.4 Hidden layers

After Bi-Interaction pooling, NFM applies fully connected hidden layers:

\[
\mathbf{z}_1 = \sigma_1(\mathbf{W}_1 f_{BI}(\mathcal{V}_x) + \mathbf{b}_1)
\]

\[
\mathbf{z}_2 = \sigma_2(\mathbf{W}_2\mathbf{z}_1 + \mathbf{b}_2)
\]

\[
\cdots
\]

\[
\mathbf{z}_L = \sigma_L(\mathbf{W}_L\mathbf{z}_{L-1} + \mathbf{b}_L)
\]

where:

- \(L\) is the number of hidden layers,
- \(\mathbf{W}_l\) and \(\mathbf{b}_l\) are parameters of layer \(l\),
- \(\sigma_l\) is an activation function such as ReLU, sigmoid, or tanh.

The purpose of the hidden layers is to model **higher-order and nonlinear feature interactions**. Since their input already contains second-order interactions, nonlinear transformations over this vector can express more complex patterns.

The paper finds that **one hidden layer is usually enough**. Adding more layers does not improve performance and can even slightly hurt performance. This supports the authors’ argument that a good low-level interaction representation can reduce the need for very deep architectures.

---

## 5.5 Prediction layer

The final hidden representation is mapped to a scalar interaction score:

\[
f(\mathbf{x}) = \mathbf{h}^T\mathbf{z}_L
\]

where \(\mathbf{h}\) is the prediction-layer weight vector.

The full NFM prediction is:

\[
\hat{y}_{NFM}(\mathbf{x}) = w_0 + \sum_{i=1}^{n} w_i x_i + \mathbf{h}^T\sigma_L\left(\mathbf{W}_L(\cdots \sigma_1(\mathbf{W}_1 f_{BI}(\mathcal{V}_x) + \mathbf{b}_1)\cdots) + \mathbf{b}_L\right)
\]

The trainable parameters are:

\[
\Theta = \{w_0, \{w_i\}, \{\mathbf{v}_i\}, \mathbf{h}, \{\mathbf{W}_l, \mathbf{b}_l\}_{l=1}^{L}\}
\]

Compared with FM, NFM adds mainly the hidden-layer parameters \(\mathbf{W}_l, \mathbf{b}_l\), while keeping the same sparse embedding foundation.

---

## 5.6 NFM generalizes FM

The paper shows that FM is a special case of NFM.

If NFM has no hidden layer, the model directly projects the Bi-Interaction vector to a scalar:

\[
\hat{y}_{NFM-0}(\mathbf{x}) = w_0 + \sum_{i=1}^{n}w_i x_i + \mathbf{h}^T\sum_{i=1}^{n}\sum_{j=i+1}^{n}x_i\mathbf{v}_i \odot x_j\mathbf{v}_j
\]

Expanding the last term:

\[
\hat{y}_{NFM-0}(\mathbf{x}) = w_0 + \sum_{i=1}^{n}w_i x_i + \sum_{i=1}^{n}\sum_{j=i+1}^{n}\sum_{f=1}^{k}h_f v_{if}v_{jf}x_i x_j
\]

If \(\mathbf{h}\) is fixed to an all-one vector, this becomes exactly the standard FM interaction term:

\[
\sum_{i=1}^{n}\sum_{j=i+1}^{n}\mathbf{v}_i^T\mathbf{v}_j x_i x_j
\]

This means FM can be viewed as a shallow NFM without nonlinear hidden layers. The paper argues that this is useful because it brings FM into a neural network framework, allowing techniques such as dropout and batch normalization to be used naturally.

---

## 5.7 Relation to Wide&Deep and DeepCross

NFM has a similar high-level neural structure to Wide&Deep and DeepCross, but the low-level interaction operation is different.

If we replace Bi-Interaction pooling with simple embedding concatenation:

- with an MLP, the architecture resembles Wide&Deep;
- with residual units, the architecture resembles DeepCross.

The key difference is that concatenation does not explicitly model feature interaction. NFM’s Bi-Interaction pooling does. Therefore, NFM gives hidden layers a stronger starting representation and makes learning easier.

This is the main architectural argument of the paper:

> A better low-level interaction operation can be more valuable than simply stacking deeper layers.

---

## 5.8 Time complexity

The Bi-Interaction pooling layer has complexity:

\[
O(kN_x)
\]

where:

- \(k\) is the embedding size,
- \(N_x\) is the number of nonzero features in the input.

For hidden layer \(l\), the matrix-vector multiplication costs:

\[
O(d_{l-1}d_l)
\]

where \(d_l\) is the dimensionality of layer \(l\), and \(d_0 = k\).

The full NFM evaluation complexity is:

\[
O\left(kN_x + \sum_{l=1}^{L} d_{l-1}d_l\right)
\]

The paper notes that this is the same general complexity form as Wide&Deep and DeepCross, but NFM gets better interaction modeling from the Bi-Interaction layer.

---

## 5.9 Learning objective

The paper focuses on regression tasks and uses squared loss:

\[
\mathcal{L}_{reg} = \sum_{\mathbf{x}\in\mathcal{X}}(\hat{y}(\mathbf{x}) - y(\mathbf{x}))^2
\]

where:

- \(\mathcal{X}\) is the training set,
- \(y(\mathbf{x})\) is the target value,
- \(\hat{y}(\mathbf{x})\) is the model prediction.

The authors note that NFM can also be used for classification or ranking by replacing the objective with log loss, hinge loss, pairwise ranking loss, or contrastive max-margin loss.

For optimization, they use mini-batch **Adagrad**, not vanilla SGD. Adagrad adapts learning rates during training, making the model easier to optimize.

The gradient through Bi-Interaction pooling is given as:

\[
\frac{\partial f_{BI}(\mathcal{V}_x)}{\partial \mathbf{v}_i}
= \left(\sum_{j=1}^{n}x_j\mathbf{v}_j\right)x_i - x_i^2\mathbf{v}_i
= \sum_{j=1,j\neq i}^{n}x_i x_j\mathbf{v}_j
\]

This shows the Bi-Interaction operation is differentiable and can be trained end-to-end with standard backpropagation.

---

## 5.10 Dropout in NFM

The paper applies dropout in two places:

1. On the **Bi-Interaction layer**
2. On the **hidden layers**

The most interesting part is dropout on the Bi-Interaction layer. After computing the \(k\)-dimensional vector \(f_{BI}(\mathcal{V}_x)\), the model randomly drops a proportion \(\rho\) of latent factors during training.

This has two effects:

- It prevents feature embeddings from co-adapting too strongly.
- It acts as a new regularization method for FM-like models.

Since NFM with no hidden layers reduces to FM, applying dropout to the Bi-Interaction layer can be interpreted as a neural regularization strategy for FM. The paper finds that this can outperform traditional L2 regularization on embeddings.

---

## 5.11 Batch normalization in NFM

The paper also applies **batch normalization (BN)** to stabilize and speed up training.

BN is applied:

1. to the output of the Bi-Interaction pooling layer;
2. to each hidden layer.

The motivation is that updating feature embeddings can change the distribution of the Bi-Interaction output, which then changes the input distribution to later layers. BN normalizes mini-batch inputs and reduces internal covariate shift.

The BN formula used is:

\[
BN(\mathbf{x}_i) = \gamma \odot \frac{\mathbf{x}_i - \mu_B}{\sigma_B} + \beta
\]

where:

- \(\mu_B\) is the mini-batch mean,
- \(\sigma_B\) is the mini-batch standard deviation,
- \(\gamma\) and \(\beta\) are trainable scale and shift parameters.

Experimentally, BN speeds up training. However, when used together with dropout, it can introduce some instability because dropout changes the input distribution that BN normalizes.

---

# 6. Experiments

## 6.1 Research questions

The experiments are designed around three questions:

1. **RQ1:** Can Bi-Interaction pooling effectively capture second-order feature interactions? How do dropout and BN affect it?
2. **RQ2:** Are hidden layers useful for modeling higher-order interactions and improving FM?
3. **RQ3:** How does NFM compare with higher-order FM and deep baselines such as Wide&Deep and DeepCross?

---

## 6.2 Datasets

The paper uses two public datasets.

### Frappe

Frappe is a context-aware app recommendation dataset. Each log contains:

- user ID,
- app ID,
- context variables such as weather, city, and daytime.

The authors one-hot encode the features, producing **5,382 features**. The final dataset has **288,609 instances**.

### MovieLens

The paper uses the tagging part of MovieLens for personalized tag recommendation. Each instance contains:

- user ID,
- movie ID,
- tag.

The one-hot representation contains **90,445 features**. The final dataset has **2,006,859 instances**.

Both datasets originally contain only positive observations. The authors sample two negative instances for every positive instance and assign negative samples target value \(-1\), while positive samples have target value \(1\).

---

## 6.3 Evaluation setting

The data is split into:

- 70% training,
- 20% validation,
- 10% test.

The evaluation metric is **RMSE**. Lower RMSE is better.

The baselines are:

- **LibFM:** official FM implementation.
- **HOFM:** higher-order factorization machine.
- **Wide&Deep:** MLP over concatenated embeddings plus wide linear part.
- **DeepCross:** residual network over concatenated embeddings.

The authors tune learning rate, regularization, dropout ratio, and embedding size. NFM is implemented in TensorFlow and optimized using mini-batch Adagrad.

---

# 7. Experimental findings

## 7.1 Bi-Interaction pooling and dropout

The authors first study **NFM-0**, which has Bi-Interaction pooling but no hidden layers. This isolates the effect of the Bi-Interaction layer.

The results show:

- Linear regression performs poorly, confirming that feature interactions matter.
- NFM-0 behaves similarly to FM, as expected.
- Dropout on the Bi-Interaction layer improves generalization.
- Dropout can outperform L2 regularization on embeddings.

On Frappe, the best dropout setting achieves validation RMSE **0.3562**, better than the best L2 regularization result **0.3799**.

This supports the claim that dropout is a useful regularizer for FM-style latent interaction models.

---

## 7.2 Batch normalization

BN speeds up convergence. On Frappe, with BN, the training error at epoch 20 is already lower than the training error at epoch 60 without BN.

BN gives slight generalization improvement, but the improvement is not statistically significant. The paper also observes that combining dropout and BN can make learning less stable because dropout randomly changes the distribution that BN tries to normalize.

---

## 7.3 Hidden layers and nonlinear activations

The paper then studies NFM with one hidden layer and different activation functions.

Main findings:

- Nonlinear hidden layers significantly improve over NFM-0/FM.
- On Frappe, one-hidden-layer NFM improves over NFM-0 by about **11.3% relative**.
- On MovieLens, it improves by about **5.2% relative**.
- Identity activation, which is only a linear transformation, performs worse than nonlinear activations.

This shows that nonlinear transformations over the Bi-Interaction vector are important for learning higher-order and nonlinear feature interactions.

---

## 7.4 Number of hidden layers

The paper compares NFM with 0 to 4 hidden layers.

Validation RMSE:

| Method | Frappe | MovieLens |
|---|---:|---:|
| NFM-0 | 0.3562 | 0.4901 |
| NFM-1 | **0.3133** | **0.4646** |
| NFM-2 | 0.3193 | 0.4681 |
| NFM-3 | 0.3219 | 0.4752 |
| NFM-4 | 0.3202 | 0.4703 |

The best result is achieved with **one hidden layer**. More layers do not help. This supports the paper’s main design philosophy: with a strong interaction-aware low-level representation, the model does not need to be very deep.

---

## 7.5 Pre-training

The authors test initializing NFM with FM embeddings.

Findings:

- Pre-training greatly speeds up convergence.
- With only 5 epochs, pre-trained NFM can reach performance similar to 40 epochs of randomly initialized NFM.
- However, pre-training does not improve final performance.
- Random initialization can even produce slightly better final results.

This differs from Wide&Deep and DeepCross, where FM pre-training is crucial for good performance. The authors interpret this as evidence that NFM is easier to optimize because Bi-Interaction pooling gives a more useful low-level representation.

---

## 7.6 Comparison with baselines

The final comparison shows that NFM consistently performs best.

For embedding size 256:

| Method | Frappe RMSE | MovieLens RMSE |
|---|---:|---:|
| LibFM | 0.3385 | 0.4735 |
| HOFM | 0.3331 | 0.4636 |
| Wide&Deep | 0.3661 | 0.5313 |
| Wide&Deep + pre-training | 0.3246 | 0.4512 |
| DeepCross | 0.4071 | 0.5907 |
| DeepCross + pre-training | 0.3548 | 0.5130 |
| NFM | **0.3095** | **0.4443** |

Main conclusions:

1. **NFM outperforms all baselines** on both datasets.
2. **HOFM improves only slightly over FM**, suggesting that modeling higher-order interactions linearly is not enough.
3. **Wide&Deep needs pre-training** to become competitive.
4. **DeepCross performs poorly**, likely due to overfitting and optimization difficulty.
5. **NFM uses fewer parameters than deep baselines** while achieving better performance.

---

# 8. Key contributions

The paper’s main contributions are:

1. **Bi-Interaction pooling:** A new pooling operation that explicitly encodes pairwise feature interactions as a vector.
2. **Neural view of FM:** FM is shown as a special case of NFM with no hidden layers.
3. **Nonlinear extension of FM:** By stacking nonlinear layers above Bi-Interaction pooling, NFM can model higher-order nonlinear feature interactions.
4. **Efficient sparse prediction:** Bi-Interaction pooling keeps FM-like linear complexity with respect to the number of nonzero features.
5. **Strong empirical performance:** One-hidden-layer NFM beats FM, HOFM, Wide&Deep, and DeepCross on two sparse regression benchmarks.

---

# 9. Why the method works

The important insight is that NFM changes the low-level input to the neural network.

In standard deep sparse models:

\[
\text{embeddings} \rightarrow \text{concatenation} \rightarrow \text{deep network}
\]

The network must learn interactions from raw embedding concatenation.

In NFM:

\[
\text{embeddings} \rightarrow \text{explicit pairwise interaction vector} \rightarrow \text{shallow nonlinear network}
\]

This gives the model a stronger inductive bias. It combines:

- FM’s strength in sparse pairwise interaction modeling;
- neural networks’ strength in nonlinear transformation;
- dropout and BN for better neural optimization and regularization.

That is why a shallow NFM can outperform much deeper models.

---

# 10. Limitations and possible weaknesses

Although the paper is strong, several limitations are worth noting:

1. **Task scope is regression-focused.** The paper says NFM can be used for classification and ranking, but experiments focus on squared-loss regression.
2. **Only two datasets are used.** Both are recommendation-style sparse prediction tasks. More diverse IR tasks would strengthen the claim.
3. **Bi-Interaction pooling sums all pairwise interactions.** This may lose fine-grained pair-level information because all interactions are pooled into one vector.
4. **No attention over interactions.** Every pair contributes through the same pooling mechanism. Later work such as attentional FM-style models can distinguish important and unimportant feature pairs.
5. **Higher-order interactions are implicit.** NFM does not explicitly enumerate third-order or higher interactions; it relies on nonlinear layers to learn them from the Bi-Interaction representation.
6. **Negative sampling may affect evaluation.** Since both datasets contain only positive observations originally, the sampled negatives may not fully represent real negative feedback.

---

# 11. Relevance to retrieval and recommendation

This paper is especially relevant for recommendation, retrieval ranking, and code/text retrieval settings where examples can be represented as sparse categorical or symbolic features.

For example, in code retrieval or generative retrieval, one may have sparse features such as:

- query cluster ID,
- document ID,
- repository ID,
- programming language,
- API/entity features,
- tag or intent labels,
- structural code features.

The NFM idea suggests that instead of only concatenating embeddings or using a linear interaction score, we can first compute an explicit pairwise interaction representation and then apply a shallow nonlinear module. This may be useful when the relevance of a query-document pair depends on combinations such as:

- query intent × API usage,
- programming language × code pattern,
- repository/domain × function type,
- query cluster × document cluster.

The main transferable lesson is:

> For sparse retrieval features, designing a strong interaction layer may be more effective and easier to train than simply making the model deeper.

---

# 12. One-paragraph summary

*Neural Factorization Machines for Sparse Predictive Analytics* proposes NFM, a model that improves Factorization Machines by adding neural nonlinear modeling on top of explicit pairwise feature interactions. Its key component is Bi-Interaction pooling, which sums element-wise products of feature embeddings to produce a vector representation of second-order interactions. This vector is passed through nonlinear hidden layers to learn higher-order interactions. FM becomes a special case of NFM when no hidden layer is used. The model is efficient, differentiable, and trainable end-to-end with Adagrad. Experiments on Frappe and MovieLens show that a one-hidden-layer NFM outperforms FM, higher-order FM, Wide&Deep, and DeepCross, while being easier to train and using fewer parameters than deep baselines. The central lesson is that an informative interaction-aware low-level representation can reduce the need for very deep neural networks in sparse predictive tasks.
