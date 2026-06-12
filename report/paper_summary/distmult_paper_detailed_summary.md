# Detailed Summary of “Embedding Entities and Relations for Learning and Inference in Knowledge Bases”

**Paper:** Bishan Yang, Wen-tau Yih, Xiaodong He, Jianfeng Gao, Li Deng. *Embedding Entities and Relations for Learning and Inference in Knowledge Bases*. ICLR 2015.

---

## 1. High-level Overview

This paper studies **knowledge base embedding models**, where entities and relations in a knowledge base (KB) are represented in low-dimensional vector spaces. A KB is represented as a set of triples:

\[
(e_1, r, e_2)
\]

where \(e_1\) is the subject entity, \(r\) is the relation/predicate, and \(e_2\) is the object entity.

The paper has two central goals:

1. **Unify existing neural KB embedding models** such as TransE and Neural Tensor Networks (NTN) under a common framework.
2. **Study what kind of relation representations are useful**, not only for link prediction, but also for extracting logical rules from a KB.

The most important result is that a very simple bilinear model, especially its diagonal version later known as **DistMult**, performs surprisingly well. On Freebase, it outperforms TransE and more complex neural tensor models for link prediction. The paper also shows that bilinear relation embeddings are useful for **compositional reasoning**, meaning that relation embeddings can be composed to discover logical Horn rules.

---

## 2. Motivation

Large knowledge bases such as Freebase, DBpedia, and YAGO contain millions or billions of facts, but they are incomplete. A common task is therefore **KB completion**, where the model predicts missing facts.

Traditional statistical relational learning methods, such as Markov logic networks, can model logical dependencies, but they often do not scale well to very large KBs. Embedding-based models are more scalable because they represent entities and relations as continuous vectors or matrices.

Before this paper, models such as TransE and NTN had already shown strong performance, but it was unclear:

- which design choices matter most;
- whether complex relation operators are actually better;
- whether learned relation embeddings capture interpretable relational semantics;
- whether embeddings can be used for explicit rule extraction, not just link prediction.

The authors address these questions through a unified formulation and empirical comparison.

---

## 3. Main Contributions

The paper makes three main contributions:

1. **A unified neural embedding framework** for multi-relational learning. It shows that existing models such as TransE, NTN, single-layer models, and distance models can be interpreted as different choices of relation scoring functions.

2. **Empirical comparison of relation modeling choices.** The paper compares models with linear, bilinear, and combined linear-bilinear relation operators. It finds that a simple bilinear diagonal model, DistMult, gives strong link prediction results, especially on Freebase.

3. **Embedding-based rule extraction.** The paper proposes **EmbedRule**, a method that uses learned relation embeddings to mine Horn rules. The key idea is that relation composition can be represented by vector addition or matrix multiplication in embedding space.

---

## 4. Problem Setup

A knowledge base is a set of triples:

\[
T = \{(e_1, r, e_2)\}
\]

The goal is to learn representations such that valid triples receive high scores and invalid triples receive low scores.

The model has two parts:

1. **Entity representation layer:** maps input entities to dense vectors.
2. **Relation scoring layer:** combines two entity vectors using relation-specific parameters and outputs a score.

For link prediction, given a test triple, the model ranks candidate replacement entities and evaluates whether the correct entity appears near the top.

---

# 5. Method Section in Detail

## 5.1 Entity Representations

Each entity begins as either:

- a **one-hot vector**, where each entity has a unique index; or
- an **n-hot feature vector**, such as a bag-of-words representation.

The entity vector is projected into a dense representation:

\[
y_{e_1} = f(Wx_{e_1}), \quad y_{e_2} = f(Wx_{e_2})
\]

where:

- \(x_{e_1}\), \(x_{e_2}\) are input entity vectors;
- \(W\) is the projection matrix;
- \(f\) can be a linear or nonlinear activation function;
- \(y_{e_1}\), \(y_{e_2}\) are learned entity embeddings.

Most KB embedding models use one-hot entity vectors. NTN is different because it represents an entity as an average of word vectors from the entity name.

The paper also investigates initialization strategies:

- random initialization;
- word-vector initialization;
- entity-vector initialization using pretrained entity embeddings.

The strongest result comes from using **entity-level pretrained vectors** with a nonlinear projection.

---

## 5.2 Relation Representations and Scoring Functions

The relation representation determines how the model scores a triple. The authors show that many previous models can be expressed using two basic components:

### 5.2.1 Linear Relation Transformation

\[
g^a_r(y_{e_1}, y_{e_2}) = A_r^T
\begin{bmatrix}
y_{e_1} \\
y_{e_2}
\end{bmatrix}
\]

This type of scoring captures additive or linear interactions between the two entity embeddings.

### 5.2.2 Bilinear Relation Transformation

\[
g^b_r(y_{e_1}, y_{e_2}) = y_{e_1}^T B_r y_{e_2}
\]

Here, \(B_r\) is a relation-specific matrix. This form captures multiplicative interactions between subject and object embeddings.

The paper argues that bilinear interaction is especially important because relations in a KB often depend on how dimensions of the subject and object embeddings interact.

---

## 5.3 How Existing Models Fit the Framework

The paper shows that several models are special cases of this general formulation.

### Distance Model

Earlier structured embedding models use a distance-based score between transformed subject and object embeddings.

### Single-layer Model

The single-layer model applies a linear transformation followed by a nonlinear activation.

### TransE

TransE represents each relation as a translation vector:

\[
y_{e_1} + v_r \approx y_{e_2}
\]

Its score is based on the distance between \(y_{e_1} + v_r\) and \(y_{e_2}\). In this paper’s framework, TransE can be rewritten as a combination of a linear term and a fixed bilinear term.

The key intuition of TransE is **additive composition**: relations move one entity embedding toward another entity embedding.

### Neural Tensor Network

NTN uses both:

- a bilinear tensor operator;
- a linear operator;
- a nonlinear hidden layer.

It is the most expressive model considered, but also has many parameters. The paper finds that NTN tends to overfit on the evaluated datasets.

---

## 5.4 The Bilinear Model

The paper studies a simpler bilinear scoring function:

\[
S(e_1, r, e_2) = y_{e_1}^T M_r y_{e_2}
\]

where \(M_r\) is a relation-specific matrix.

This model directly scores how compatible the subject and object embeddings are under relation \(r\). Compared with TransE, it uses multiplicative rather than additive interaction.

The full bilinear model is expressive, but it requires a full matrix for every relation. If entity embeddings have dimension \(d\), then each relation has \(d \times d\) parameters.

---

## 5.5 Bilinear-diag / DistMult

To reduce parameters, the paper restricts \(M_r\) to be diagonal:

\[
S(e_1, r, e_2) = y_{e_1}^T \operatorname{diag}(r) y_{e_2}
\]

Equivalently:

\[
S(e_1, r, e_2) = \sum_i y_{e_1,i} r_i y_{e_2,i}
\]

This is the model commonly known as **DistMult**.

### Why DistMult is important

DistMult has the same number of relation parameters as TransE, but it uses multiplicative interactions. The paper finds that this simple multiplicative scoring performs better than TransE on Freebase.

### Main limitation

Because the relation matrix is diagonal, DistMult is symmetric:

\[
S(e_1, r, e_2) = S(e_2, r, e_1)
\]

This makes it theoretically unable to model asymmetric relations perfectly, such as *parentOf* or *bornIn*. However, despite this limitation, it works very well empirically on the evaluated link prediction benchmarks.

---

## 5.6 Additive vs. Multiplicative Interactions

A key conceptual comparison in the paper is:

- **TransE / DistAdd:** additive relation modeling;
- **DistMult:** multiplicative relation modeling.

TransE models a relation as a vector offset:

\[
y_{e_1} + v_r \approx y_{e_2}
\]

DistMult models a relation as dimension-wise compatibility:

\[
\sum_i y_{e_1,i} r_i y_{e_2,i}
\]

The experiments show that DistMult significantly outperforms TransE on Freebase, especially across relation categories such as one-to-many, many-to-one, and many-to-many.

This result is important because it suggests that multiplicative interactions can capture relational compatibility more effectively than simple translations, even with the same number of parameters.

---

## 5.7 Training Objective

The models are trained with a **margin-based ranking loss**.

For every positive triple \((e_1, r, e_2)\), the model creates negative triples by corrupting either the subject or object:

\[
(e'_1, r, e_2), \quad (e_1, r, e'_2)
\]

The model is trained so that positive triples receive higher scores than negative triples by at least a margin:

\[
L(\Omega) = \sum_{(e_1,r,e_2) \in T} \sum_{(e'_1,r,e'_2) \in T'}
\max\{S(e'_1,r,e'_2) - S(e_1,r,e_2) + 1, 0\}
\]

where:

- \(T\) is the set of observed positive triples;
- \(T'\) is the set of corrupted negative triples;
- \(S\) is the scoring function.

Training uses mini-batch stochastic gradient descent with AdaGrad. Entity vectors are normalized to unit length after each gradient update, and relation parameters use L2 regularization.

---

# 6. Inference Task I: Link Prediction

## 6.1 Task Definition

The first evaluation task is **link prediction**. Given a triple, the model predicts a missing subject or object entity.

For example:

\[
(?, \text{BornInCity}, \text{London})
\]

or:

\[
(\text{Christopher Nolan}, \text{BornInCity}, ?)
\]

The model scores all candidate entities and ranks them.

## 6.2 Datasets

The paper evaluates on:

- **WordNet (WN):** 151,442 triples, 40,943 entities, 18 relations.
- **Freebase FB15k:** 592,213 triples, 14,951 entities, 1,345 relations.
- **FB15k-401:** a subset of FB15k with frequent relations only, containing 401 relations.

## 6.3 Metrics

The paper uses:

- **MRR:** Mean Reciprocal Rank.
- **Hits@10:** percentage of cases where the correct entity appears in the top 10.
- **MAP with type checking:** used in the pretrained-vector experiment.

## 6.4 Main Link Prediction Results

The paper compares NTN, Bilinear+Linear, TransE, Bilinear, and Bilinear-diag/DistMult.

Main findings:

- NTN performs worst on Freebase and WordNet, likely because it overfits.
- TransE is strong but is outperformed by bilinear variants.
- Bilinear performs very well, especially on WordNet.
- DistMult achieves the best results on Freebase.

On FB15k-401, DistMult obtains:

- **MRR:** 0.36
- **Hits@10:** 58.5

With nonlinear projection and entity-vector initialization, DistMult-tanh-EV-init improves to:

- **MRR:** 0.42
- **Hits@10:** 73.2
- **MAP with type checking:** 88.2

This is one of the strongest empirical results in the paper.

---

# 7. Inference Task II: Rule Extraction

The second task is more interesting from a reasoning perspective. The authors ask whether learned relation embeddings can be used to extract logical rules.

## 7.1 Horn Rules

The paper focuses on Horn rules such as:

\[
\text{BornInCity}(a,b) \wedge \text{CityInCountry}(b,c)
\Rightarrow \text{Nationality}(a,c)
\]

This means: if person \(a\) was born in city \(b\), and city \(b\) is in country \(c\), then person \(a\) likely has nationality \(c\).

The body relations form a path in the graph, and the head relation closes the path.

The paper focuses on rules of length 2 and 3.

---

## 7.2 Key Idea of EmbedRule

The core idea is that the body of a rule is a **composition of relations**.

For a length-2 rule:

\[
B_1(a,b) \wedge B_2(b,c) \Rightarrow H(a,c)
\]

The composed relation \(B_1 \circ B_2\) should behave similarly to the head relation \(H\).

So if relation embeddings capture semantics well, then:

\[
M_{B_1} M_{B_2} \approx M_H
\]

for bilinear matrix embeddings.

For TransE-style vector embeddings, relation composition is modeled as vector addition:

\[
v_{B_1} + v_{B_2} \approx v_H
\]

For bilinear relation embeddings, relation composition is modeled as matrix multiplication:

\[
M_{B_1} M_{B_2} \approx M_H
\]

This is one of the most important insights in the paper: **bilinear relation embeddings naturally support compositional reasoning through matrix multiplication.**

---

## 7.3 EmbedRule Algorithm

For each candidate head relation \(r\), EmbedRule:

1. Selects possible start relations whose subject domain overlaps with the subject domain of \(r\).
2. Selects possible end relations whose object domain overlaps with the object domain of \(r\).
3. Enumerates possible relation sequences satisfying type constraints.
4. Composes the embeddings of the body relations.
5. Computes the distance between the composed body embedding and the head relation embedding.
6. Selects nearest-neighbor relation sequences as candidate rules.
7. Ranks extracted rules by confidence.

The method is efficient because its search depends mainly on the number of relation types, not the total number of facts in the KB.

---

## 7.4 Rule Extraction Results

The authors compare EmbedRule with **AMIE**, a state-of-the-art rule mining system.

The results show:

- EmbedRule using bilinear embeddings outperforms AMIE on length-2 rules.
- DistMult and Bilinear variants are better than TransE-style additive embeddings for rule extraction.
- For length-3 rules, full Bilinear performs better at the top of the ranked list, suggesting that full matrices capture more complex relation semantics than diagonal matrices.
- DistMult-tanh-EV-init performs strongly as more predictions are generated.

This supports the claim that bilinear relation embeddings capture compositional semantics.

---

# 8. Important Visual Results

## Page 5: Link Prediction Tables

The tables on page 5 show that DistMult outperforms TransE on Freebase-style datasets. The same page also breaks results down by relation category, showing that DistMult is especially stronger on one-to-many and many-to-one settings.

## Page 9: Rule Extraction Precision Curves

The figures on page 9 compare the aggregated precision of rules extracted by different methods. The bilinear variants of EmbedRule consistently perform better than AMIE for length-2 rules, and they also achieve strong precision for length-3 rules.

## Page 11: Relation Embedding Visualization

The t-SNE visualizations on page 11 compare relation embeddings learned by DistAdd and DistMult. The DistMult embeddings form clearer semantic clusters, suggesting that multiplicative bilinear embeddings capture more interpretable relation structures.

---

# 9. Strengths of the Paper

## 9.1 Simple but Strong Model

The paper shows that a very simple diagonal bilinear model can outperform more complex models. This is important because it challenges the assumption that more expressive neural architectures are always better.

## 9.2 Clear Unification of Existing Models

By showing how TransE, NTN, and bilinear models fit into one framework, the paper clarifies the design space of KB embedding models.

## 9.3 Strong Analysis of Relation Semantics

The rule extraction section goes beyond standard link prediction. It asks what the learned embeddings actually encode and demonstrates that bilinear embeddings capture relation composition.

## 9.4 Connection Between Embeddings and Logic

The paper provides a bridge between distributed embedding models and symbolic logical rules. This is valuable because KB reasoning often requires both statistical generalization and logical interpretability.

---

# 10. Limitations

## 10.1 DistMult Cannot Model Asymmetry Well

Because DistMult uses diagonal relation matrices, its score is symmetric with respect to subject and object. This makes it unsuitable for strongly asymmetric relations in theory.

Later models such as ComplEx and RotatE address this limitation by allowing asymmetric scoring.

## 10.2 Evaluation Uses Older Benchmarks

The paper uses FB15k and WordNet-style benchmarks. Later work showed that some early KB completion datasets contain inverse relation leakage, making them easier than more carefully filtered benchmarks such as FB15k-237 and WN18RR.

## 10.3 Rule Extraction Precision Is Estimated

Because KBs are incomplete, a predicted fact that is not in the test set may still be true. The authors partially address this by manually checking some predictions, but evaluation remains approximate.

## 10.4 Full Bilinear Model Is Parameter-heavy

Full bilinear relation matrices are more expressive but require many parameters. DistMult is more scalable, but it loses expressive power.

---

# 11. Relation to Later Knowledge Graph Embedding Work

This paper is important historically because it popularized and analyzed the model now widely called **DistMult**.

Later models can be viewed as addressing DistMult’s limitations:

- **ComplEx** extends DistMult to complex-valued embeddings, enabling asymmetric relations.
- **RotatE** models relations as rotations in complex space.
- **ConvE** uses convolutional neural networks for scoring triples.
- **CompGCN** later uses relation composition operations, including multiplication and circular correlation, inside graph neural networks.

The paper’s insight that relation composition can be modeled through embedding operations directly influences later work on neural-symbolic reasoning and relational graph neural networks.

---

# 12. Practical Takeaways

For someone working on retrieval, representation learning, or graph-based reasoning, the paper gives several useful lessons:

1. **Multiplicative interaction is powerful.** DistMult’s strong performance suggests that matching dimensions between two embeddings through relation-specific weights is very effective.

2. **A simple model can outperform a complex one.** NTN is more expressive but performs worse, likely due to overfitting.

3. **Relation embeddings can encode compositional semantics.** This is useful for rule mining and multi-hop reasoning.

4. **Pretraining can significantly improve KB embeddings.** Entity-level pretrained vectors help much more than word-level initialization.

5. **Diagonal relation matrices are efficient but limited.** They are useful for scalability, but full matrices or complex embeddings are needed for richer asymmetric reasoning.

---

# 13. Concise Method Summary

The paper proposes a unified neural framework for KB embedding models. Entities are projected into dense vectors, and relation-specific scoring functions determine whether a triple is plausible. The authors compare additive models like TransE with multiplicative bilinear models. Their main model, Bilinear-diag/DistMult, scores triples by:

\[
S(e_1,r,e_2)=\sum_i y_{e_1,i} r_i y_{e_2,i}
\]

It is trained with a margin ranking loss using corrupted negative triples. Despite its simplicity, it outperforms TransE and NTN on Freebase link prediction. The paper further proposes EmbedRule, which composes relation embeddings to mine Horn rules. For bilinear models, relation composition is modeled as matrix multiplication, allowing the model to discover rules such as:

\[
\text{BornInLocation}(a,b) \wedge \text{LocationInCountry}(b,c)
\Rightarrow \text{Nationality}(a,c)
\]

Overall, the paper shows that bilinear embeddings are not only strong for link prediction but also useful for extracting compositional relational knowledge.

---

# 14. Why This Paper Matters

This paper is significant because it demonstrates that the structure of the scoring function matters deeply in knowledge graph embedding. It shows that simple multiplicative bilinear models can outperform more complex tensor models and translation models. It also provides one of the early strong demonstrations that neural embeddings can support symbolic-style rule extraction.

In short, the paper is not only about improving link prediction. It is about understanding what relation embeddings learn and how they can be used for reasoning.
