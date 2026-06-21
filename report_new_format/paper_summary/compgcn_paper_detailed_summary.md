# Detailed Summary: Composition-based Multi-Relational Graph Convolutional Networks (CompGCN)

**Paper:** *Composition-based Multi-Relational Graph Convolutional Networks*  
**Authors:** Shikhar Vashishth, Soumya Sanyal, Vikram Nitin, Partha Talukdar  
**Venue:** ICLR 2020  

---

## 1. High-level Overview

This paper proposes **CompGCN**, a graph convolutional framework for **multi-relational graphs**, especially knowledge graphs. A multi-relational graph contains edges with both **relation labels** and **directions**, for example `(Christopher Nolan, Born-in, London)`.

The main problem addressed by the paper is that many existing Graph Convolutional Network (GCN) methods are designed for simple undirected graphs, while real-world graphs often contain many relation types. Existing relational GCN methods usually suffer from two limitations:

1. **Over-parameterization:** If each relation has its own weight matrix, the number of parameters grows quickly as the number of relations increases.
2. **Node-only representation learning:** Many relational GCNs learn embeddings only for nodes/entities, but not for relations. This is problematic for tasks like **knowledge graph link prediction**, where relation embeddings are also needed.

CompGCN solves this by jointly learning **node embeddings** and **relation embeddings**. Instead of using a separate transformation matrix for every relation, it composes a neighboring node embedding with the relation embedding on the edge connecting them. This makes the message passing process relation-aware while remaining parameter-efficient.

---

## 2. Motivation

### 2.1 Why ordinary GCNs are insufficient

Standard GCNs operate on graphs where edges are usually untyped and undirected. The update for a node aggregates transformed features from its neighbors. However, in a knowledge graph, edges have meaning: `Born-in`, `Directed-by`, `Citizen-of`, etc. Treating all edges as identical loses important semantic information.

### 2.2 Why earlier relational GCNs are limited

Earlier multi-relational GCNs extend GCNs by using relation-specific transformations. For example, Relational-GCN uses different parameters for each relation type. This captures relation information, but it can become expensive when the graph has many relations.

Another limitation is that many methods update only entity/node embeddings. They do not naturally update relation embeddings, even though relations are central to multi-relational graphs and necessary for link prediction.

### 2.3 Key idea of CompGCN

CompGCN borrows the idea of **entity-relation composition** from knowledge graph embedding methods. Instead of sending only the neighbor node embedding as a message, CompGCN sends a composed representation of:

```text
neighbor node embedding + relation embedding
```

This composition can be implemented using operations such as subtraction, multiplication, or circular correlation.

---

## 3. Background

## 3.1 Standard GCN on undirected graphs

Given a graph:

```text
G = (V, E, X)
```

where:

- `V` is the set of nodes,
- `E` is the set of edges,
- `X` is the node feature matrix,

standard GCN computes node representations as:

```text
H = f(A_hat X W)
```

where:

- `A_hat` is the normalized adjacency matrix with self-loops,
- `W` is a learnable weight matrix,
- `f` is a non-linear activation function.

Stacking multiple GCN layers allows the model to capture multi-hop neighborhood information.

## 3.2 GCN on multi-relational graphs

For a multi-relational graph:

```text
G = (V, R, E, X)
```

where each edge is a triple:

```text
(u, v, r)
```

meaning that relation `r` goes from node `u` to node `v`.

A direct relational GCN update may use a relation-specific weight matrix:

```text
h_v = f( sum over neighbors W_r h_u )
```

However, this can lead to over-parameterization because every relation may require its own transformation matrix.

---

# 4. Method: CompGCN

The method section is the core contribution of the paper. CompGCN introduces a relation-aware message passing mechanism where both node and relation embeddings are updated.

---

## 4.1 Graph construction with inverse edges and self-loops

Following earlier directed relational GCN work, CompGCN allows information to flow in both directions along a directed edge.

For every original edge:

```text
(u, v, r)
```

CompGCN adds an inverse edge:

```text
(v, u, r^{-1})
```

It also adds a self-loop edge:

```text
(u, u, self)
```

Therefore, the relation set is expanded to include:

1. Original relations
2. Inverse relations
3. Self-loop relation

This is important because message passing should distinguish between:

- original outgoing relation direction,
- inverse incoming relation direction,
- self-loop information.

---

## 4.2 Joint node and relation embeddings

Unlike many previous GCN-based approaches, CompGCN learns embeddings for both:

```text
nodes/entities: h_v
relations: h_r
```

This is important for knowledge graphs because relations are not just labels; they carry semantic meaning. For example, `Born-in` and `Directed-by` should influence message passing differently.

The relation embeddings can be initialized from relation features if available. If relation features are not available, they can be learned as parameters.

---

## 4.3 Relation-based composition

The central idea is to compose a neighbor node embedding with the relation embedding on the connecting edge.

For a triple:

```text
(subject, relation, object)
```

CompGCN uses a composition function:

```text
phi(e_s, e_r) = e_o-like representation
```

where:

- `e_s` is the subject/entity embedding,
- `e_r` is the relation embedding,
- `phi` is a composition operator.

The paper evaluates three non-parametric composition operations:

### 4.3.1 Subtraction

Inspired by TransE:

```text
phi(e_s, e_r) = e_s - e_r
```

This reflects a translation-style view of relations.

### 4.3.2 Multiplication

Inspired by DistMult:

```text
phi(e_s, e_r) = e_s * e_r
```

where `*` is element-wise multiplication.

This allows the relation to act like a feature-wise gate on the entity embedding.

### 4.3.3 Circular correlation

Inspired by HolE:

```text
phi(e_s, e_r) = e_s ? e_r
```

where `?` denotes circular correlation.

This operation can capture richer interactions between the entity and relation embeddings than simple subtraction or multiplication.

The authors note that CompGCN is generic: future or more powerful knowledge graph composition operators can also be plugged into the framework.

---

## 4.4 CompGCN node update equation

For a node `v`, CompGCN aggregates messages from its neighbors. Each message is produced by composing the neighbor embedding with the relation embedding, then applying a direction-specific transformation.

The update is:

```text
h_v = f( sum_{(u,r) in N(v)} W_{lambda(r)} phi(x_u, z_r) )
```

where:

- `N(v)` is the neighborhood of node `v`,
- `x_u` is the initial or previous-layer representation of neighbor node `u`,
- `z_r` is the representation of relation `r`,
- `phi(x_u, z_r)` composes the neighbor and relation embeddings,
- `W_{lambda(r)}` is a direction-specific weight matrix,
- `f` is an activation function.

The transformation matrix depends on the direction/type of the edge:

```text
W_O : original relation direction
W_I : inverse relation direction
W_S : self-loop
```

So the model does not need one full matrix per relation. Instead, it uses only direction-specific matrices and relation embeddings.

This is one of the main reasons CompGCN is more parameter-efficient.

---

## 4.5 Relation update equation

CompGCN also updates relation embeddings. After a node update layer, relation embeddings are transformed using:

```text
h_r = W_rel z_r
```

where:

- `z_r` is the current relation embedding,
- `W_rel` is a learnable transformation matrix.

For multiple stacked CompGCN layers, the relation update becomes:

```text
h_r^{k+1} = W_rel^k h_r^k
```

This allows relation embeddings to evolve layer by layer together with node embeddings.

---

## 4.6 Multi-layer CompGCN

For a stacked CompGCN with `k` layers, the node update is:

```text
h_v^{k+1} = f( sum_{(u,r) in N(v)} W_{lambda(r)}^k phi(h_u^k, h_r^k) )
```

and the relation update is:

```text
h_r^{k+1} = W_rel^k h_r^k
```

Thus, each layer updates both nodes and relations.

The use of `phi(h_u^k, h_r^k)` means that messages are relation-aware at every layer.

---

## 4.7 Basis decomposition for scalability

To scale to graphs with many relations, CompGCN uses a basis-vector formulation for relation embeddings.

Instead of learning a separate embedding for every relation independently, each relation embedding is represented as a linear combination of basis vectors:

```text
z_r = sum_{b=1}^{B} alpha_{br} v_b
```

where:

- `B` is the number of basis vectors,
- `v_b` is a learnable basis vector,
- `alpha_{br}` is a learnable scalar coefficient for relation `r` and basis `b`.

This reduces the number of relation parameters and helps the model scale when the relation vocabulary is large.

The paper emphasizes that this differs from R-GCN basis decomposition. R-GCN decomposes relation-specific matrices, while CompGCN decomposes relation embeddings as vectors. Later layers share relation information through the learned relation transformations.

---

## 4.8 How CompGCN generalizes earlier GCN models

The paper shows that CompGCN can reduce to several existing GCN variants by choosing specific forms of the weight matrices and composition function.

### 4.8.1 Standard Kipf-GCN

If the model ignores relation type and uses:

```text
W_{lambda(r)} = W
phi(h_u, h_r) = h_u
```

then CompGCN becomes a standard GCN.

### 4.8.2 Directed-GCN

If the model uses direction-specific weights but ignores relation embeddings in the composition, then it behaves like Directed-GCN.

### 4.8.3 Relational-GCN

If the model uses a separate relation-specific matrix `W_r` and ignores relation embeddings in the composition, then it becomes similar to R-GCN.

### 4.8.4 Weighted-GCN

If the composition is equivalent to multiplying the neighbor representation by a relation-specific scalar, then it becomes similar to Weighted-GCN.

This generalization argument is important because it shows that CompGCN is not just another isolated architecture. It is a broader framework that contains several earlier models as special cases.

---

# 5. Link Prediction with CompGCN

For link prediction, CompGCN is used as an encoder. It produces entity embeddings and relation embeddings. These embeddings are then passed to a knowledge graph scoring function.

The paper evaluates CompGCN with three scoring functions:

1. **TransE**
2. **DistMult**
3. **ConvE**

The general structure is:

```text
Knowledge graph -> CompGCN encoder -> entity/relation embeddings -> score function -> triple score
```

This is different from GCN encoders that only output entity embeddings. Since CompGCN also updates relation embeddings, it is naturally suitable for link prediction.

---

# 6. Experiments

The paper evaluates CompGCN on three types of tasks:

1. Link prediction
2. Node classification
3. Graph classification

---

## 6.1 Link prediction datasets

The paper uses:

### FB15k-237

A pruned version of FB15k where inverse relations are removed to avoid direct leakage.

### WN18RR

A pruned version of WN18 derived from WordNet, also designed to remove inverse relation leakage.

Metrics include:

- Mean Reciprocal Rank (MRR)
- Mean Rank (MR)
- Hits@1
- Hits@3
- Hits@10

---

## 6.2 Node classification datasets

The paper uses:

### MUTAG Node

A relational dataset about complex molecules. The task is to classify whether a molecule is carcinogenic.

### AM

A dataset from the Amsterdam Museum. The task is to predict the category of an artifact based on graph links and attributes.

---

## 6.3 Graph classification datasets

The paper uses:

### MUTAG Graph

A bioinformatics dataset of mutagenic compounds. The task is binary graph classification.

### PTC

A dataset of chemical compounds labeled by carcinogenicity.

---

# 7. Main Results

## 7.1 Link prediction performance

CompGCN achieves strong performance on both FB15k-237 and WN18RR.

On FB15k-237, it outperforms previous methods on 4 out of 5 metrics. On WN18RR, it outperforms previous methods on 3 out of 5 metrics.

The best-performing setup is generally:

```text
ConvE + CompGCN with circular correlation composition
```

This suggests that using a powerful decoder like ConvE together with relation-aware CompGCN embeddings is highly effective.

---

## 7.2 Effect of different GCN encoders

The paper compares different GCN encoders combined with different scoring functions.

Main observation:

- Using GCN encoders usually improves link prediction.
- CompGCN improves performance more consistently than Directed-GCN, R-GCN, and Weighted-GCN.
- CompGCN performs well across TransE, DistMult, and ConvE scoring functions.

The reason is that CompGCN jointly learns entity and relation embeddings, while many competing encoders only update entity embeddings.

---

## 7.3 Effect of composition operator

The paper compares subtraction, multiplication, and circular correlation.

Key findings:

- With DistMult scoring, multiplication works best.
- With ConvE scoring, circular correlation works best.
- Circular correlation generally performs very well and often surpasses simpler operators.

This shows that the choice of composition operator matters. More expressive composition functions can improve relational message passing.

---

## 7.4 Scalability with relation basis vectors

The paper studies whether CompGCN remains effective when relation embeddings are represented using a limited number of basis vectors.

Findings:

- Performance improves as the number of basis vectors increases.
- With around 100 basis vectors, performance becomes comparable to using separate embeddings for all relations.
- Even with fewer basis vectors, CompGCN remains competitive.

This supports the claim that CompGCN scales well with increasing numbers of relations.

---

## 7.5 Scalability with number of relations

The authors create pruned versions of FB15k-237 by keeping only the top `m` most frequent relations, where:

```text
m = 10, 25, 50, 100, 237
```

They show that CompGCN with only 5 relation basis vectors performs comparably to the full model and consistently outperforms R-GCN.

This suggests that the basis-vector strategy is effective and parameter-efficient.

---

## 7.6 Node classification results

CompGCN outperforms the baselines on node classification.

On MUTAG Node and AM, it achieves better accuracy than methods such as:

- RDF2Vec
- R-GCN
- Weighted-GCN
- SynGCN

This demonstrates that relation-aware composition is useful beyond link prediction.

---

## 7.7 Graph classification results

For graph classification, CompGCN performs competitively.

It achieves strong results on PTC and comparable results on MUTAG Graph. The paper uses average pooling over node embeddings to obtain graph-level representations.

---

# 8. Why CompGCN Works

CompGCN is effective because it combines strengths from two areas:

## 8.1 From GCNs

It uses neighborhood aggregation, so each node representation captures graph structure and multi-hop context.

## 8.2 From knowledge graph embeddings

It uses entity-relation composition, so messages are not relation-agnostic. Each neighbor contributes information conditioned on the relation connecting it to the target node.

## 8.3 Parameter efficiency

Instead of assigning a full matrix to every relation, CompGCN uses:

- direction-specific matrices,
- relation embeddings,
- optional basis decomposition.

This reduces parameter growth while still preserving relation-specific semantics.

## 8.4 Joint relation learning

Updating relation embeddings allows CompGCN to support tasks where relations are central, especially link prediction.

---

# 9. Important Figures and Tables

## Figure 1: Overview of CompGCN

Figure 1 illustrates the core message passing process. For a central node such as `Christopher Nolan`, CompGCN composes each neighboring node embedding with the relation embedding on the edge. The composed message is then transformed using direction-specific filters for original and inverse edges. Messages are aggregated to update the central node. Relation embeddings are also transformed separately.

## Table 1: Comparison with other GCN methods

Table 1 compares GCN, Directed-GCN, Weighted-GCN, Relational-GCN, and CompGCN. CompGCN is the most complete framework because it models:

- node embeddings,
- edge directions,
- relation embeddings,
- relation-aware message passing,
- parameter efficiency.

## Table 2: Reduction to existing methods

Table 2 shows how different choices of composition function and weight matrix reduce CompGCN to earlier GCN models.

## Table 3: Link prediction results

Table 3 shows that CompGCN achieves state-of-the-art or near state-of-the-art results on FB15k-237 and WN18RR.

## Table 4: Encoder and composition comparison

Table 4 shows that CompGCN improves across different scoring functions and that circular correlation with ConvE gives the best overall result.

## Table 5: Node and graph classification results

Table 5 shows that CompGCN performs strongly on node classification and competitively on graph classification.

---

# 10. Strengths

1. **Joint node and relation embedding**  
   Unlike many GCN methods, CompGCN updates both entities and relations.

2. **Generic framework**  
   It can use different composition operators from knowledge graph embedding methods.

3. **Parameter-efficient**  
   It avoids full relation-specific transformation matrices.

4. **Scalable**  
   Basis decomposition allows it to handle many relations.

5. **Strong empirical performance**  
   It performs well on link prediction, node classification, and graph classification.

6. **Unifies earlier GCN models**  
   Several existing relational GCNs can be seen as special cases of CompGCN.

---

# 11. Limitations

1. **Composition operator selection matters**  
   The best operator depends on the downstream scoring function and dataset.

2. **Only non-parametric composition operators are studied deeply**  
   The paper mentions that parameterized operators such as NTN or ConvE-style composition could be explored, but leaves this for future work.

3. **Relation updates are relatively simple**  
   Relation embeddings are transformed by a linear matrix. More complex relation update mechanisms may further improve performance.

4. **Graph classification gains are less dramatic**  
   The strongest improvements are in link prediction and node classification. Graph classification results are competitive but not always best.

5. **Transductive setting**  
   Like many knowledge graph embedding methods, the main link prediction setup assumes known entity and relation sets during training.

---

# 12. Relevance to Code Retrieval / Code-Text Graph Modeling

This paper is relevant to code retrieval if we model code, queries, functions, APIs, variables, documentation, and dependencies as a multi-relational graph.

For example, possible relation types in a code retrieval graph could include:

```text
query --describes--> function
function --calls--> function
function --uses--> API
function --contains--> variable
docstring --documents--> function
function --belongs_to--> class
```

A CompGCN-style model could learn embeddings for both nodes and relations. This is useful because relation semantics matter. For instance, `calls`, `uses`, and `documents` should not be treated as the same type of connection.

The key idea that could transfer to code retrieval is:

```text
message = compose(neighbor representation, relation representation)
```

This would allow a code representation to aggregate information differently depending on whether the neighboring node is connected by a semantic relation, structural relation, dependency relation, or natural-language relation.

Potential applications:

1. **Query-code retrieval**  
   Learn representations of queries and code entities through a heterogeneous relation graph.

2. **Multi-positive retrieval**  
   Different queries may connect to the same code through different relation types. Relation embeddings could help model these multiple relevance paths.

3. **Code knowledge graph construction**  
   CompGCN can help encode a graph containing functions, classes, APIs, comments, and natural language descriptions.

4. **Cross-language code retrieval**  
   Language-specific and language-agnostic relations could be represented separately.

5. **Graph-enhanced generative retrieval**  
   A generative retriever could use CompGCN embeddings as structured supervision or as auxiliary representations.

---

# 13. Concise Takeaway

CompGCN is a relation-aware GCN framework for multi-relational graphs. Its main innovation is to compose neighbor node embeddings with relation embeddings during message passing. This allows the model to jointly learn node and relation representations, avoid over-parameterization, and improve performance on knowledge graph link prediction, node classification, and graph classification. The method is especially useful when edge labels carry important semantics, which makes it relevant for knowledge graphs and potentially for code retrieval graphs.
