---
author: 
 - Jan Heiland & Peter Benner & Yongho Kim (MPI Magdeburg)
title: CNNs and Clustering for LD-LPV Approximations of Incompressible Navier-Stokes Equations
subtitle: Math+ -- Berlin -- September 2023
title-slide-attributes:
    data-background-image: pics/mpi-bridge.gif
parallaxBackgroundImage: pics/csc-en.svg
parallaxBackgroundSize: 1000px 1200px
bibliography: nn-nse-ldlpv-talk.bib
nocite: |
  @*
---

# Introduction 

$$\dot x = f(x) + Bu$$

---

## {data-background-video="pics/triple_swingup_slomo.MP4"}

. . .

::: {style="position: absolute; width: 60%; right: 0; box-shadow: 0 1px 4px rgba(0,0,0,0.5), 0 5px 25px rgba(0,0,0,0.2); background-color: rgba(0, 0, 0, 0.9); color: #fff; padding: 20px; font-size: 40px; text-align: left;"}

Control of an inverted pendulum

 * 9 degrees of freedom
 * but nonlinear controller.

:::

## {data-background-image="pics/dbrc-v_Re50_stst_cm-bbw.png"}

. . .

::: {style="position: absolute; width: 60%; right: 0; box-shadow: 0 1px 4px rgba(0,0,0,0.5), 0 5px 25px rgba(0,0,0,0.2); background-color: rgba(0, 0, 0, 0.9); color: #fff; padding: 20px; font-size: 40px; text-align: left;"}

Stabilization of a laminar flow

 * 50'000 degrees of freedom
 * but linear regulator.

:::

# LPV Representation

\begin{align}
\dot x -Bu & = f(x) \\
       & \approx [A_0+\rho_1(x)A_1+ \dotsm + \rho_r(x) A_r]\, x
\end{align}

---

The *linear parameter varying* (LPV) representation/approximation
$$
\dot x \approx  \bigl [\Sigma \,\rho_i(x)A_i \bigr]\, x + Bu
$$
for nonlinear controller can call on

 * a general structure (**linear** but parameter-varying)
 * and extensive theory (**LPV** controller design)

. . .


**The smaller the LPV parametrization the better.**


# How to Design an LPV approximation

A general procedure

---

If $f(0)=0$ and under mild conditions, the flow $f$ can be factorized
$$
f( x) = [A(x)]\,x
$$ 
with some $A\colon \mathbb R^{n} \to \mathbb R^{n\times n}$.

. . .

1. If $f$ has a strongly continuous Jacobian $\partial f$, then
$$
f(x) = [\int_0^1 \partial f(sx)\mathsf{d} s]\, x
$$
2. The trivial choice of
$$
f(x) = [\frac{1}{x^Tx}f(x)x^T]\,x
$$
doesn't work well -- neither do the improvements [@LinVL15].


---

For the factorization $f(x)=A(x)\,x$, one can say that

1. it is not unique
2. it can be a design parameter
3. often, it is indicated by the structure.

. . .

... like in the advective term in the *Navier-Stokes* equations:
$$
(v\cdot \nabla)v = \mathcal A_s(v)\,v.
$$

---

Now, we have an *state-dependent coefficient* representation

$$ f(x) = A(x)\,x.$$

. . .

## How to obtain an LPV representation/approximation?


---

## $\dot x = A(x)\,x + Bu$

 * Trivially, this is an LPV representation 
 $$
 \dot x = A(\rho(x))\, x + Bu
 $$
 with $\rho(x) = x$.

 * Take any model order reduction scheme that *encodes* (via $\mathcal P$) the state and *decodes* it (via $\mathcal L$) so that
 $$
 \tilde x = \mathcal L(\hat x) = \mathcal L (\mathcal P(x)) \approx x
 $$

. . .

 * Then $\rho = \mathcal P(x)$ gives a low-dimensional LPV approximation by means of
 $$
 A(x)\,x \approx A(\tilde x)\, x = A(\mathcal L \rho (x))\,x.
 $$

---

## Observation

   * If $x\mapsto A(x)$ itself is affine linear 
   * and $\mathcal L$ is linear, 
   * then
   $$
   \dot x \approx A(\mathcal L \rho(x))\,x + Bu = [A_0 + \sum_{i=1}^r \rho_i(x) A_i]\, x + Bu
   $$
   is **affine** with 

     * $\rho_i(x)$ being the components of $\rho(x)\in \mathbb R^r$ 
     * and constant matrices $A_0$, $A_1$, ..., $A_r \in \mathbb R^{n\times n}$.

## Intermediate Summary

 * Generally, a nonlinear $f$ can be factorized as $f(x) = A(x)\,x$.

 * Model order reduction provides a low dimensional LPV representation $A(x)\,x\approx A(\mathcal \rho(x))\,x$.

 * The needed affine-linearity in $\rho$ follows from system's structure (or from another layer of approximation.

 * We will look for **linear** decoding and possibly nonlinear encoding.

# (Convolutional) Autoencoders

## {data-background-image="pics/cw-Re60-t161-cm-bbw.png" data-background-size="cover"}

. . .

::: {style="position: absolute; width: 60%; right: 0; box-shadow: 0 1px 4px rgba(0,0,0,0.5), 0 5px 25px rgba(0,0,0,0.2); background-color: rgba(0, 0, 0, 0.9); color: #fff; padding: 20px; font-size: 40px; text-align: left;"}
The *Navier-Stokes* equations

$$
\dot v + (v\cdot \nabla) v- \frac{1}{\mathsf{Re}}\Delta v + \nabla p= f, 
$$

$$
\nabla \cdot v = 0.
$$
:::

---

## {data-background-image="pics/cw-Re60-t161-cm-bbw.png" data-background-size="cover"}

::: {style="position: absolute; width: 60%; right: 0; box-shadow: 0 1px 4px rgba(0,0,0,0.5), 0 5px 25px rgba(0,0,0,0.2); background-color: rgba(0, 0, 0, 0.9); color: #fff; padding: 20px; font-size: 40px; text-align: left;"}
Simulation model:

 * we use *finite elements* to obtain
 * the dynamical model of type

 $\quad \quad \quad \dot x = Ax + N(x)\,x,$

 * with $N$ linear in $x$
 * and a state dimension of about $n=50'000$.

:::

---


## Question

We want to reduce the state-dimension as much as possible -- 
can we do better than POD?

## {data-background-image="pics/scrsho-lee-cb.png"}

. . .

::: {style="position: absolute; width: 60%; right: 0; box-shadow: 0 1px 4px rgba(0,0,0,0.5), 0 5px 25px rgba(0,0,0,0.2); background-color: rgba(0, 0, 0, 0.9); color: #fff; padding: 20px; font-size: 40px; text-align: left;"}

Lee/Carlberg (2019): *MOR of dynamical systems on nonlinear manifolds using deep convolutional autoencoders*
:::

## {data-background-image="pics/scrsho-choi.png"}

. . .

::: {style="position: absolute; width: 60%; right: 0; box-shadow: 0 1px 4px rgba(0,0,0,0.5), 0 5px 25px rgba(0,0,0,0.2); background-color: rgba(0, 0, 0, 0.9); color: #fff; padding: 20px; font-size: 40px; text-align: left;"}

Kim/Choi/Widemann/Zodi (2020): *Efficient nonlinear manifold reduced order model*
:::

## Convolutional Autoencoders for NSE

1. Consider solution snapshots $v(t_k)$ as pictures.

2. Learn convolutional kernels to extract relevant features.

3. While extracting the features, we reduce the dimensions.

4. Encode $v(t_k)$ in a low-dimensional $\rho_k$.

## Our Example Architecture Implementation


## {data-background-image="pics/nse-cnn.jpg"}

. . .

::: {style="position: absolute; width: 60%; right: 0; box-shadow: 0 1px 4px rgba(0,0,0,0.5), 0 5px 25px rgba(0,0,0,0.2); background-color: rgba(0, 0, 0, 0.9); color: #fff; padding: 20px; font-size: 40px; text-align: left;"}

 * A number of convolutional layers for feature extraction and reduction

 * A full linear layer with nonlinear activation for the final encoding $\rho\in \mathbb R^{r}$

 * A linear layer (w/o activation) that expands $\rho \to \tilde \rho\in \mathbb R^{k}$.

:::

## Input:

 * Velocity snapshots $v_i$ of an FEM simulation with $$n=50'000$$ degrees of freedom

 * interpolated to two pictures with `63x95` pixels each.

 
## Side note: FEM aware training

For the loss functions, we implemented the correct FEM norms
$$
\| v_i - \tilde v_i\|_M
$$
and 

$$\| (v_i\cdot \nabla)v_i - (\tilde v_i \cdot \nabla )v_i\|_{M^{-1}}$$

with the mass matrix $M$ of the FEM discretization.

## Results

---

::: {style="position: absolute; width: 60%; right: 0; box-shadow: 0 1px 4px rgba(0,0,0,0.5), 0 5px 25px rgba(0,0,0,0.2); background-color: rgba(0, 0, 0, 0.9); color: #fff; padding: 20px; font-size: 40px; text-align: left;"}

Drag/lift phase portrait of 

 * the cylinder wake at $\mathsf{RE}=40$ with
 * convection parametrized as $N(v)\,v\approx \tilde N(\rho(v))\,v$ 
 * with $\rho(v(t))\in \mathbb R^3$ (in numbers: three)
 * obtained through POD and CNN

:::

---

## {data-background-image="pics/dlppt-cs3.svg" data-background-size="contain"}

# Clustering

---

**OBSERVATION**

For POD, the parametrization might be **good**, but the range of the decoding basis is **insufficient**.

. . .

**IDEA**

 * Decode $\rho \to \tilde v$ with local bases (or local decoders)
 * still, $\rho$ is a parametrization of $v$, however,
 * the decoding becomes **nonlinear** and (even **noncontinuous**).

## Our Approach

0. Train/compute the encoding: $v\to \rho$
1. Identify clusters $c_1,\dotsc,c_K$ in the values of $\rho$
2. On each cluster $c_k$ train/compute a decoder $\mathcal L_k\colon \rho \to \tilde v$
4. Decode by (1) assigning a cluster to $\rho(t)$ and (2) apply $\mathcal L_k$

## Our Results -- using 5 clusters

## {data-background-image="pics/cae-pod-cluster.png" data-background-size="contain"}

# Conclusion

 * Convolutional Neural Networks easily outperform POD at very low dimensions.

 * Further improvements possible through clustering.

## ... and Outlook

 * Controller design!

 * Proof of concept for nonlinear controller design with POD and truncated SDRE [@HeiW23].

 * Current work -- make clustering a smooth operation.

. . .

Thank You!

---

## References
