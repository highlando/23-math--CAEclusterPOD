---
author: 
 - Jan Heiland & Peter Benner & Steffen Werner (MPI Magdeburg)
title: Low-dimensional LPV approximations for nonlinear control
subtitle: Blacksburg -- May 2023
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
for nonlinear controller comes with

 * a general structure (**linear** but parameter-varying)

and extensive theory on

 * LPV controller design

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

 * The needed affine-linearity in $\rho$ follows from system's structure (or from another layer of approximation (see, e.g, [@KoeT20]).

 * We will look for **linear** decoding and possibly nonlinear encoding.

# Numerical Realization

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
Control Problem:

 * use two small outlets for fluid at the cylinder boundary
 * to stabilize the unstable steady state
 * with a few point observations in the wake.

:::

---

## {data-background-image="pics/cw-Re60-t161-cm-bbw.png" data-background-size="cover"}

::: {style="position: absolute; width: 60%; right: 0; box-shadow: 0 1px 4px rgba(0,0,0,0.5), 0 5px 25px rgba(0,0,0,0.2); background-color: rgba(0, 0, 0, 0.9); color: #fff; padding: 20px; font-size: 40px; text-align: left;"}
Simulation model:

 * we use *finite elements* to obtain
 * the dynamical model of type

 $\dot x = Ax + N(x,x) + Bu, \quad y = Cx$

 * with $N$ being bilinear in $x$
 * and a state dimension of about $n=50'000$.

:::

---

## Low-dimensional LPV

**Approximation** of *Navier-Stokes Equations* by *Convolutional Neural Networks*

---


## {data-background-image="pics/cw-Re60-t161-cm-bbw.png" data-background-size="cover"}

. . .

::: {style="position: absolute; width: 60%; right: 0; box-shadow: 0 1px 4px rgba(0,0,0,0.5), 0 5px 25px rgba(0,0,0,0.2); background-color: rgba(0, 0, 0, 0.9); color: #fff; padding: 20px; font-size: 40px; text-align: left;"}
The *Navier-Stokes* equations

$$
\dot v + (v\cdot \nabla) v- \frac{1}{\Re}\Delta v + \nabla p= f, 
$$

$$
\nabla \cdot v = 0.
$$
:::

---

* Let $v$ be the velocity solution and let
$$
V =
\begin{bmatrix}
V_1 & V_2 & \dotsm & V_r
\end{bmatrix}
$$
be a, say, *POD* basis with $$v(t) \approx VV^Tv(t)=:\tilde v(t),$$

* then $$\rho(v(t)) = V^Tv(t)$$ is a parametrization.

---

* And with $$\tilde v = VV^Tv = V\rho = \sum_{k=1}^rV_k\rho_k,$$

* the NSE has the low-dimensional LPV representation via
$$
(v\cdot \nabla) v \approx (\tilde v \cdot \nabla) v = [\sum_{k=1}^r\rho_k(V_k\cdot \nabla)]\,v.
$$

## Question

Can we do better than POD?

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

## Convolution Autoencoders for NSE

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

 * interpolated to two pictures with `63x95` pixels each

 * makes a `2x63x69` tensor. 

## Training for minimizing:
$$
\| v_i - VW\rho(v_i)\|^2_M
$$
which includes

 1. the POD modes $V\in \mathbb R^{n\times k}$,

 2. a learned weight matrix $W\in \mathbb R^{k\times r}\colon \rho \mapsto \tilde \rho$,

 3. the mass matrix $M$ of the FEM discretization.

## Going PINN

Outlook: 
the induced low-dimensional affine-linear LPV representation of the convection
$$\| (v_i\cdot \nabla)v_i - (VW\rho_i \cdot \nabla )v_i\|^2_{M^{-1}}$$
as the target of the optimization.

Implementation issues:

 * Include FEM operators while
 * maintaining the *backward* mode of the training.

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

# Conclusion

## ... and Outlook

 * General approach to model **structure** reduction by low-dimensional affine LPV systems.

 $$f(x) \quad \to\quad  A(x)\,x\quad  \to\quad  \tilde A(\rho(x))\,x\quad  \to\quad  [A_0 + \sum_{k=1}^r\rho_k(x)A_k]\,x$$

 * Proof of concept for nonlinear controller design with POD and truncated SDRE [@HeiW23].

 * General and performant but still heuristic approach.

. . .

* Detailed roadmap for developing the LPV (systems) theory is available.

* PhD student wanted!

. . .

Thank You!

---

## References
