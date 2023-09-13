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

## Control of Nonlinear & Large-Scale Systems

A general approach would include

 * powerful backends (linear algebra / optimization)
 * exploitation of general structures
 * model order reduction
 * data-driven surrogate models
 * all of it?!


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

Spoiler: 

 In this talk, we will consider LPV series expansions of control laws.

---

## LPV system approaches

For linear parameter-varying systems
$$
\dot x = A(\rho(x))\,x + Bu
$$
there exist established methods that provide control laws based one

 * robustness against parameter variations [@PeaA01]
 * adaption with the parameter, i.e. *gain scheduling*, [@ApkGB95]

A major issue: require solutions of coupled LMI systems.


# SDRE series expansion

. . .

Consider the optimal regulator control problem

$$
\int_0^\infty \|y\|^2 + \alpha \|u\|^2\, \mathsf{d}s \to \min_{(y, u)}
$$
subject to 
$$
\dot x = A(\rho(x))\,x+Bu, \quad y=Cx.
$$

---

**Theorem** [@BeeTB00]

If there exists $\Pi$ as a function of $x$ such that
$$
\begin{aligned}
& \dot{\Pi}(x)+\bigl[\frac{\partial(A(\rho(x)))}{\partial x}\bigr]^T \Pi(x)\\
& \quad+\Pi(x) A(\rho(x))+A^T(\rho(x)) \Pi(x)-\frac{1}{\alpha} \Pi(x) BB^T \Pi(x)=-C^TC .
\end{aligned}
$$

Then $$u=-\frac{1}{\alpha}B^T\Pi(x)\,x$$ is an optimal feedback for the control problem.

---

In **Praxis**, parts of the HJB are discarded and we use $\Pi(x)$ that solely solves the state-dependent Riccati equation (SDRE)
$$
\Pi(x) A(\rho(x))+A^T(\rho(x)) \Pi(x)-\frac{1}{\alpha} \Pi(x) BB^T\Pi(x)=-C^TC,
$$
and the SDRE feedback
$$
u=-\frac{1}{\alpha}B^T\Pi(x)\,x.
$$

* numerous application examples and
* proofs of performance [@BanLT07]
* also beyond smallness conditions [@BenH18]

---

* Although the SDRE is an approximation already,

* the repeated solve of the Riccati equation is not feasible.

---

* However, for affine LPV systems, a series expansion 

* enables an efficient approximation at runtime.




---

## The series expansion

We note that $\Pi$ depends on $x$ through $A(\rho(x))$. 

Thus, we can consider $\Pi$ as a function in $\rho$ and its corresponding multivariate Taylor expansion up to order $K$
\begin{equation} \label{eq:taylor-expansion-P}
  \Pi (\rho) \approx \Pi (0) + \sum_{1\leq |\beta| \leq K} 
    \rho^{(\beta)}P_{\beta},
\end{equation}
where

* $\beta=(\beta_1, \dotsc, \beta_r)\in \mathbb N^r$ is a multiindex and the
* $P_{\beta}\in \mathbb R^{n\times n}$ are **constant** matrices.

---

**Theorem** 

If $A(\rho)$ is affine, i.e. $A(\rho) = A_0 + \sum_{k=1}^r \rho_k A_k$.

. . .

Then the coefficients of the first order Taylor approximation
  $$
  \Pi (\rho) \approx \Pi(0) + \sum_{|\beta| = 1}  \rho^{(\beta)}P_{\beta} =: P_0 +
  \sum_{k=1}^r \rho_k L_k.
  $$
are the solutions to

* $A_{0}^{T} P_{0} + P_{0} A_{0} - P_{0} B B^{T} P_{0} = -C^{T} C$,

and, for $k=1,\dotsc,r$,

* $(A_{0} - B B^{T} P_{0})^{T} L_{k} + L_{k} ( A_{0} - B B^{T} P_{0} )= -(A_{k}^{T} P_{0} + P_{0} A_{k})$.


---

**Proof**

Insert the Taylor expansion of $\Pi$ and the LPV representation of $A$ into the SDRE and *match the coefficients*. 

**Corollary**

The corresponding nonlinear feedback is realized as
$$
u = -\frac{1}{\alpha}B^T[P_0 + \sum_{k=1}^r \rho_k(x) L_k]\,x.
$$

. . .

Cp., e.g.,  [@BeeTB00] and [@AllKS23].


## Intermediate Summary

A representation/approximation of the nonlinear system via
$$
\dot x = [A_0 + \sum_{k=1}^r \rho_k(x) A_k]\, x + Bu
$$
enables the nonlinear feedback design through truncated expansions of the SDRE.



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
(v\cdot \nabla)v = \mathcal A_s(v)\,v
$$
with $s\in[0,1]$ and the linear operator $\mathcal A_s(v)$ defined via 
$$\mathcal A_s(v)\,w := s\,(v\cdot \nabla)w + (1-s)\, (w\cdot \nabla)v.$$

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

 * Take any model order reduction scheme that compresses (via $\mathcal P$) the state and lifts it back (via $\mathcal L$) so that
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

## The Algorithm

Nonlinear controller design for 
$$
\dot x = f(x) + Bu
$$
by LPV approximations and truncated SDRE expansions.

. . .

1. Compute an affine LPV approximative model with 
$$f(x)\approx A_0x +  \sum_{k=1}^r \rho_k(x)A_kx.$$

2. Solve one *Riccati* and $r$ *Lyapunov* equations for $P_0$ and the $L_k$s.
3. Close the loop with $u = -\frac{1}{\alpha}B^T[P_0x + \sum_{k=1}^r \rho_k(x) L_kx ].$

## Step-1 -- Compute the LPV Approximation

We use POD coordinates with the matrix $V\in \mathbb R^{n\times r}$ of POD modes $v_k$

 * $\rho(x) = V^T x$, 

 * $\tilde x = V\rho(x)=\sum_{k=1}^r\rho_i(x)v_k.$

. . .

Then:
$$N(x,x)\approx N(\tilde x, x) = N(\sum_{k=1}^r\rho_i(x)v_k, x) = \sum_{k=1}^r\rho_i(x) N(v_k, x) $$
which is readily realized as
$$ [\sum_{k=1}^r\rho_i(x) A_k]\,x.$$

## Step-2 -- Compute $P_0$ and the $L_k$s

This requires solutions of large-scale ($n=50'000$) matrix equations

1. Riccati -- nonlinear but fairly standard
2. Lyapunovs -- linear but indefinite.

We use state-of-the-art low-rank ADI iterations (ask Steffen for details).


---

## Step-3 -- Close the Loop {data-background-image="pics/cw-v-Re60-stst-cm-bbw.png" data-background-size="cover"}


---

* Setup: Start from the steady-state
* Goal: Stabilize the steady-state

Comparison of feedback designs

 * `LQR` -- plain LQR controller
 * `xSDRE-r` -- truncated (at `r`) SDRE feedback



---

## Parameters of the Control Setup

We check the performance with respect to two parameters

 * $\alpha$ ... the regularization parameter that penalizes the control

 * $t_{\mathsf c} > 0$ ... time before the controller is activated 

. . .

...

 * The parameter $t_c$ describes the domain of attraction.

 * For `r=0` the `xSDRE-r` feedback recovers the `LQR` feedback.


<!-- \in \{10^{p} \colon p = -2, -1, 0, 1, 2, 3\}
span in which the controller is idle and a test signal is applied to trigger the instabilities
-->

---

## {data-background-image="pics/Re60-sut1250-fbs-lqg1e+03.png" data-background-size="cover"}

. . .

::: {style="position: absolute; width: 60%; right: 0; box-shadow: 0 1px 4px rgba(0,0,0,0.5), 0 5px 25px rgba(0,0,0,0.2); background-color: rgba(0, 0, 0, 0.9); color: #fff; padding: 20px; font-size: 40px; text-align: left;"}

Norm plot of the feedback signals.

* `LQR` fails to stabilize
* increasing `r` means better performance
* stability achieved at `r=10`

:::

---

## {data-background-image="pics/Re60-sut6500-fbs-lqg1e+00.png" data-background-size="cover"}

. . .

::: {style="position: absolute; width: 60%; right: 0; box-shadow: 0 1px 4px rgba(0,0,0,0.5), 0 5px 25px rgba(0,0,0,0.2); background-color: rgba(0, 0, 0, 0.9); color: #fff; padding: 20px; font-size: 40px; text-align: left;"}

Less regularization 

* less smooth feedback actions
* again `LQR` fails
* `xSDRE` can achieve stability
* stability achieved for certain `r`

:::

---

## The Full Picture{data-background-image="pics/parametermap.png" data-background-size="contain"}

--- 

## {data-background-image="pics/parametermap.png" data-background-size="contain"}


---

## Conclusion for the Numerical Results

* Measurable and reliable improvements with respect to $\alpha$

  * *more performant feedback action at higher regularization*

. . .

* no measurable performance gain with respect to $t_{\mathsf c}$

  * *no extension of the domain of attraction*

. . .

* still much space for improvement

  * find better bases for the parametrization?
  * increase the `r`?
  * second order truncation of the SDRE?


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
