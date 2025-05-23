# pyDbar for Electrical Impedance Tomography

This is a fork of the [original `pydbar`](https://github.com/NablaIP/pydbar) implementation by NablaIP. The main updates are as follows:
- Modernize: the last commit in the original repo was made in 2021, so some of the code had to be updated to follow modern conventions.
- Documentation: remove unnecessary components from `readme` and clarify documentation.

## Related work
- [pyEIT](https://github.com/liubenyuan/pyEIT): Python-based toolkit for EIT
- [Matlab implementation of d-bar method](https://wiki.helsinki.fi/display/mathstatHenkilokunta/EIT+with+the+D-bar+method%3A+discontinuous+heart-and-lungs+phantom)

## Overview
The D-bar method can be used for Electrical Impedance Tomography in a 2D framework. This method is based on the theoretical reconstruction procedure of Nachman and a regularizing strategy that was established by Siltanen and provides a natural framework for numerical implementation

## Algorithm Review
Here we explain the main steps of the Dbar algorithm with either direct numerical implementation or with theoretical motivation. We assume that the system we use applies current and measures voltages along a (finite) set of electrodes. For convenience, we also assume that the body of interest is a disk of radius $r = 1$.

### Neumann-to-Dirichlet (ND) Map
Let $L$ be the number of electrodes and let $\mathbf{\tilde{j}}^k \in \R^L$ denote the $k$<sup>th</sup> **current pattern** vector such that $\mathbf{\tilde{j}}^k = \begin{bmatrix}\tilde{j}^k_1 & \ldots & \tilde{j}^k_L\end{bmatrix}^\intercal$. By Kirchoff's current law (KCL),
$$
    \sum_{\ell=1}^L \tilde{j}^k_\ell = 0,
$$
so the set of valid current patterns live in an $L-1$-dimensional linear subspace of $\R^L$. That is, we can apply at most $L-1$ linearly independent current patterns since the $L$<sup>th</sup> value would be linearly dependent in order to satisfy KCL. Then 
$$
    k \in \{1, 2, \ldots, N \leq L-1\}.
$$

There are a few choices that can be considered for current patterns, but for now we focus on three:
- Adjacent current patterns: 
$$
    \tilde{j}_\ell^k =
    \begin{cases}
        M,\ k = \ell \\
        -M,\ k = \ell + 1 \\
        0,\ \text{otherwise}
    \end{cases},
    \qquad M \in \R^+
$$

- Trigonometric current patterns (by [Isaacson et al.](https://ieeexplore.ieee.org/document/1309705)): 
$$
    \tilde{j}_\ell^k = M
    \begin{cases}
        \cos(k \theta_\ell),\ k = 1, \ldots, \frac{L}{2} - 1 \\
        \cos(\pi\ell),\ k = \frac{L}{2} \\
        \sin((k - \frac{L}{2})\theta_\ell),\ k = \frac{L}{2} + 1, \ldots, L-1
    \end{cases},
    \qquad M \in \R^+,
$$
where $\theta_\ell = \frac{2\pi\ell}{L}$, the angle of the center of the $\ell$<sup>th</sup> electrode.

- Rotated trigonometric patterns (optimal by [Demidenko et al.](https://ieeexplore.ieee.org/document/1386561))
$$
    \tilde{j}_\ell^k = M
    \begin{cases}
        \cos(k \theta_\ell),\ k = 1, \ldots, \frac{L}{2} - 1 \\
        Q \cos(k \theta_\ell),\ k = \frac{L}{2} \\
        \sin((k - \frac{L}{2})\theta_\ell),\ k = \frac{L}{2} + 1, \ldots, L-1
    \end{cases},
$$
where $\theta_0$ is some offset such that $\theta_0 L \neq \pm \pi$,
$$
    \theta_\ell = \theta_0 + \frac{2\pi\ell}{L},
    \qquad
    M = \sqrt{2 \over L},
    \qquad \text{and} \qquad
    Q = \frac{1}{\sqrt{2} \cos(\theta_0 \frac{L}{2})}.
$$

For each current pattern $\mathbf{\tilde{j}}^k$, we obtain a voltage pattern $\mathbf{\tilde{v}}^k \in \R^L$. Since the resultant voltages are only determined up to an additive constant, for mathematical convenience, we enforce that $\sum_\ell v_\ell^k = 0$. This follows common practice in EIT. In our implementation, we don't expect the incoming data to have such property, so we enforce is when needed, in the `Mapper` object.

To numerically obtain the ND map, `Mapper` performs the following:
#### 1. Zero-mean the voltages:
Compute sample mean
$$
    \mu^k = \frac{1}{L}\sum_{\ell = 1}^L \tilde{v}^k_\ell.
$$
For each $\ell = 1,\ldots,N$,
$$
    \hat{v}^k_\ell = \tilde{v}^k_\ell - \mu^k
    \implies
    \mathbf{\hat{v}}^k = \begin{bmatrix}
        \hat{v}^k_1 & \ldots & \hat{v}^k_L
    \end{bmatrix}^\intercal
$$

#### 2. Normalize measurements:
$$
\mathbf{j}^k = \frac{\mathbf{\tilde{j}}^k}{\|\mathbf{\tilde{j}}^k\|_2}
\qquad \text{and} \qquad
\mathbf{v}^k = \frac{\mathbf{\hat{v}}^k}{\|\mathbf{\tilde{j}}^k\|_2},
$$
where $\mathbf{j}^k$ and $\mathbf{v}^k$ are normalized currents and voltages we use moving forward.

#### 3. Define the discrete approximation of the ND map $\mathbf{\tilde{R}}_\sigma \in \R^{N \times N}$
First, for convenience, define
$$
    \mathbf{R}_\sigma(m, n) = \langle \mathbf{j}^m, \mathbf{v}^n \rangle_L,
$$
where
$$
    \langle u(\cdot), w(\cdot) \rangle_L = \sum_{\ell = 1}^{L} \overline{u(\theta_\ell)} w(\theta_\ell).
$$
Then let
$$
\begin{align*}
    \mathbf{\tilde{R}}_\sigma(m, n) &= \langle \frac{\mathbf{j}^m}{A_\ell}, \mathbf{v}^n \rangle_L \\
                            &= \sum_{\ell = 1}^{L} \frac{\overline{j^m_\ell}}{A_\ell} \cdot v^n_\ell,
\end{align*}
$$
where $1 \leq m, n \leq N$ and $A_\ell$ is the area of the $\ell$<sup>th</sup> electrode. Most likely, $\forall \ell,\ A_\ell = A$, so we can simplify the above to
$$
    \mathbf{\tilde{R}}_\sigma(m, n)
    = \frac{1}{A} \sum_{\ell = 1}^{L} \overline{j^m_\ell} \cdot v^n_\ell
    = \frac{\mathbf{R}_\sigma}{A}.
$$

### Dirichlet-to-Neumann (DN) Map
Having obtained our ND map approximation $\mathbf{\tilde{R}}_\sigma$, we simply invert it to obtain the approximation to the Dirichlet-to-Neumann map:
$$
    \mathbf{L}_\sigma := \mathbf{\tilde{R}}_\sigma^{-1} = \left(\frac{\mathbf{R}}{A}\right)^{-1}.
$$
The relationship above is mathematically guaranteed. Additionally, $\mathbf{R}_\sigma$ is guaranteed to be non-singular (and thus invertible). We won't get into both guarantees here.

Furthermore, the Dbar algorthm requires either the DtoN map of a body with the same geometry but with conductivity 1 or a reference DtoN map that corresponds to the same body with a different physiology (e.g. expiration and inspiration). This last case is designated by **tdEIT** and is very useful when monitoring health conditions that change with time like air ventilation in the lungs. With this in mind, the ***read_data class*** was constructed to compute one DtoN map from one set of current patterns and corresponding measured voltages. 

### Scattering Transform

 The second step of the algorithm is to compute the scattering transform. This transform is of non-physical nature, i.e. has no physical explanation, but is tightly connect with the the conductivity and can be determined by the Dirichlet-to-Neumann map. It is given in the new spectral parameter **k** in the following manner:
 
 <p align="center">
<img src="https://latex.codecogs.com/png.latex?%5Ctextbf%7Bt%7D%28k%29%20%3D%20%5Cint_%7B%5Cpartial%5COmega%7D%20e%5E%7Bi%5Cbar%7Bk%7D%5Cbar%7Bz%7D%7D%5Cleft%28%5CLambda_%7B%5Cgamma%7D%20-%20%5CLambda_%7B1%7D%20%5Cright%20%29%5Cpsi%28z%2C%20k%29%5C%2C%20ds%28z%29%20%3D%20%5Cint_%7B%5Cmathbb%7BR%7D%5E2%7D%20e%5E%7Bi%5Cbar%7Bk%7D%5Cbar%7Bz%7D%7Dq%28z%29%5Cpsi%28z%2C%20k%29%20%5C%2Cdxdy%5C%2C" /> </p>

Now due to the ill-posedness of the inverse problem, the noise in the measurements highly affects the values of **t** for large **|k|**. The regularization strategy proven by Siltanen shows that we only need to compute the following approximation for the scattering transform:
<p align="center">
<img src="https://latex.codecogs.com/png.latex?%5Ctextbf%7Bt%7D%5ER%28k%29%20%3D%20%5Cleft%5C%7B%20%5Cbegin%7Bmatrix%7D%20%5Cint_%7B%5Cpartial%5COmega%7D%20e%5E%7Bi%5Cbar%7Bk%7D%5Cbar%7Bz%7D%7D%5Cleft%28%5CLambda_%7B%5Cgamma%7D%20-%20%5CLambda_%7B1%7D%20%5Cright%20%29%5Cpsi%28z%2C%20k%29%5C%2C%20ds%28z%29%2C%20%5Ctext%7B%20for%20%7D%20%7Ck%7C%3CR%20%5C%5C%20%5C%5C%200%2C%20%5Cquad%20%5Cquad%20%5Cquad%5Cquad%5Cquad%5Cquad%5Cquad%5Cquad%5Cquad%5Cquad%5Cquad%5Cquad%5Ctext%7B%20otherwise%7D%20%5Cend%7Bmatrix%7D%5Cright." /> </p>

The beauty of this regularization strategy lies in a finite discretization of **k** inside a disk of radius **R** being required. The choice of **R** dependends in an estimate of the noise present in the measurements.

To completely understand the computation of the scattering transform we need to obtain **&psi;** at the boundary. Nachman's was able to prove that the values of &psi; at the boundary can be determined by the solution of a boundary integral equation for **k &in; &#8450; \ {0}**:
<p align="center">
<img src="https://latex.codecogs.com/png.latex?%5Cleft.%5Cpsi%28%5Ccdot%2C%20k%29%5Cright%7C_%7B%5Cpartial%5COmega%7D%20%3D%20%5Cleft.%20e%5E%7Bikz%7D%5Cright%7C_%7B%5Cpartial%5COmega%7D%20-%20S_%7Bk%7D%28%5CLambda_%7B%5Cgamma%7D-%7B%5CLambda_1%7D%29%5Cpsi%28%5Ccdot%2C%20k%29" /> </p>

However, this equation takes a lot of time to solve, thus for speedup purposes we choose an approximate solution. We can have two approaches:
- Solve the boundary integral equation with **S<sub>0</sub>** substituting **S<sub>k</sub>** and with the following definition:
<p align="center">
<img src="https://latex.codecogs.com/png.latex?%5Cleft%28S_0%20f%20%5Cright%20%29%28x%29%20%3D%20%5Cint_%7B%5Cpartial%5COmega%7D%20G_0%28x%20-%20y%29f%28y%29%5C%2C%20d%5Csigma%28y%29%2C%20%5Ctext%7B%20where%20%7D%20G_0%28x%29%20%3D%20-%5Cfrac%7B1%7D%7B2%5Cpi%7D%5Ctext%7B%20log%7D%5C%2C%7Cx%7C." /> </p>

This approach is present in [2].

- Take **&psi;&sime; e<sup> ikz</sup>**, since this is the leading behavior and do not solve any integral equation.
This is the fastest way, but its triviality leads to more errors.

Now the last step is to solve the Dbar equation, which gives the name to the algorithm.

### D-bar equation

Now, having computed the scattering transform or an approximation of it, we are able to solve the following PDE in the **k** variable:
<p align="center">
 <img src="https://latex.codecogs.com/png.latex?%5Ctext%7BFor%20all%20%7D%20z%5Cin%5COmega%2C%20%5Ctext%7Bsolve%20the%20D-bar%20equation%3A%20%7D%5Cquad%5Cquad%20%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%5Cbar%7Bk%7D%7D%20%5Cmu%28z%2C%20k%29%20%3D%20%5Cfrac%7B%5Ctextbf%7Bt%7D%28k%29%7D%7B4%5Cpi%5Cbar%7Bk%7D%7De%5E%7B-2i%5Ctext%7BRe%7D%28kz%29%7D%5Coverline%7B%5Cmu%28z%2Ck%29%7D" /> </p>
 
Afterwards, we can determine the conductivity by:
<p align="center">
 <img src="https://latex.codecogs.com/png.latex?%5Csigma%28z%29%20%3D%20%5Cleft%28%5C%2C%5Cmu%28z%2C%200%29%5C%2C%5Cright%29%5E2"/> </p>.
 
 By substituting the scattering transform by its approximations and solving the equation with respect to it we obtain respective approximations to the conductivity.
 
 ## Numerical Implementation
 
 Above, we present an immediate numerical implementation of the Dirichlet-to-Neumann map that trivially translates to code and a theoretical implementation of the regularization strategy for the D-bar and corresponding determination of the conductivity. Accordingly, we present the numerical implementation for both versions of the scattering transform and the solution to the Dbar equation.
 
 ### t-exp Scattering transform:
 
 The t-exp scattering transform corresponds to take the exponential approximation of &psi;, which has the following form:
 
 <p align="center">
  <img src="https://latex.codecogs.com/png.latex?%5Ctextbf%7Bt%7D%5E%7B%5Ctext%7Bexp%7D%7D%28k%29%20%3D%20%5Cint_%7B%5Cpartial%5COmega%7D%20e%5E%7Bi%5Cbar%7Bk%7D%5Cbar%7Bz%7D%7D%5Cleft%28%20%5CLambda_%7B%5Cgamma%7D%20-%20%5CLambda_1%20%5Cright%20%29e%5E%7Bikz%7D%5C%2C%20ds%28z%29"/> </p>
  
Let us consider that we have equally distributed electrodes around the boundary. By our assumption of &Omega; being a disc we define, without loss of generality, the positions of the electrodes to be <img src="https://render.githubusercontent.com/render/math?math=\vec{z}=(z^1,\, z^2, \, ...,\, z^L)" > where <img src="https://render.githubusercontent.com/render/math?math=z^j=(2\pi j)/L" >.

From this, we can take an approximation of the exponentials in these set points as an expansion in the orthonormal basis of current pattern (**L-1** degrees of freedom):

<p align="center">
 <img src="https://latex.codecogs.com/png.latex?%5Cbg_white%20e%5E%7Bik%5Cvec%7Bz%7D%7D%20%3D%20%5Csum_%7Bj%3D1%7D%5E%7BL-1%7D%20a_j%28k%29c%5Ej%20%5Cquad%20%5Cquad%20%5Cquad%20%5Ctext%7B%20and%20%7D%20%5Cquad%20%5Cquad%20%5Cquad%20e%5E%7Bi%5Cbar%7Bk%7D%5Cbar%7B%5Cvec%7Bz%7D%7D%7D%20%3D%20%5Csum_%7Bj%3D1%7D%5E%7BL-1%7D%20b_j%28k%29c%5Ej" />. </p>
 
By immediate substitution and approximation of the integral at the electrode points this leads too:

<p align="center">
<img src="https://latex.codecogs.com/png.latex?%5Cbg_white%20%5Ctextbf%7Bt%7D%5E%7B%5Ctext%7Bexp%7D%7D%28k%29%20%3D%20%5Csum_%7Bj%3D1%7D%5E%7BL-1%7D%5Csum_%7Bi%3D1%7D%5E%7BL-1%7D%20a_i%28k%29b_j%28k%29%20%28c%5Ej%2C%20%5Cleft%28%5CLambda_%7B%5Cgamma%7D%20-%20%5CLambda_1%20%5Cright%20%29c%5Ei%29%20%3D%20%5Csum_%7Bj%3D1%7D%5E%7BL-1%7D%5Csum_%7Bi%3D1%7D%5E%7BL-1%7D%20a_i%28k%29b_j%28k%29%20%5Cleft%28L_%7B%5Cgamma%7D%28j%2C%20i%29%20-%20L_1%28j%2C%20i%29%20%5Cright%29" />. </p>


### t-B Scattering Transform:

Here the first step is to solve the boundary integral equation with **S<sub>0</sub>**. For such we consider an initial vector approximation **&psi;**<sup>B</sup> of the solution of the (continuous) boundary integral equation and take the expansion in terms of the orthonormal current patterns:
<p align="center">
 <img src="https://latex.codecogs.com/png.latex?%5Cbg_white%20%5Cpsi%5EB%28k%29%20%3D%20%5Csum_%7Bj%3D1%7D%5E%7BL-1%7D%20p_j%28k%29%5C%2Cc%5Ej%20%3D%20%5Cleft%5Bc%5E1%5C%2C%20c%5E2%5C%2C%20...%20%5C%2Cc%5E%7BL-1%7D%20%5Cright%20%5D%5Ctextbf%7Bp%7D_k" />. </p> 
 
 Further, we also take a **LxL** matrix approximation of **G<sub>0</sub>** that defines the **S<sub>0</sub>** operator as:
 
 <p align="center">
  <img src="https://latex.codecogs.com/png.latex?%5Cbg_white%20%5Ctextbf%7BG%7D_0%28i%2C%20j%29%20%3D%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20-%5Cfrac%7B1%7D%7B2%5Cpi%7D%5Ctext%7Blog%7D%20%5C%2C%7Cz%5Ei%20-%20z%5Ej%7C%2C%20%5Cquad%5Ctext%7B%20if%20%7D%20i%5Cneq%20j%20%5C%5C%20%5C%5C%200%5Cquad%5Cquad%5Cquad%5Cquad%5Cquad%5Cquad%5C%2C%2C%5Cquad%5C%2C%20%5Ctext%7B%20if%20%7D%20i%3Dj%20%5Cend%7Bmatrix%7D%5Cright." />. </p>
  
 Through this matrix approximation, we can translate the boundary integral equation into a linear system of equations. Defining **&Phi;** to be the current pattern matrix (**L x L-1**) leads us too:
 
 <p align="center">
  <img src="https://latex.codecogs.com/png.latex?%5Cbg_white%20%5CPhi%5C%2C%5Ctextbf%7Bp%7D_k%20%3D%20e%5E%7Bik%5Cvec%7Bz%7D%7D%20-%20%5Ctextbf%7BG%7D_0%5Cleft%28L_%7B%5Cgamma%7D%20-%20L_1%20%5Cright%20%29%5CPhi%5C%2C%5Ctextbf%7Bp%7D_k%20%5C%5C%20%5C%5C%20%5CRightarrow%20%28%5CPhi%5ET%5CPhi%20&plus;%20%5CPhi%5ET%5Ctextbf%7BG%7D_0%5Cleft%28L_%7B%5Cgamma%7D%20-%20L_1%20%5Cright%20%29%5CPhi%29%5Ctextbf%7Bp%7D_k%20%3D%20%5CPhi%5ETe%5E%7Bik%5Cvec%7Bz%7D%7D" />, </p>
  
  
To compute the scattering transform we need to solve the last system of equations for each **k** in a disk of radius **R**. Afterwards, we compute it by:

<p align="center">
  <img src="https://latex.codecogs.com/png.latex?%5Cbg_white%20%5Ctextbf%7Bt%7D%5EB%28k%29%20%3D%20A%20%5Csum_%7Bl%3D1%7D%5E%7BL%7D%20%5Csum_%7Bn%3D1%7D%5E%7BL-1%7D%20e%5E%7Bi%5Cbar%7Bk%7D%5Cbar%7Bz%7D%5El%7D%5Ctextbf%7Bp%7D_k%5En%28%5CPhi%5Cdelta%20L%20%29%28l%2C%20n%29" />, </p>
  where &delta;L is the difference between L<sub>&gamma;</sub> and L<sub>1</sub>.
 
 
 A more in depth look at the details can be found in [1] with respect to the solution of the boundary integral equation with **S<sub>k</sub>**.
 
 
 ### D-bar Equation:
 
 Now we have to solve the D-bar equation in the **k**-variable. This is the essence of this new spectral parameter, it allows for the reconstruction of the conductivity without iterative solutions of an FEM model of conductivity.
 Although, we do not solve the PDE directly... we pass it to an integral equation and use Fourier Transform after discretization to solve the linear system of equations.
 
 ***Integral Equation***:
 
 Solve for each z&in;&Omega;: 
 <p align="center">
  <img src="https://latex.codecogs.com/png.latex?%5Cbg_white%20%5Cmu%28z%2C%20k%29%20%3D%201%20&plus;%20%5Cfrac%7B1%7D%7B%282%5Cpi%29%5E2%7D%5Cint_%7B%5Cmathbb%7BR%7D%5E2%7D%20%5Cfrac%7B%5Ctextbf%7Bt%7D%28k%27%29%7D%7B%28k%20-%20k%27%29%5Coverline%7Bk%27%7D%7De%5E%7B-2i%5C%2C%5Ctext%7BRe%7D%5C%2C%28zk%27%29%20%7D%5Coverline%7B%5Cmu%28z%2Ck%27%29%7D%5C%2C%20dk%27_1dk%27_2"/> </p>
 
 
 Recall that, the regularization procedure establishes that for **|k|>R** the scattering transform is **0**, in both cases. Hence, the integral can be taken in the disk of radius **R**. To practically solve this equation we need to deal with the singularity properly. A fast approach to solve the equation is to transform it into a periodic setting (initial idea was by Vainikko, [3], and adapted to the Dbar equation in [4]). Let <img src="https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20G_%7B%5Cbar%7B%5Cpartial%7D%7D%28k%29%20%3D%201/%28%5Cpi%20k%29" /> be the fundamental solution of the D-bar operator. First, we write the integral equation in convolution terms:
 
<p align="center">
  <img src="https://latex.codecogs.com/png.latex?%5Cbg_white%20%5Cmu%28z%2Ck%29%20%3D%201%20&plus;%20G_%7B%5Cbar%7B%5Cpartial%7D%7D%28k%29%5C%2C%5Cast%20%5Cleft%28T%28z%2C%20k%29%5Coverline%7B%5Cmu%28z%2C%20k%29%7D%20%5Cright%20%29%2C%20%5Ctext%7B%20where%20%7D%20T%28z%2C%20k%29%20%3D%20%5Cfrac%7B%5Ctextbf%7Bt%7D%28k%29%7D%7B4%5Cpi%5Cbar%7Bk%7D%7De%5E%7B-2i%5C%2C%5Ctext%7BRe%7D%5C%2C%28zk%29%7D"/> </p>
 
 
 The periodization is done by tiling the **k-plane** with a square Q=[-2R-3&epsilon;, 2R + 3&epsilon;]<sup>2</sup> for some &epsilon;>0, in this manner, Q containes the disk of radius 2R. Now, we take a smooth cut-off function &eta; which fulfils &eta;(k)=1 for |k|<2R+&epsilon; and &eta;(k)=0 for |k|>2R+2&epsilon;. With this we define the periodic approximate green function <img src="https://latex.codecogs.com/png.latex?%5Cinline%20%5Cwidetilde%7BG%7D_%7B%5Cbar%7B%5Cpartial%7D%7D%28k%29%20%3D%20%5Ceta%28k%29G_%7B%5Cbar%7B%5Cpartial%7D%7D%28k%29" />.
 
 In this framework we have the periodic version of the integral D-bar equation given by:
 
 <p align="center">
  <img src="https://latex.codecogs.com/png.latex?%5Cbg_white%20%5Cmu_%5Ctext%7BR%7D%28z%2C%20k%29%20%3D%201%20&plus;%20%5Cwidetilde%7BG%7D_%7B%5Cbar%7B%5Cpartial%7D%7D%28k%29%5Cstar%5Cleft%28T_%7B%5Ctext%7BR%7D%7D%28z%2Ck%29%5Coverline%7B%5Cmu_%7B%5Ctext%7BR%7D%7D%28z%2Ck%29%7D%29%20%5Cright%20%29"/> </p>
  
  where &Star; denotes periodic convolution and T<sub>R</sub> is the defined as T for |k|<R and 0 otherwise. The interesting part is that the solution of this equation and of the full equation coincide in |k|<R, which is what we desire.
  
 By discretizing the **k-plane** into an equally spaced grid with step **h**, the periodic convolution can be computed efficiently by Fast Fourier transforms and we obtain the following system of equations:
 
 <p align="center">
  <img src="https://latex.codecogs.com/png.latex?%5Cbg_white%20%5Cmu_%5Ctext%7BR%7D%20%3D%20I%20&plus;%20%5Ctext%7BIFFT%7D%5Cleft%28%5Ctext%7BFFT%7D%28%5Ctextbf%7BG%7D_%7B%5Cbar%5Cpartial%7D%29%5Ccdot%5Ctext%7BFFT%7D%28%5Ctextbf%7BT%7D_%7B%5Ctext%7BR%7D%7D%5Coverline%7B%5Cmu_R%7D%29%20%5Cright%20%29" /> </p>
  
  where the boldface notation represents the matrices formed by evaluation of the functions in the **k-grid** and &sdot; represents matrix multiplication. This equation can be written in a simpler form, which will be helpful to describe our code below:
  
  <p align="center">
  <img src="https://latex.codecogs.com/png.latex?%5Cbg_white%20%5BI%20&plus;%20%5Ctextbf%7BP%7D%5D%5Cmu_R%28z%2C%20%5Ccdot%29%20%3D%20%5Ctextbf%7B1%7D%2C%20%5Ctext%7B%20where%20%7D%20%5Ctextbf%7BP%7D%5Cmu_R%28z%2Ck%29%20%3D%20%5Ctext%7BIFFT%7D%28%5Ctext%7BFFT%7D%28%5Ctextbf%7BG%7D_%7B%5Cbar%5Cpartial%7D%29%5Ccdot%20%5Ctext%7BFFT%7D%28%5Ctextbf%7BT%7D_R%5Coverline%7B%5Cmu_R%28z%2Ck%29%7D%29%29" /> </p>.
  
  Since &mu;<sub>R</sub> is inside the Fourier transform, we can not describe this system through the simple form Ax=b. Hence, the system of equations lends itself well to a solution by a matrix-free iterative solver like the **GMRES**.
  
  Due to conjugation on &mu;<sub>R</sub> the equation is &Ropf;-linear and separation of the real and imaginary parts, which is done by justposition of the real and imaginary parts into a vector. Further details, can be seen in [1].
  
  
  This is what we need for implementation and it provides a framework for the code we present in this package. A description of each class and function and its role is presented next.
 
 ## Documentation:

Here, we provide an overview of each class and examples of usage. Our package possesses three classes:

                      sim, read_data, k_grid, scattering, dBar.
                      
 The first three classes are independent and concern the input information and framework. The fourth connects the gather data into a single operator, which is transmitted to the **dBar** class. This class is dependent of both the scattering and k_grid and performs the solver. 

# sim class

*class sim(anomaly, L)*

This class is a simulator of voltages obtained through the FEM-solution of the Direct problem. We can create the conductivity of a body in a disk and then solve the direct problem for this conductivity by applying pair-wise adjacent current patterns (as described above). Thereafter, we can determine the voltage at the electrode positions and hence create the necessary data to apply an inversion procedure.

This class is based solely on pyEIT.

**Parameters:**

- ***anomaly: list***

	By assumption we have that the background conductivity is equal to 1. Then we can create disk anomalies by selecting the center, radius and the conductivity of the region.

- ***L: int***

	Number of electrodes.

**Attributes:**
- ***Current: (L-1, L)***

	Matrix with the currents patterns applied to the body.

- ***Voltage: (L-1, L)***

	Matrix with the "measured" voltages obtained to sucessive FEM-solutions of the direct problem.
	
**Methods:**

- ***simulate():***
	
	This function uses each of the adjacent pair current patterns, solves the direct problem by FEM, and determines the voltages at the electrodes.


# read_data class

*class read_data(Object, r, AE, L)*

Defines the Dirichlet-to-Neumann matrix approximation from the experimental data: electrical measurements, radius of the body, area of electrode and number of electrodes.
The current version only holds for circular domains and for objects which have conductivity equal to 1 near the boundary.

**Parameters:** 
- ***Current: (L-1, L) 2darray***

	Matrix of current patterns. Each line is a current pattern.

- ***Voltage: (L-1, L) 2darray***

	Matrix of voltage measurements. Each line corresponds to the current pattern on the same line of the current matrix.
             
- ***r: float***        
             
    Radius of the body;
             
- ***AE: float***     
             
    Area of one electrode;
             
- ***L: int***          
            
    Number of electrodes;

**Attributes:**
- ***Current: (L, L-1) 2darray*** 
 
    Matrix of the normalized current patterns, each column represents one normalized current pattern;

- ***Voltage: (L, L-1) 2darray***

    Matrix of the normalized voltages, where the i-th column represents the voltages measured for the i-th current pattern;
              
- ***DNmap: (L-1, L-1) 2darray***  

    Matrix approximation of Dirichlet-to-Neumann map for the measured data.

**Methods:**  
- ***load():*** 

     This function starts by performing the normalization of the current and voltage measurements and thereafter computes the Dirichlet-to-Neumann as described above. It is called directly from the constructor and the user does need to use it directly.
            
### Examples:


# k_grid class

*class k_grid(R, m)*

Definition of the grid discretization of the **k-plane** with respect to the parameters **R** and **m**. As required by the periodic discrete convolution, the grid is defined in [-2.3R, 2.3R]^2 with step size h=2(2.3R)/2<sup>m</sup>. 

**Parameters:**
- ***R: float***

    Defines the size of the k grid

- ***m: int***
    
    Defines the step size.

**Attributes:**
- ***R: float***

    As the parameter.
 
- ***m: int***
    
    As the parameter.
    
- ***s: float***

    It is defined as 2.3R, to simplify definitions further ahead.

- ***h: float***
      
    Step size of the k grid.
      
- ***N: int*** 
       
    It is defined as 2<sup>m</sub>. Therefore, it is the number of elements on each line of the grid.

- ***k: (N, N) complex 2darray***
    
    Complex values of the k_grid. Essential for complex-valued computations. This grid contains the value **0**.

- ***pos_x, pos_y: array_like***

    Arrays that store the indexes of the elements of ***k*** which are inside the disk of radius ***R***.

- ***index: int***

    Represents the position of zero.
    
 - ***FG: (N, N) complex 1darray***

    The fundamental solution <img src="https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20G_%7B%5Cbar%5Cpartial%7D"/> of the D-bar operator, with the same structure of ***k***.
    
**Methods:**
 
 - ***generate():***
 
    Defines the complex k-grid ***k*** and determines the positions inside the disk of radius R.
    
- ***Green_FS():***

    Defines the FFT of fundamental solution **FG**. Recall that we perform a periodization of the convolution equation and therefore we need to establish a smooth decay to 0 close to the square limits of the grid.
 
# scattering class

 *class scattering(scat_type, k_grid, Now, Ref):*
 
 **Parameters:**
 
 - ***scat_type: string***
 	
     Allows to choose between the approximations of the scattering transform. Choices: "partial", "exp"!.

- ***k_grid: Object***
	
     An object of k_grid type.

- ***Now, Ref: Objects***

     Objects of read_data type (we require the DN map for our body of choice and an homogeneous/reference body).

**Attributes:**

- ***tK: (k_grid.N, k_grid.N) complex 2darray***

     Matrix containing the values of an intermediare operator containing information about the DN map of our body of choice. Essential for the Dbar method.

**Methods:**

- ***load_scattering(scat_type, k_grid, Now, Ref):***

    Given the choice of approximation for the scattering transform this function selects between two functions: one compute the exponential aproximation, another computes the partial approximation;

- ***exp_scattering(Now, Ref, k_grid):***
 
    Computes the exponential approximation of the scattering transform.
    
- ***partial_scattering(Now, Ref, k_grid):***
 
    Computes the partial approximation of the scattering transform.
    
   
 # dBar class
 
 *class dBar(R_z, m_z)*
 
 **Parameters:**
  
- ***R_z: float***

    Defines the size of the z-grid.

- ***m_z: int***

    Defines the step size of the z-grid.
    
    
**Attributes:**
     
 - ***Z: (2<sup>m_z</sup>, 2<sup>m_z</sup>) complex 2darray***

    Planar region of the body.

 - ***sigma: (2<sup>m_z</sup>, 2<sup>m_z</sup>) complex 2darray***
    
    Conductivity values at the points of the Z-plane grid.


**Methods:**

- ***load_mesh(R, m):***

    Defines the complex z-grid ***Z***.
    
- ***dBar(mu, k_grid, tK, zz):***

    Defines the operator [I + **P**] for a **zz** element of the z-grid, which is the left-hand side of the Dbar system. It takes as input the vector &mu; which is a 1darray that contains the real part concatenated with the imaginary part. We assume &mu; is 0 outside the disk of radius R, for speed-up purposes, however this is not a constraint since the scattering transform would cut-off this terms when defining the operator.

- ***solve(k_grid, tK):***

    Solve for each **z** on the z-grid the Dbar system with ***GMRES***. We use the ***dBar*** function to define the [I+**P**] operator for each **z** and use the initial approximation of **mu** being close to **1** to start off. After solving the system, we store the value of the conductivity through **&gamma;(z)=(&mu;(z,0))^2**.
    
- ***plot():***

    Plot the obtained conductivity on the respective Z grid.  


## Bibliography:

[1] Mueller, J. L., & Siltanen, S. (2020). The d-bar method for electrical impedance tomography—demystified. Inverse problems, 36(9), 093001.

[2] El Arwadi, T., & Sayah, T. (2021). A new regularization of the D-bar method with complex conductivity. Complex Variables and Elliptic Equations, 66(5), 826-842.

[3] Vainikko, G. (2000). Fast solvers of the Lippmann-Schwinger equation. In Direct and inverse problems of mathematical physics (pp. 423-440). Springer, Boston, MA.

[4] Knudsen, K., Mueller, J., & Siltanen, S. (2004). Numerical solution method for the dbar-equation in the plane. Journal of Computational Physics, 198(2), 500-517.
