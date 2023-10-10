# Damped Trust Region Ratio
This introduces two algorithms: `DampedRatioBFGS` and `DampedRatioLM` for minimisation and nonlinear least-squares respectively. Both use a trust-region update with a damped trust-region ratio given by:

$$ \hat{\rho}_k = \frac{f(x_k + p) - f(x_k)}{\nabla f(x_k)^T p + \frac{1}{2} p^T (H_k + \lambda_k I) p } $$

where $H_k$ is the BFGS matrix at step $k$ in `DampedRatioBFGS` and the Gauss-Newton matrix $J_k^T J_k$ for Jacobian $J_k$ in `DampedRatioLM`.

These are both implemented by writing a single `search` in [Optimistix](https://github.com/patrick-kidger/optimistix), touching only the trust-region update code.


## Further Details
At an iterate $x_k$, many second-order optimisation methods locally approximate the objective function $f: \mathbb{R}^n \to \mathbb{R}$ with the quadratic model function

$$ m_k(p) = f(x_k) + p^T \nabla f(x_k) + \frac{1}{2} p^T H_k p $$

where $H_k$ is a positive semidefinite approximation to the Hessian $H_k \approx \nabla^2 f(x_k)$. This model function $m_k$ is minimised at $x_k$ to get the next iterate $x_{k + 1}$.

Tikhnov regularisation adds a term $\lambda_k I$ to $H_k$ at each step, defining

$$ \hat{m}_k(p) = f(x_k) + p^T \nabla f(x_k) + \frac{1}{2} p^T (H_k + \lambda_k I) p. $$

This can be interpreted as interpolating between the Newton and gradient update (with some scaling.) Higher values of $\lambda_k$ correspond to smaller steps with more gradient influence, which is useful in regions of quickly changing curvature. Lower values of $\lambda_k$ exploit the curvature information in $H_k$ more, and are better when the curvature is slowly varying.

The Levenberg-Marquardt (LM) heuristic is a method which attempts to adjust $\lambda_k$ dynamically. The heuristic adjusts the size $\lambda_k$ based on the trust-region ratio:

$$\rho_k = \frac{f(x_k) - f(x_k  + p)}{m_k(0) - m_k(p)}.$$

When $m_k$ does a good job of locally approximating $f$, we expect $\rho_k$ to be near $1$. When it does a poor job, we expect $\rho_k$ to be closer to $0$. As such, the LM heuristic is to set

$$ \lambda_{k+1} = \begin{cases} c_2 \lambda_k & \rho_k > C_2 \\ 
                             \lambda_k & C_1 < \rho_k < C_2 \\
                             c_1 \lambda_k & \rho_k < C_1 \end{cases} $$

for some multipliers $c_1, c_2$ and some cutoffs $C_1, C_2$. The choice of these constants varies, by default we choose those found to perform best in [this review paper](https://www.numerical.rl.ac.uk/people/nimg/pubs/GoulOrbaSartToin05_4or.pdf) (low cutoff $C_1 = 0.01$, low mulitplier $c_1 = 0.25$, high cutoff $C_2 = 0.99$, and high multiplier $c_2 = 3.5$.)

The trust region ratio $\rho_k$ and LM heuristic use the model function $m_k$; however, the actual model function which LM uses is $\hat{m}_k$. To address this we introduce

$$\hat{\rho}_k = \frac{f(x_k) - f(x_k  + p)}{\hat{m}_k(0) - \hat{m}_k(p)} = \frac{f(x_k + p) - f(x_k)}{\nabla f(x_k)^T p + \frac{1}{2} p^T (H_k + \lambda_k I) p }$$

and use the same LM heuristic but with $\hat{\rho}_k$ instead of $\rho_k$.

We introduce two optimisers `DampedRatioBFGS`, and `DampedRatioLM`. `DampedRatioBFGS` is a minimiser which sets $H_k$ to the [BFGS approximation](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm#Algorithm) $B_k$, and `DampedRatioLM` is a least-squares algorithm which sets $H_k$ to the Gauss-Newton approximation $J_k^TJ_k$, where $J_k$ is the Jacobian of the model function at step $k$.

This idea is not novel, I took it from [Training Deep and Recurrent Networks with
Hessian-Free Optimization, section 8](https://www.cs.toronto.edu/~jmartens/docs/HF_book_chapter.pdf). However, it is also not common, and you won't find it in an off-the-shelf implementation using BFGS.

The authors found this modification didn't make a huge difference, which is unsurprising. However, I want to stress that without Optimistix it's difficult verify this claim. In particular, `DampedRatioBFGS` would require a fully custom implementation of BFGS, as using BFGS + Tikhnov regularisation is not a standard technique. Using Optimistix, it requires writing only the `search` responsible for controlling the trust-region update, and gives both the full BFGS and Gauss-Newton algorithms "for free."
