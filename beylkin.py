"""
beylkin.py

Usage:  python beylkin.py

Note:  This code fits growing and decaying functions that may be well approximated
in a basis of complex exponential functions.  Data sets may be fitted using the
driver_load method.  Equally spaced data points are required.  

Reference:  G. Beylkin and L. Monzon, Appl. Comput. Harmon. Anal. 19, 17 (2005).

Author: Ian S. Dunn, Columbia University Department of Chemistry, isd2107@columbia.edu
"""

# Import modules.
import sys
import numpy as np
from bessel import bessel
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import scipy


class Beylkin:

    def __init__(self, decaying=False):

        self.x = None
        self.h = None
        self.N = None
        self.H = None
        self.ceigenvec = None
        self.gamma = None

        self.decaying = decaying

    def load_data(self, x, h):

        assert len(x) == len(h)

        self.N = (len(x) - 1) / 2

        self.x = np.array(x[:2*self.N+1], dtype=float)
        self.h = h[:2*self.N+1]

        self.dx = float(self.x[1] - self.x[0])

        self.T = self.x[-1] - self.x[0]

    def sample(self, f, start=0, end=1, N=214):

        # Number of points for sampling.
        self.N = N

        # Sampling grid.
        self.x = np.linspace(0, 1, 2 * N + 1, endpoint=True, dtype=np.complex128)

        # Evaluate function on grid.
        self.h = f(self.x)

    def plot_input(self):

        # Plot function.
        plt.plot(self.x, self.h)
        plt.show()

    def build_hankel(self):

        assert self.h is not None

        # Build Hankel matrix.
        self.H = scipy.linalg.hankel(self.h[:2*self.N+1].astype(np.complex128))[:self.N+1, :self.N+1]

        # Check that Hankel matrix is real and symmetric.
        assert np.allclose(self.H, self.H.T, atol=1.e-15)

    def eigen(self, nsing):

        assert self.H is not None

        # Initial Krylov vector.
        v0 = np.zeros(len(self.H), dtype=np.complex128)
        v0[0] = 1.

        # Calculate singular vectors and values of Hankel matrix.
        w, v = eigsh(self.H, k=nsing+1, v0=v0)

        # Find smallest numerically acceptable c-eigenpair.
        w_max = np.max(abs(w))
        threshold = w_max * 1.e-15
        large = np.where(abs(w) > threshold)[0]
        v = v[:, large]
        w = w[large]

        ind = np.argmin(abs(w))
        self.ceigenvec = v[:, ind]

        #print "Approximate Prony analysis with", small_ind, "th singular value."

    def nodes(self):

        assert self.ceigenvec is not None

        # Evaluate roots of c-eigenpolynomial.
        gamma = np.roots(self.ceigenvec[::-1])

        # Remove large roots.
        max_gamma = (np.max(abs(self.h)) * 1000.) ** (1. / (2*self.N))
        large_inds = np.where(abs(gamma) > max_gamma)
        gamma = np.delete(gamma, large_inds)

        # Remove small roots.
        if self.decaying:
            gamma = np.delete(gamma, np.where(abs(gamma) > 1.))
        else:
            gamma = np.delete(gamma, np.where(abs(gamma) < 1.))

        # Store nodes.
        self.gamma = gamma

    def calculate_weights(self, nsing):

        assert self.gamma is not None
        assert self.gamma.size > 0

        # Build Vandermonde matrix from c-eigenroots.
        self.vand = np.vander(self.gamma, N=2*self.N+1).transpose()[::-1]

        # Normalize Vandermonde columns to improve conditioning of least squares by SVD.
        self.vand_norm = np.linalg.norm(self.vand, axis=0)
        self.vand /= self.vand_norm
        self.vand[np.where(abs(self.vand) < 1.e-14)] = 0

        # Change basis from complex exponentials to oscillating real exponentials.
        lamda_dt = np.log(self.gamma)
        omega_real_dt = np.real(lamda_dt)
        omega_imag_dt = np.imag(lamda_dt)
        self.vand = abs(self.vand)
    
        # Indices where weights correspond to damped cosines, damped sines, and monotonically damped functions.
        self.cos_inds = np.where(omega_imag_dt > 1.e-10)[0]
        self.sin_inds = np.where(omega_imag_dt < -1.e-10)[0]
        self.monotonic_inds = np.where(abs(omega_imag_dt) <= 1.e-10)[0]
        
        for i, wt in enumerate(omega_imag_dt):
        
            if wt > 1.e-10:
                self.vand[:, i] *= np.cos(omega_imag_dt[i] * np.arange(self.vand.shape[0]))
            elif wt < -1.e-10:
                self.vand[:, i] *= -np.sin(omega_imag_dt[i] * np.arange(self.vand.shape[0]))

        # Calculate Prony weights using least squares fit.
        lstsq_ret = scipy.linalg.lstsq(self.vand, self.h, cond=1.e-13)
        self.weights = lstsq_ret[0]

        # Remove small weights.
        self.weights[np.where(abs(self.gamma) < 1.e-14)] = 0.

        # Sort weights.
        inds = np.argsort(abs(self.weights))

        # Set small weights to zero.
        self.weights[inds[:-nsing]] = 0

    def plot_components(self):

        # Plot absolute value of Prony components.
        for i in range(self.vand.shape[1]):
            plt.plot(abs(self.vand[:, i]))
        plt.show()

    def prony(self):

        assert self.vand is not None
        assert self.weights is not None

        # Construct approximate Prony approximation.
        self.approx = np.dot(self.vand, self.weights)

    def plot_prony(self):

        assert self.approx is not None

        # Plot original function and approximate Prony approximation.
        plt.plot(self.approx)
        plt.plot(self.h)
        plt.show()

    def correction(self):

        inds = np.where(abs(self.gamma) <= 1.)
        weights = np.copy(self.weights)
        #weights[inds] = 0.

        return np.dot(self.vand, weights)

    def prony_function(self, x):
        """Function for calculating approximate Prony decomposition on arbitrary grid."""

        # Convert domain to float.
        x = np.array(x - self.x[0], dtype=float)

        # Rescale domain to units where sampled data has a spacing of 1.
        x /= self.dx

        # Array for storing output.
        result = np.zeros_like(x)

        # Decay constants and oscillation frequencies.
        lamda_dt = np.log(self.gamma)
        lamda_dt_real = np.real(lamda_dt)
        lamda_dt_imag = abs(np.imag(lamda_dt))

        # Account for rescaling in least squares fit.
        weights = self.weights / self.vand_norm
        weights[np.where(lamda_dt_real < 2.e-2 * self.dx)] = 0.

        max_weight = np.max(abs(weights))
        weights[np.where(abs(weights) < 1.e-13 * max_weight)] = 0.

        self.cos_inds = np.delete(self.cos_inds, np.where(abs(weights[self.cos_inds]) < 1.e-15))
        self.sin_inds = np.delete(self.sin_inds, np.where(abs(weights[self.sin_inds]) < 1.e-15))
        self.monotonic_inds = np.delete(self.monotonic_inds, np.where(abs(weights[self.monotonic_inds]) < 1.e-15))

        # Evaluate damped cosines.
        for ind in self.cos_inds:
            result += weights[ind] * np.exp(lamda_dt_real[ind] * x) * np.cos(lamda_dt_imag[ind] * x)

        # Evaluate damped sines.
        for ind in self.sin_inds:
            result += weights[ind] * np.exp(lamda_dt_real[ind] * x) * np.sin(lamda_dt_imag[ind] * x)
            
        # Evaluate monotonic terms.
        for ind in self.monotonic_inds:
            result += weights[ind] * np.exp(lamda_dt_real[ind] * x)

        # Return approximate Prony fit f(x).
        return result

    def roots(self, p, nsing):
        """
        Return nsing roots of a polynomial with coefficients given in p.
    
        The values in the rank-1 array `p` are coefficients of a polynomial.
        If the length of `p` is n+1 then the polynomial is described by::
    
          p[0] * x**n + p[1] * x**(n-1) + ... + p[n-1]*x + p[n]
    
        Parameters
        ----------
        p : array_like
            Rank-1 array of polynomial coefficients.
    
        Returns
        -------
        out : ndarray
            An array containing the complex roots of the polynomial.
    
        Raises
        ------
        ValueError
            When `p` cannot be converted to a rank-1 array.
    
        See also
        --------
        poly : Find the coefficients of a polynomial with a given sequence
               of roots.
        polyval : Compute polynomial values.
        polyfit : Least squares polynomial fit.
        poly1d : A one-dimensional polynomial class.
    
        Notes
        -----
        The algorithm relies on computing the eigenvalues of the
        companion matrix [1]_.
    
        References
        ----------
        .. [1] R. A. Horn & C. R. Johnson, *Matrix Analysis*.  Cambridge, UK:
            Cambridge University Press, 1999, pp. 146-7.
    
        """
        import numpy.core.numeric as NX

        # If input is scalar, this makes it an array
        p = np.core.atleast_1d(p)
        if len(p.shape) != 1:
            raise ValueError("Input must be a rank-1 array.")
    
        # find non-zero array entries
        non_zero = NX.nonzero(NX.ravel(p))[0]
    
        # Return an empty array if polynomial is all zeros
        if len(non_zero) == 0:
            return NX.array([])
    
        # find the number of trailing zeros -- this is the number of roots at 0.
        trailing_zeros = len(p) - non_zero[-1] - 1
    
        # strip leading and trailing zeros
        p = p[int(non_zero[0]):int(non_zero[-1])+1]
    
        # casting: if incoming array isn't floating point, make it floating point.
        if not issubclass(p.dtype.type, (NX.floating, NX.complexfloating)):
            p = p.astype(float)
    
        N = len(p)
        if N > 1:
            # build companion matrix and find its eigenvalues (the roots)
            A = np.diag(NX.ones((N-2,), p.dtype), -1)
            A[0,:] = -p[1:] / p[0]

            #power = Power(A)

            #for i in range(nsing):
            #    power.iterate(np.random.rand(len(A)))

            roots = scipy.sparse.linalg.eigs(A, k=nsing)[0]
            #roots = power.u

        else:
            roots = NX.array([])
   
        return roots

    def lamda(self):

        lamda = np.log(self.gamma**(2.*self.N / self.T))
        return lamda

    def check_fit(self, label=''):
        # Normalize trajectories.
        new = self.correction()
        norm_factor = np.linalg.norm(new)
        new /= norm_factor
        inp = self.h.copy()
        inp /= norm_factor

        # Absolute error.
        error = abs(inp - new)

        # Zero out error for values of trajectory < 1000.
        small_inds = np.where(abs(self.h[:len(new)]) < max(np.max(abs(self.h[:len(new)]))*1.e-14, 1000))
        error[small_inds] = 0.

        # Location of largest error.
        location = np.argmax(error)

        # Plot trajectories and interrupt if error is significantly large.
        if error[location] / abs(inp)[location] > 0.01:
            print "Inaccurate Prony approximation."
            print "Hierarchy element:", tup
            print "Max error location:", location
            print "Error:", error[location] / abs(inp)[location]
            print ""
            np.save('fit'+label+'.npy', self.correction())
            np.save('data'+label+'.npy', self.h)
            plt.plot(self.correction(), label='Correction')
            plt.plot(self.h, label='Trajectory')
            plt.legend()
            plt.savefig('prony'+label+'.png')
            plt.close()
            sys.exit()

    def driver(self, f, nsing):

        self.sample(f)

        self.build_hankel()

        self.eigen(nsing)

        self.nodes()

        self.calculate_weights(nsing)

        self.prony()

    def driver_load(self, x, h, nsing):

        self.load_data(x, h)

        self.build_hankel()

        self.eigen(nsing)

        self.nodes()

        self.calculate_weights(nsing)

        self.prony()

if __name__ == "__main__":

    f = bessel

    b = Beylkin(decaying=True)

    b.driver(f, 20)

    b.plot_prony()
