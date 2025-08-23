# nr_link_full.py
import numpy as np
import tensorflow as tf

def lin2db(x): return 10.0*np.log10(max(float(x), 1e-12))

def eesm_eff_snr_lin(gamma_lin, beta=2.5):
    z = tf.exp(-tf.convert_to_tensor(gamma_lin, tf.float32)/tf.cast(beta, tf.float32))
    m = tf.reduce_mean(z)
    m = tf.maximum(m, tf.constant(1e-30, tf.float32))
    return float(-beta * tf.math.log(m).numpy())

class NRLinkFull:
    """
    Simple full-link wrapper:
      - draws freq-selective channel (Rayleigh w/ mild freq correlation fallback)
      - computes post-equalization per-subcarrier SNRs
      - returns EESM effective SINR (dB) per TTI
    """
    def __init__(self,
                 carrier_freq_hz=3.5e9,
                 scs_khz=30,
                 n_prb=24,
                 n_symb_slot=14,
                 n_tx=1, n_rx=1,
                 channel_model="TDL-C",
                 delay_spread_s=100e-9,
                 ue_speed_mps=3.0,
                 pilot_spacing=(2,2),
                 eesm_beta=2.5,
                 seed=123):
        self.scs_khz = scs_khz
        self.n_prb = int(n_prb)
        self.n_sc  = int(n_prb)*12
        self.n_sym = int(n_symb_slot)
        self.n_tx, self.n_rx = int(n_tx), int(n_rx)
        self.beta = float(eesm_beta)
        self.rng  = np.random.default_rng(seed)

        # Try to import 38.901 models; fall back if unavailable
        try:
            from sionna.channel.tr38901 import TDL, CDL  # noqa: F401
            self._has_38901 = True
        except Exception:
            try:
                from sionna.channel import TDL, CDL  # noqa: F401
                self._has_38901 = True
            except Exception:
                self._has_38901 = False
                print("⚠️ Sionna TR 38.901 channel not found. Falling back to i.i.d. Rayleigh.")

        # Pre-build a small smoothing kernel for frequency correlation (fallback)
        L = 7
        w = np.hanning(2*L+1); w = w/np.sum(w)
        self._smooth_kernel = w.astype(np.float64)

    def _draw_channel_freq(self):
        """Return H[f] as tf.complex64 of shape [n_rx, n_tx, n_sc]. Fallback uses Rayleigh + freq smoothing."""
        # Fallback: freq-correlated Rayleigh per slot
        k = self._smooth_kernel
        z = (self.rng.standard_normal((self.n_sc + k.size - 1,))
             + 1j*self.rng.standard_normal((self.n_sc + k.size - 1,))) / np.sqrt(2)
        h = np.convolve(z, k, mode="valid")[:self.n_sc]
        # Normalize average power to 1
        p = np.mean(np.abs(h)**2)
        if p > 0: h = h / np.sqrt(p)
        H = h.reshape(1, 1, -1).astype(np.complex64)  # [n_rx,n_tx,n_sc]
        return tf.constant(H)

    def _mmse_equalize_and_snr(self, H, noise_var_lin):
        """
        H: [n_rx,n_tx,n_sc] complex64
        noise_var_lin: float32 scalar σ²
        Returns per-subcarrier SNRs (linear, float32)  [n_sc]
        """
        H = tf.cast(H, tf.complex64)
        sigma2 = tf.cast(noise_var_lin, tf.float32)

        if self.n_tx == 1 and self.n_rx == 1:
            # SISO: post-eq SNR ≈ |h|^2 / σ²
            h = tf.squeeze(H)                      # [n_sc] complex64
            p = tf.math.square(tf.abs(h))          # [n_sc] float32
            return p / sigma2

        # Simple MIMO proxy: trace(H^H H)/σ² / n_tx (real)
        Hr = tf.transpose(H, perm=[2,0,1])         # [n_sc, n_rx, n_tx]
        snrs = []
        for k in tf.range(self.n_sc):
            Hk   = tf.cast(Hr[k], tf.complex64)    # [n_rx, n_tx]
            gram = tf.matmul(tf.linalg.adjoint(Hk), Hk)       # [n_tx,n_tx]
            tr   = tf.math.real(tf.linalg.trace(gram))        # float32
            snrs.append(tr / (sigma2 * tf.cast(self.n_tx, tf.float32)))
        return tf.stack(snrs, axis=0)

    def eff_sinr_db(self, avg_snr_db):
        """
        1 TTI: draw H, compute per-RE post-eq SNRs, EESM-compress -> effective SINR in dB.
        """
        avg_lin   = tf.convert_to_tensor(10.0**(float(avg_snr_db)/10.0), tf.float32)
        noise_var = tf.cast(1.0/avg_lin, tf.float32)
        H = self._draw_channel_freq()
        post_snr_sc = self._mmse_equalize_and_snr(H, noise_var)  # float32 [n_sc]
        gamma_eff = eesm_eff_snr_lin(post_snr_sc, beta=self.beta)
        return lin2db(gamma_eff)
