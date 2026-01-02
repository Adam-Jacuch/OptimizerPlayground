function dataOf(A) {
    if (!A || !A.selection || !A.selection.data) {
        throw new Error("This numjs build does not expose A.selection.data (needed for micro-ops).");
    }
    return A.selection.data;
}

function mapNd(A, fn) {
    const out = A.clone();
    const d = dataOf(out);
    for (let i = 0; i < d.length; i++) d[i] = fn(d[i], i);
    return out;
}

const Activations = {
  relu: {
    f:  (z) => mapNd(z, v => (v > 0 ? v : 0)),
    df: (z) => mapNd(z, v => (v > 0 ? 1 : 0)),
  },

  leaky_relu: {
    // default alpha = 0.01
    f:  (z, alpha = 0.01) => mapNd(z, v => (v > 0 ? v : alpha * v)),
    df: (z, alpha = 0.01) => mapNd(z, v => (v > 0 ? 1 : alpha)),
  },

  elu: {
    // default alpha = 1.0
    f: (z, alpha = 1.0) =>
      mapNd(z, v => (v >= 0 ? v : alpha * (Math.exp(v) - 1))),
    df: (z, alpha = 1.0) =>
      mapNd(z, v => (v >= 0 ? 1 : alpha * Math.exp(v))),
  },

  gelu: {
    // GELU (tanh approximation): 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    f: (z) => {
      const c = Math.sqrt(2 / Math.PI);
      return mapNd(z, x => {
        const x3 = x * x * x;
        const u = c * (x + 0.044715 * x3);
        const t = Math.tanh(u);
        return 0.5 * x * (1 + t);
      });
    },
    df: (z) => {
      const c = Math.sqrt(2 / Math.PI);
      return mapNd(z, x => {
        const x2 = x * x;
        const x3 = x2 * x;

        const u = c * (x + 0.044715 * x3);
        const t = Math.tanh(u);
        const sech2 = 1 - t * t;

        // du/dx = c * (1 + 3*0.044715*x^2)
        const du = c * (1 + 3 * 0.044715 * x2);

        // d/dx [0.5*x*(1+t)] = 0.5*(1+t) + 0.5*x*sech^2(u)*du
        return 0.5 * (1 + t) + 0.5 * x * sech2 * du;
      });
    }
  },

  swish: {
    // swish(x) = x * sigmoid(x)
    f: (z) => mapNd(z, x => {
      const s = 1 / (1 + Math.exp(-x));
      return x * s;
    }),
    df: (z) => mapNd(z, x => {
      const s = 1 / (1 + Math.exp(-x));
      // derivative: s + x*s*(1-s)
      return s + x * s * (1 - s);
    }),
  },

  tanh: {
    f: (z) => mapNd(z, v => {
      const e2 = Math.exp(2 * v);
      return (e2 - 1) / (e2 + 1);
    }),
    df: (z) => {
      const t = mapNd(z, v => {
        const e2 = Math.exp(2 * v);
        return (e2 - 1) / (e2 + 1);
      });
      return mapNd(t, v => 1 - v * v);
    }
  },

  sigmoid: {
    f: (z) => mapNd(z, v => 1 / (1 + Math.exp(-v))),
    df: (z) => {
      const s = mapNd(z, v => 1 / (1 + Math.exp(-v)));
      return mapNd(s, v => v * (1 - v));
    }
  },

  softsign: {
    // softsign(x) = x / (1 + |x|)
    f: (z) => mapNd(z, x => x / (1 + Math.abs(x))),
    df: (z) => mapNd(z, x => {
      // derivative: 1 / (1 + |x|)^2
      const d = 1 + Math.abs(x);
      return 1 / (d * d);
    }),
  },

  linear: {
    f:  (z) => z,
    df: (z) => mapNd(z, () => 1)
  }
};

function asCol(v) {
    return (v.shape && v.shape.length === 1) ? v.reshape(v.shape[0], 1) : v;
}

function mseGrad(yPred, yTrue) {
    return yPred.subtract(yTrue);
}

class Optimizer {
    constructor({
        lr = 0.01,
        momentum = 0.0,
        l2 = 0.0,
        weightDecay = 0.0,
        clipNorm = 0.0
    } = {}) {
        this.lr = lr;
        this.momentum = momentum;
        this.l2 = l2;
        this.weightDecay = weightDecay;
        this.clipNorm = clipNorm;

        this.vW = null;
        this.vb = null;

        this.vW2 = null;
        this.vb2 = null;
    }

    _initState(model) {
        const L = model.weights.length;
        this.vW = new Array(L);
        this.vb = new Array(L);
        for (let l = 0; l < L; l++) {
            this.vW[l] = nj.zeros(model.weights[l].shape);
            this.vb[l] = nj.zeros(model.biases[l].shape);
        }
        this.vW2 = new Array(L);
        this.vb2 = new Array(L);
        for (let l = 0; l < L; l++) {
            this.vW2[l] = nj.zeros(model.weights[l].shape);
            this.vb2[l] = nj.zeros(model.biases[l].shape);
        }
    }

    _maybeClip(grads) {
        if (!this.clipNorm || this.clipNorm <= 0) return grads;

        let sumsq = 0;

        const addSumsq = (G) => {
        const data = G.selection.data;
            for (let i = 0; i < data.length; i++) sumsq += data[i] * data[i];
        };

        for (const g of grads.dW) addSumsq(g);
        for (const g of grads.db) addSumsq(g);

        const norm = Math.sqrt(sumsq);
        if (norm <= this.clipNorm) return grads;

        const scale = this.clipNorm / (norm + 1e-12);
        return {
            dW: grads.dW.map(g => g.multiply(scale)),
            db: grads.db.map(g => g.multiply(scale)),
        };
    }

    apply(model, grads) {
        if (!this.vW) this._initState(model);

        grads = this._maybeClip(grads);

        const L = model.weights.length;
        const lr = this.lr;
        const mu = this.momentum;

        for (let l = 0; l < L; l++) {
            let gW = grads.dW[l];
            let gb = grads.db[l];

            // L2
            if (this.l2) gW = gW.add(model.weights[l].multiply(this.l2));

            // Momentum
            if (mu) {
                this.vW[l] = this.vW[l].multiply(mu).add(gW.multiply(1 - mu));
                this.vb[l] = this.vb[l].multiply(mu).add(gb.multiply(1 - mu));
                gW = this.vW[l];
                gb = this.vb[l];
            }

            // Weight Decay
            if (this.weightDecay) {
                model.weights[l] = model.weights[l].multiply(1 - lr * this.weightDecay);
            }
            
            model.weights[l] = model.weights[l].subtract(gW.multiply(lr));
            model.biases[l]  = model.biases[l].subtract(gb.multiply(lr));
        }
    }
}

class MLP {
    constructor(sizes, { activation = "relu" } = {}) {
        this.sizes = sizes.slice();
        this.weights = [];
        this.biases = [];

        this.act = typeof activation === "string" ? Activations[activation] : activation;
        if (!this.act || !this.act.f || !this.act.df) throw new Error("activation must be a name in Activations or an object {f, df}");

        for (let l = 0; l < sizes.length - 1; l++) {
            const fanIn = sizes[l];
            const fanOut = sizes[l + 1];

            this.weights.push(nj.random([fanOut, fanIn]).subtract(0.5).multiply(2 / Math.sqrt(fanIn)));
            this.biases.push(nj.zeros([fanOut, 1]));
        }
    }

    forward(x, { training = false } = {}) {
        let a = asCol(x);
        const cache = training ? { a: [a], z: [] } : null;

        const L = this.weights.length;

        for (let l = 0; l < L - 1; l++) {
            const z = nj.dot(this.weights[l], a).add(this.biases[l]);
            a = asCol(this.act.f(z));

            if (training) {
                cache.z.push(z);
                cache.a.push(a);
            }
        }
        const out = nj.dot(this.weights[L - 1], a).add(this.biases[L - 1]);
        const y = asCol(out);
        if (training) {
            cache.z.push(y);
            cache.a.push(y);
        }

        return training ? { y, cache } : y;
    }

    backward(yTrue, cache) {
        const L = this.weights.length;
        const dW = new Array(L);
        const db = new Array(L);

        const a = cache.a;
        const z = cache.z;

        let delta = mseGrad(a[L], yTrue);
        dW[L - 1] = nj.dot(delta, a[L - 1].T);
        db[L - 1] = delta.clone();

        for (let l = L - 2; l >= 0; l--) {
            const wNextT = this.weights[l + 1].T;
            delta = nj.dot(wNextT, delta).multiply(this.act.df(z[l]));

            dW[l] = nj.dot(delta, a[l].T);
            db[l] = delta.clone();
        }

        return { dW, db };
    }

    step(grads, lr = 0.01) {
        for (let l = 0; l < this.weights.length; l++) {
            this.weights[l] = this.weights[l].subtract(grads.dW[l].multiply(lr));
            this.biases[l] = this.biases[l].subtract(grads.db[l].multiply(lr));
        }
    }
}
