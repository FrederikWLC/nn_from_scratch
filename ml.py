import numpy as np
import math
import string

class Network:
  def __init__(self, shape, activations, labels=None, cost_func=None):
    assert len(activations) == len(shape), f"Amount of activations '{len(activations)}' must be equal to amount of layers {len(shape)}"
    self.shape = list(shape)
    self.activations = activations
    self.biases = [np.random.randn(l).tolist() for l in shape]
    self.weights = [np.random.randn(l,next_l).tolist() for l, next_l in zip(shape[:-1],shape[1:])]
    self.n_layers = len(shape)
    self.labels = labels if labels else [name(n=i+1,letters=string.ascii_lowercase+string.ascii_uppercase) for i in range(shape[-1])]
    if callable(cost_func): self.cost_func = cost_func
  
  def add_neuron(self, r, label=None):
    assert 0 < r < self.n_layers, f"Neuron layer number 'r={r}' cannot be out of range {self.n_layers-1}"
    assert not (r+1 == self.n_layers and (not label or label in self.labels)), f"All neurons in last layer must have a distinct 'label'"
    self.shape[r] += 1
    self.biases[r].append(np.random.randn())
    # If layer of this neuron is not first
    if r!=0:
      #Add previous layer's weights connected to this neuron
      for i in range(self.shape[r-1]):
        self.weights[r-1][i].append(np.random.randn())
    # If layer of this neuron is not last
    if r!=self.n_layers-1:
      #Add weights from this neuron connected to next layer
      self.weights[r].append(np.random.randn(shape[i+1]).tolist())
    else:
      self.labels.append(label)
      
  def remove_neuron(self, r, n):
    assert 0 < r < self.n_layers, f"Neuron layer index 'r={r}' cannot be out of range {self.n_layers-1}"
    assert 0 < n < self.shape[r], f"Neuron index 'n={n}' in layer 'r={r}' cannot be out of range {self.shape[r]-1}"
    self.shape[r] -= 1
    self.biases[r].pop(n)
    # If layer of neuron is not first
    if r!=0:
      #Remove previous weights connected to neuron
      for i in range(self.shape[r-1]):
        self.weights[r-1][i].pop()
      self.weights[r-1] = list(map(lambda wg: wg[:n] + wg[n+1:], self.weights[r-1]))
    # If layer of neuron is not last
    if r+1!=self.n_layers:
      #Remove weights from neuron connected to next layer
      self.weights[r].pop(n)
    else:
      self.labels.pop(n)

  def predict(self, x, track=False):
    assert np.shape(x) == (self.shape[0],), f"Input shape {np.shape(x)} must be same as input layer's {(self.shape[0],)}"
    assert self.shape[-1], f"The last layer must contain neurons! See current shape => shape={self.shape}"
    if track: Z, A = [], []
    for i in range(self.n_layers):
      x = np.add(x,self.biases[i])
      if track: Z.append(x)
      x = self.activations[i](x)
      if track: A.append(x)
      if i < self.n_layers-1:
        x = np.dot(x, self.weights[i])
    if track:
      return Z,A
    return list(np.nan_to_num(x))

  def predict_with_labels(self, x):
    return self.predict_with_and_without_labels[0]

  def predict_with_and_without_labels(self,x):
    predictions = self.predict(x)
    return dict(sorted(zip(self.labels, predictions), key=lambda x: x[1], reverse=True)), predictions

  def fit(self, x,y, lr=0.1):
    delta_biases, delta_weights = self.stochastic_gradients(X, Y)
    self.apply_gradients(delta_biases,delta_weights,lr=lr)
    
  def apply_gradients(self, delta_biases, delta_weights, lr=0.1):
    self.weights = [(W - lr * delta_W).tolist() for W, delta_W in zip(self.weights, delta_weights)]
    self.biases = [(b - lr * delta_b).tolist() for b, delta_b in zip(self.biases, delta_biases)]

  def stochastic_gradients(self, x,y):
    Z,A = self.predict(x,track=True)
    delta_weights = [np.zeros(np.shape(w)) for w in self.weights]
    delta_biases = [np.zeros(np.shape(b)) for b in self.biases]
    # Iterating backwards in the number of weight groups
    E = self.cost(y,A[-1])
    d = np.dot(E,self.activations[-1](Z[-1],prime=True))
    delta_biases[-1] = d
    for l in reversed(range(self.n_layers-1)):
      E = np.dot(np.transpose(self.weights[l]),E) # ADD TRANSPOSE
      d = np.dot(E,self.activations[l](Z[l],prime=True))
      delta_biases[l] = d
      delta_weights[l] = np.dot(d,np.transpose(A[l]))

    return delta_biases, delta_weights
  
  def do_gradient_descent(self, X,Y, num_of_batches=1, lr=0.1, print_loss=False, epoch=0):
    X_batches = np.array_split(X,num_of_batches)
    Y_batches = np.array_split(Y,num_of_batches)
    for X_batch, Y_batch in zip(X_batches,Y_batches):
      delta_weights_n = []
      delta_biases_n = []
      Z_n,A_n = zip(*[self.predict(x,track=True) for x in X_batch])
      O_n = [A[-1] for A in A_n]
      O_E = self.weighted_cost(Y_ideal=Y_batch, Y_real=O_n)
      if print_loss: print(f"epoch: {epoch} ; loss: {O_E}")
      for i, (x,y) in enumerate(zip(X_batch,Y_batch)):
        Z,A = Z_n[i], A_n[i]
        delta_weights = [np.zeros(np.shape(w)) for w in self.weights]
        delta_biases = [np.zeros(np.shape(b)) for b in self.biases]
        E = O_E #Output error
        d = np.multiply(E,self.activations[-1](Z[-1],prime=True))
        delta_biases[-1] = d
        for l in reversed(range(self.n_layers-1)):
          E = np.dot(E,np.transpose(np.atleast_2d(self.weights[l]))) # CALCULATE ERROR
          # d_W[L] = E x a[l+1]'(z[l+1]) x A_l.T
          # d_B[L] = E x a[l+1]'(z[l+1])'
          d = np.multiply(E,self.activations[l+1](Z[l+1],prime=True))
          delta_biases[l] = d
          delta_weights[l] = np.dot(np.atleast_2d(d), np.atleast_2d(A[l]).T)
        delta_biases_n.append(delta_biases)
        delta_weights_n.append(delta_weights)
      mean_delta_biases = np.mean(delta_biases_n,axis=0)
      mean_delta_weights = np.mean(delta_weights_n,axis=0)
      self.apply_gradients(mean_delta_biases,mean_delta_weights, lr=lr)
  
  def do_stochastic_gradient_descent(self, X, Y, lr=0.1, epoch=0, print_loss=False):
    self.do_gradient_descent(X,Y, num_of_batches=len(X),lr=lr, epoch=epoch, print_loss=print_loss)
  
  def train_stochastically(self, X, Y, epochs=10, lr=0.1, print_loss=True, epochs_per_print_loss=100):
    for j in range(epochs):
      print_loss_this_epoch = j % epochs_per_print_loss == 0 if print_loss else False
      self.do_stochastic_gradient_descent(X,Y,lr,print_loss=print_loss_this_epoch,epoch=j)

  def train_in_batches(self,X,Y,num_of_batches=1,epochs=10,lr=0.1, print_loss=False, epochs_per_print_loss=100):
    for j in range(epochs):
      print_loss_this_epoch = j % epochs_per_print_loss == 0 if print_loss else False
      self.do_gradient_descent(X,Y,num_of_batches,lr,print_loss_this_epoch,epoch=j)

  def train_from_labels(self, labeled_data, epochs=10, lr=0.1):
    # labeled_data = {"label1":{[3,2,...], [3,5,...]...}, "label2":{[5,1,...], [5,5,...]...}, ...}
    X,Y = zip(*[(inp,[1 if i == self.labels.index(label) else 0 for i in range(self.shape[-1])]) for label in labeled_data for inp in labeled_data[label]])
    self.train_in_batches(X,Y,epochs,lr)

  def test(self, X, Y):
    return np.array([self.cost_func(y,self.predict(x)) for x,y in zip(X,Y)]).mean()

  def cost(self, y_ideal,y_real):
    return np.square(np.subtract(y_ideal,y_real))

  def weighted_cost(self, Y_ideal, Y_real):
    return np.mean([self.cost(y_ideal=y,y_real=o) for y, o in zip(Y_ideal, Y_real)],axis=0)

# Defining some activation functions ----
@np.vectorize
def sigmoid(x, prime=False):
  if not prime:
    if x < 0:
      return 1-1/(1+np.exp(x))
    else:
      return 1/(1+np.exp(-x))
  else:
    return sigmoid(x)*(1-sigmoid(x))

def softmax(x, prime=False):
    """Compute the softmax of vector x."""
    if not prime:
      # subtracting np.max(x) to prevent overflow
      e = np.exp(x - np.max(x))
      return e / np.sum(e)
    else:
      # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
      s = softmax(x).reshape(-1,1)
      return np.diagflat(s) - np.dot(s, s.T)

def ReLu(x, prime=False):
  return np.maximum(0,x) if not prime else np.maximum(np.maximum(0, x),1)

def SiLu(x, prime=False):
  return x*sigmoid(x) if not prime else sigmoid(x)+x*sigmoid(x,prime=True)

def lin(x, prime=False):
  return x if not prime else np.ones_like(x)

# -----------------

def name(n, letters, len_letters=None):
  assert n>0, "n must be a positive integer"
  if not len_letters: len_letters = len(letters)
  chunk = (n//len_letters)*len_letters
  return name(chunk//len_letters, letters=letters, len_letters=len_letters)+letters[n-chunk-1] if n>len_letters else letters[n-chunk-1]

# Tests - THE TRAINING DOESN't really WORK! ;(
if __name__ == '__main__':
  np.random.seed(5)
  # TRAIN ON 1 DATA POINT - IT ONLY WORKS FOR ONE data point AND ONE OUTPUT NEURON...
  nn = Network(shape=(2,1), activations=[SiLu,sigmoid])
  print(f"First prediction: {nn.predict([1,1])}")
  nn.train_in_batches(X=[[1,1]],Y=[[0.65]], lr=10,epochs=10000, print_loss=True, epochs_per_print_loss=10)
  print(f"Second prediction: {nn.predict([1,1])}")



