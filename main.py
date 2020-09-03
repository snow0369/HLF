import numpy as np
import itertools
from collections.abc import Iterable
import qiskit as qsk

# Errors
class notSymmetricException(Exception):
	def __init__(self, A):
		self.expression = A.__repr__()
		self.message = f"{A} is not symmetric."
class emptyInputException(Exception):
	def __init__(self):
		self.message = "Specify one input field."
class fullInputException(Exception):
	def __init__(self):
		self.message = "Specify only one input field."
class invalidAmatException(Exception):
	def __init__(self, A):
		self.expression = A.__repr__() 
		self.message = f"{A} has invalid connection."
class invalidArgumentTypeException(Exception):
	def __init__(self, a, r=""):
		self.expression = a
		self.message = f"{a} has invalid type of {type(a)}, required: {r}"
class invalidConnectionNodesException(Exception):
	def __init__(self, n1, n2, rD):
		self.expression = (n1, n2).__repr__()
		self.message = f"{n1}, {n2} has diff of {abs(n1-n2)}, reauired: {rD}"


class HLF(object):
	def __init__(self, qubitDim, A):
		if qubitDim == None and A == None :
			raise emptyInputException()
		elif qubitDim == None :
			if not isinstance(A, np.ndarray):
				self.A = np.asarray(A)
			else :
				self.A = A
			self.qubitDim = self.AtoQubitArr()
			self.numQubits = self.A.shape[0]
		elif A == None : 
			self.qubitDim = qubitDim
			self.A = np.zeros(qubitDim, dtype=bool)
			self.numQubits = self.qubitDim[0] * self.qubitDim[1]
		else : 
			raise fullInputException()
		self.quantumCircuit = None
		self.quantumRegister = None


	def checkSymmetry(self):
		if not isinstance(self.A, np.ndarray):
			self.A = np.asarray(self.A)
		return np.allclose(self.A, self.A.T, rtol=1e-5, atol=1e-8)

	def validityCheck(self, func):
		def inner(*args, **kwargs):
			if not isinstance(self.A, np.ndarray):
				self.A = np.asarray(self.A)
			if not self.checkSymmetry():
				raise notSymmetric(self.A)
			return func(*args, **kwargs)
		return inner

	@self.validityCheck
	def reduceAmat(self):
		ret = list()
		for i in range(self.A.shape[0]):
			if any(self.A[i]) :
				ret.append(self.A[i])
		return asarray(ret)

	def AtoQubitArr(self):
		self.A = self.reduceAmat()
		# Horizontal : M
		# Vertical : N
		M = 1
		for i in range(self.A.shape[0]):
			trueIdxs = np.argwhere(self.A[i])
			trueIdxs = trueIdxs[trueIdxs != i]
			if  ((i==0 or i==self.A.shape[0]-1) and len(trueIdxs)>2) or (len(trueIdxs)>4) :
				raise invalidAmatException(self.A)

			verticalConn = [abs(i-j) for j in trueIdxs if j!=i and j!=i+1 and j!=i-1]
			if len(verticalConn == 0) :
				pass
			elif len(verticalConn == 1):
				if M == 1 : 
					M = verticalConn[0]
				elif M != verticalConn[0]:
					raise invalidAmatException(self.A)
			else :
				raise invalidAmatException(self.A)
		N = (self.A.shape[0]-1)//M +1
		return (M, N)

	def connect(self, i, j):
		nodes = list()
		for node in [i,j]:
			if isinstance(node, int):
				nodes.append(node)
			elif isinstance(node, Iterable) and len(node)==2 and isinstance(node[0], int) and isinstance(node[1], int):
				nodes.append(node[0]+node[1]*self.qubitDim[0])
			else :
				raise invalidArgumentTypeException(node, "int or iterable")
		n1 = nodes[0]
		n2 = nodes[1]
		if not ((abs(n1-n2)==1) or (abs(n1-n2)==M)):
			raise invalidConnectionNodesException(n1, n2, (1, M))
		self.A[n1][n2] = 1
		self.A[n2][n1] = 1

	def circuitImplementation(self):
		self.quantumRegister = qsk.QuantumRegister(self.numQubits)
		self.quantumCircuit = qsk.QuantumCircuit(self.quantumRegister)
		# 1. hadamard
		self.quantumCircuit.h(self.quantumRegister)
		# 2. CZ Connect
		# 2-1. Up
		for i, hvec in enumerate(self.A):
			if i<M : continue
			elif hvec[i-M] :
				self.quantumCircuit.cz(i, i-M)
		# 2-2. Right
		for i, hvec in enumerate(self.A):
			if i%M > M-1 : continue
			elif hvec[i+1] :
				self.quantumCircuit.cz(i, i+1)
		# 2-3. Down
		for i, hvec in enumerate(self.A):
			if i//M > N-1 : continue
			elif hvec[i+M] :
				self.quantumCircuit.cz(i, i+M)
		# 2-4. Left
		for i, hvec in enumerate(self.A):
			if i%M < 1 : continue
			elif hvec[i+M] :
				self.quantumCircuit.cz(i, i-1)
		# 
