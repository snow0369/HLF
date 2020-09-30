import numpy as np
import itertools
from collections.abc import Iterable

import qiskit as qsk
from qiskit import Aer

from graphic import visdomRequest, matplotRequest
from HLFErrors import *

class QHLF(object):
	def __init__(self, qubitDim=None, A=None, enableViz=False, enableMatplot=False):
		self.enableViz = enableViz
		self.enableMatplot = enableMatplot
		if self.enableViz :
			self.vizRequest = visdomRequest()
		if self.enableMatplot :
			self.matplotRequest = matplotRequest()
		if qubitDim == None and A == None :
			raise emptyInputException()
		elif qubitDim == None :
			if not isinstance(A, np.ndarray):
				self.A = np.asarray(A)
			else :
				self.A = A
			self.qubitDim = self.AtoQubitArr()
			if self.enableViz :
				self.vizRequest.addPointAsGrid(self.qubitDim[0],self.qubitDim[1],True)
			self.numQubits = self.A.shape[0]
		elif A == None : 
			self.qubitDim = qubitDim
			self.A = np.zeros((qubitDim[0]*qubitDim[1], qubitDim[0]*qubitDim[1]), dtype=bool)
			self.numQubits = self.qubitDim[0] * self.qubitDim[1]
			if self.enableViz :
				self.vizRequest.addPointAsGrid(self.qubitDim[0],self.qubitDim[1],True)
		else : 
			raise fullInputException()
		self.quantumCircuit = None
		self.qubitMap = None


	def checkSymmetry(self):
		if not isinstance(self.A, np.ndarray):
			self.A = np.asarray(self.A)
		return np.allclose(self.A, self.A.T, rtol=1e-5, atol=1e-8)

	def validityCheck(func):
		def inner(self, *args, **kwargs):
			if not isinstance(self.A, np.ndarray):
				self.A = np.asarray(self.A)
			if not self.checkSymmetry():
				raise notSymmetric(self.A)
			return func(*args, **kwargs)
		return inner

	def drawfromA(self):
		selectedX = list()
		selectedY = list()
		connection = list()
		for i in range(self.numQubits):
			xi, yi = self.qubitIdxToGrid(i)
			if self.A[i][i] :
				selectedX.append(xi)
				selectedY.append(yi)
			for j in range(0, i):
				if self.A[i][j] :
					xj, yj = self.qubitIdxToGrid(j)
					connection.append([[xi, yi],[xj, yj]])
		if self.enableViz :
			self.vizRequest.clear()
			self.vizRequest.addPointAsGrid(self.qubitDim[0], self.qubitDim[1],True)
			self.vizRequest.addSelectPoint(selectedX, selectedY)
			self.vizRequest.addConnections(connection)

	@validityCheck
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

	def qubitIdxToGrid(self, idx):
		return (idx%self.qubitDim[0], idx//self.qubitDim[1])

	def qubitGridToIdx(self, x, y):
		return x+self.qubitDim[0]*y

	def connect(self, i, j):
		M = self.qubitDim[0]
		N = self.qubitDim[1]
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
		self.drawfromA()

	def disconnect(self, i, j):
		M = self.qubitDim[0]
		N = self.qubitDim[1]
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
		self.A[n1][n2] = 0
		self.A[n2][n1] = 0
		self.drawfromA()

	def select(self, node):
		if isinstance(node, Iterable) and len(node)==2 and isinstance(node[0], int) and isinstance(node[1], int):
			node= node[0]+node[1]*self.qubitDim[0]
		else :
			raise invalidArgumentTypeException(node, "int or iterable")
		self.A[node][node] = 1
		self.drawfromA()

	def unselect(self, node):
		if isinstance(node, Iterable) and len(node)==2 and isinstance(node[0], int) and isinstance(node[1], int):
			node= node[0]+node[1]*self.qubitDim[0]
		else :
			raise invalidArgumentTypeException(node, "int or iterable")
		self.A[node][node] = 0
		self.drawfromA()

	def circuitImplementation(self, reduced, printCircuit):
		if not reduced :
			self.qubitMap = None
			qr = qsk.QuantumRegister(self.numQubits)
			self.quantumCircuit = qsk.QuantumCircuit(qr)
			M = self.qubitDim[0]
			N = self.qubitDim[1]
			# 1. hadamard
			self.quantumCircuit.h(qr)
			# 2. CZ Connect
			# 2-1. Up
			for i, hvec in enumerate(self.A):
				if i<M or i%2==1: continue # If it is the first row no up.
				elif hvec[i-M] :
					self.quantumCircuit.cz(i, i-M)
			# 2-2. Right
			for i, hvec in enumerate(self.A):
				if i%M > M-2 or i%2==1 : continue # If it is the last coloum, no right
				elif hvec[i+1] :
					self.quantumCircuit.cz(i, i+1)
			# 2-3. Down
			for i, hvec in enumerate(self.A): # If it is the last row, no down
				if i//M > N-2 or i%2==1 : continue
				elif hvec[i+M] :
					self.quantumCircuit.cz(i, i+M)
			# 2-4. Left
			for i, hvec in enumerate(self.A):
				if i%M < 1 or i%2==1 : continue
				elif hvec[i-1] :
					self.quantumCircuit.cz(i, i-1)
			# 3. apply S for diagonal
			for i, hvec in enumerate(self.A) :
				if hvec[i] :
					self.quantumCircuit.s(i)
			# 4. hadamard
			self.quantumCircuit.h(qr)
		else :
			qubitSet = set()
			qubitCZ = set()
			qubitS = set()
			M = self.qubitDim[0]
			N = self.qubitDim[1]
			# 2-1. Up
			for i, hvec in enumerate(self.A):
				if i<M or i%2==1: continue # If it is the first row no up.
				elif hvec[i-M] :
					qubitCZ.add((i, i-M))
					qubitSet.add(i)
					qubitSet.add(i-M)
			# 2-2. Right
			for i, hvec in enumerate(self.A):
				if i%M > M-2 or i%2==1 : continue # If it is the last coloum, no right
				elif hvec[i+1] :
					qubitCZ.add((i, i+1))
					qubitSet.add(i)
					qubitSet.add(i+1)
			# 2-3. Down
			for i, hvec in enumerate(self.A): # If it is the last row, no down
				if i//M > N-2 or i%2==1 : continue
				elif hvec[i+M] :
					qubitCZ.add((i, i+M))
					qubitSet.add(i)
					qubitSet.add(i+M)
			# 2-4. Left
			for i, hvec in enumerate(self.A):
				if i%M < 1 or i%2==1 : continue
				elif hvec[i-1] :
					qubitCZ.add((i, i-1))
					qubitSet.add(i)
					qubitSet.add(i-1)
			# 3. apply S for diagonal
			for i, hvec in enumerate(self.A) :
				if hvec[i] :
					qubitS.add(i)
					qubitSet.add(i)
			#qr = [qsk.QuantumRegister(1, name=f"q0_{i}") for i in sorted(list(qubitSet))]
			self.qubitMap=sorted(list(qubitSet))
			qr = qsk.QuantumRegister(len(self.qubitMap))
			self.quantumCircuit = qsk.QuantumCircuit(qr)
			self.quantumCircuit.h(qr)
			for con in qubitCZ :
				self.quantumCircuit.cz(qr[self.qubitMap.index(con[0])], qr[self.qubitMap.index(con[1])])
			for sel in qubitS : 
				self.quantumCircuit.s(qr[self.qubitMap.index(sel)])
			self.quantumCircuit.h(qr)

		if printCircuit :
			print(self.quantumCircuit)
			if(reduced) :
				for i,p in enumerate(self.qubitMap) :
					print(f"q0_{i} is qubit {p}")

	def performStatevectorSim(self):
		self.quantumCircuit = qsk.transpile(self.quantumCircuit)
		backend_SV = Aer.get_backend('statevector_simulator')
		job = qsk.execute(self.quantumCircuit, backend_SV)
		state = [x for x in job.result().get_statevector(self.quantumCircuit)]
		if self.qubitMap != None : 
			totalstate = np.zeros(2**self.numQubits, dtype=complex)
			for i, p in enumerate(state) :
				iBin = "{0:b}".format(i).rjust(len(self.qubitMap), "0")[::-1]
				trIdxBin = ""
				for j in range(self.numQubits): #From LSB
					if j not in self.qubitMap : 
						trIdxBin = "0"+trIdxBin
					else :
						trIdxBin = iBin[self.qubitMap.index(j)]+trIdxBin
				print(iBin, trIdxBin)
				totalstate[int(trIdxBin, 2)] = p
		else :
			totalstate = state
		return np.array(totalstate)