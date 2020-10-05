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

		# set A, numQubits and qubitDim
		# qubitDim[0] : num column (X)
		# qubitDim[1] : num raw (Y)
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

		#self.B = np.eye(self.A.shape[0], M=self.A.shape[1], dtype=int) ## 1st single qubit gate idx.
		self.B = np.zeros(self.A.shape, dtype=int) ## 1st single qubit gate idx.
		self.quantumCircuit = None
		self.qubitMap = None

		self.singleQubitGate1 = '_'
		self.twoQubitGate = '_'
		self.singleQubitGate2 = '_'

		self.params = list()

		# Set edge name
		self.edgeName = dict() # (node1, node2) :name (node1 < node2)
		M = self.qubitDim[0]
		N = self.qubitDim[1]
		a ,b, c, d = 0, 0, 0, 0
		# horizontal
		for node in range(self.numQubits):
			# Right
			if node % M == N-1 :
				continue
			oddLine = (node // M)%2 == 1
			oddNode = (node %  M)%2 == 1
			if oddLine != oddNode :
				name = f"c{c}"
				c += 1
			else :
				name = f"a{a}"
				a += 1
			self.edgeName.update({(node, node+1) : name})			
		# vertical
		for node in range(self.numQubits):
			# Down
			if node < M :
				continue
			oddLine = (node // M)%2 == 1
			oddNode = (node %  M)%2 == 1
			if oddLine != oddNode :
				name = f"b{b}"
				b += 1
			else :
				name = f"d{d}"
				d += 1
			self.edgeName.update({(node-M, node) : name})
		emptyEdges = [(self.qubitIdxToGrid(x[0]), self.qubitIdxToGrid(x[1])) for x in self.edgeName.keys()]
		nameWithGrid = [ x for x in self.edgeName.values() ]
		if self.enableViz :
			self.vizRequest.setEmptyEdges(emptyEdges, nameWithGrid)
		if self.enableMatplot :
			self.matplotRequest.setEmptyEdges(emptyEdges, nameWithGrid)
		self.edgeNameInv = {self.edgeName[(k1, k2)]: (k1, k2) for k1, k2 in self.edgeName}
		#print(self.edgeName)
		#print(self.edgeNameInv)

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
		selected1X = list()
		selected1Y = list()
		selected2X = list()
		selected2Y = list()
		connection = list()
		for i in range(self.numQubits):
			xi, yi = self.qubitIdxToGrid(i)
			if self.B[i][i] :
				selected1X.append(xi)
				selected1Y.append(yi)
			if self.A[i][i] :
				selected2X.append(xi)
				selected2Y.append(yi)
			for j in range(0, i):
				if self.A[i][j] :
					xj, yj = self.qubitIdxToGrid(j)
					connection.append([[xj, yj],[xi, yi]])

		if self.enableViz :
			self.vizRequest.clear()
			emptyEdges = [(self.qubitIdxToGrid(sorted(x)[0]), self.qubitIdxToGrid(sorted(x)[1])) for x in self.edgeName.keys()]
			nameWithGrid = [ x for x in self.edgeName.values() ]
			self.matplotRequest.setEmptyEdges(emptyEdges, nameWithGrid)
			self.vizRequest.addPointAsGrid(self.qubitDim[0], self.qubitDim[1],True)
			self.vizRequest.addSelectPoint(selected1X, selected1Y, l=1)
			self.vizRequest.addSelectPoint(selected2X, selected2Y, l=2)
			self.vizRequest.addConnections(connection)
			self.vizRequest.request()
		if self.enableMatplot :
			self.matplotRequest.clear()
			emptyEdges = [(self.qubitIdxToGrid(sorted(x)[0]), self.qubitIdxToGrid(sorted(x)[1])) for x in self.edgeName.keys()]
			nameWithGrid = [ x for x in self.edgeName.values() ]
			self.matplotRequest.setEmptyEdges(emptyEdges, nameWithGrid)
			self.matplotRequest.addPointAsGrid(self.qubitDim[0], self.qubitDim[1], True)
			self.matplotRequest.addSelectPoint(selected1X, selected1Y, l=1)
			self.matplotRequest.addSelectPoint(selected2X, selected2Y, l=2)
			self.matplotRequest.addConnections(connection)
			self.matplotRequest.request()		

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

	def addConnectionsWithIndex(self, twoQubitGate, params, twoQubitIdxA, twoQubitIdxB, twoQubitIdxC, twoQubitIdxD):
		self.twoQubitGate = twoQubitGate
		self.params = params
		for idxA in twoQubitIdxA :
			name = f"a{idxA}"
			edge = self.edgeNameInv[name]
			self.connect(edge[0], edge[1])
		for idxB in twoQubitIdxB :
			name = f"b{idxB}"
			edge = self.edgeNameInv[name]
			self.connect(edge[0], edge[1])
		for idxC in twoQubitIdxC :
			name = f"c{idxC}"
			edge = self.edgeNameInv[name]
			self.connect(edge[0], edge[1])
		for idxD in twoQubitIdxD :
			name = f"d{idxD}"
			edge = self.edgeNameInv[name]
			self.connect(edge[0], edge[1])

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

	def addSelectPointsWithIndex(self, l, gate, idx):
		if l == 1 :
			self.singleQubitGate1 = gate
			self.B = np.zeros(self.B.shape)
			for i in idx : 
				self.B[i][i] = 1
		elif l == 2 :
			self.singleQubitGate2 = gate
			for i in idx :
				self.A[i][i] = 1
		else :
			print("invalid number of layer")
			raise ValueError
		self.drawfromA()

	def select(self, node):
		if isinstance(node, Iterable) and len(node)==2 and isinstance(node[0], int) and isinstance(node[1], int):
			node= node[0]+node[1]*self.qubitDim[0]
		elif isinstance(node, int):
			pass
		else :
			raise invalidArgumentTypeException(node, "int or iterable")
		self.A[node][node] = 1
		self.drawfromA()

	def unselect(self, node):
		if isinstance(node, Iterable) and len(node)==2 and isinstance(node[0], int) and isinstance(node[1], int):
			node= node[0]+node[1]*self.qubitDim[0]
		elif isinstance(node, int):
			pass
		else :
			raise invalidArgumentTypeException(node, "int or iterable")
		self.A[node][node] = 0
		self.drawfromA()

	def applySingleQubit(self, qc, qr, l):
		if l == 1 :
			gate = self.singleQubitGate1
		elif l == 2 :
			gate = self.singleQubitGate2
		else :
			raise ValueError
		if gate == "H":
			qc.h(qr)
		elif gate == "X":
			qc.x(qr)
		elif gate == "Y":
			qc.y(qr)
		elif gate == "Z":
			qc.z(qr)
		elif gate == "T":
			qc.t(qr)
		elif gate == "S":
			qc.s(qr)
		elif gate == "S_DAG":
			qc.sdg(qr)
		elif gate == "T_DAG":
			qc.tdg(qr)
		else :
			raise ValueError

	def applyTwoQubit(self, qc, qr1, qr2):
		if self.twoQubitGate == "CX":
			qc.cx(qr1, qr2)
		elif self.twoQubitGate == "CY":
			qc.cy(qr1, qr2)
		elif self.twoQubitGate == "CZ":
			qc.cz(qr1, qr2)
		elif self.twoQubitGate == "CRX":
			qc.crx(self.param[0], qr1, qr2)
		elif self.twoQubitGate == "CRY":
			qc.cry(self.param[0], qr1, qr2)
		elif self.twoQubitGate == "CRZ":
			qc.crz(self.param[0], qr1, qr2)
		elif self.twoQubitGate == "CU1":
			qc.cu1(self.param[0], qr1, qr2)
		elif self.twoQubitGate == "CU3":
			qc.cu3(self.param[0],self.param[1],self.param[2], qr1, qr2)
		else :
			raise ValueError

	def circuitImplementation(self, reduced, printCircuit):
		if not reduced :
			self.qubitMap = None
			qr = qsk.QuantumRegister(self.numQubits)
			self.quantumCircuit = qsk.QuantumCircuit(qr)
			M = self.qubitDim[0]
			N = self.qubitDim[1]
			# 1. 1st SingleQubit Gate
			for i in range(self.numQubits):
				if self.B[i][i] :
					self.applySingleQubit(self.quantumCircuit, qr[i], l=1)
			self.quantumCircuit.barrier()
			# 2. CZ Connect
			# 2-1. Up
			for i, hvec in enumerate(self.A):
				if i<M or i%2==1: continue # If it is the first raw no up.
				elif hvec[i-M] :
					self.applyTwoQubit(self.quantumCircuit, qr[i], qr[i-M])
			# 2-2. Right
			for i, hvec in enumerate(self.A):
				if i%M > M-2 or i%2==1 : continue # If it is the last column, no right
				elif hvec[i+1] :
					self.applyTwoQubit(self.quantumCircuit, qr[i], qr[i+1])
			# 2-3. Down
			for i, hvec in enumerate(self.A): # If it is the last raw, no down
				if i//M > N-2 or i%2==1 : continue
				elif hvec[i+M] :
					self.applyTwoQubit(self.quantumCircuit, qr[i], qr[i+M])
			# 2-4. Left
			for i, hvec in enumerate(self.A):
				if i%M < 1 or i%2==1 : continue
				elif hvec[i-1] :
					self.applyTwoQubit(self.quantumCircuit, qr[i], qr[i-1])
			# 3. apply S for diagonal
			self.quantumCircuit.barrier()
			for i, hvec in enumerate(self.A) :
				if hvec[i] :
					self.applySingleQubit(self.quantumCircuit, qr[i], l=2)
			# 4. hadamard
			#self.quantumCircuit.h(qr)
		else :
			qubitSet = set()
			qubitCZ = set()
			qubitS = set()
			M = self.qubitDim[0]
			N = self.qubitDim[1]
			# 2-1. Up
			for i, hvec in enumerate(self.A):
				if i<M or i%2==1: continue # If it is the first raw no up.
				elif hvec[i-M] :
					qubitCZ.add((i, i-M))
					qubitSet.add(i)
					qubitSet.add(i-M)
			# 2-2. Right
			for i, hvec in enumerate(self.A):
				if i%M > M-2 or i%2==1 : continue # If it is the last column, no right
				elif hvec[i+1] :
					qubitCZ.add((i, i+1))
					qubitSet.add(i)
					qubitSet.add(i+1)
			# 2-3. Down
			for i, hvec in enumerate(self.A): # If it is the last raw, no down
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
			for i in range(self.numQubits):
				if self.B[i][i] :
					self.applySingleQubit(self.quantumCircuit, qr[i], l=1)
			self.quantumCircuit.barrier()
			for con in qubitCZ :
				self.applyTwoQubit(self.quantumCircuit, qr[self.qubitMap.index(con[0])], qr[self.qubitMap.index(con[1])])
			self.quantumCircuit.barrier()
			for sel in qubitS : 
				self.applySingleQubit(self.quantumCircuit, qr[self.qubitMap.index(sel)], l=2)
			#self.quantumCircuit.h(qr)

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
				#print(iBin, trIdxBin)
				totalstate[int(trIdxBin, 2)] = p
		else :
			totalstate = state
		return np.array(totalstate)