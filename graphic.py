import matplotlib.pyplot as plt

import numpy as np
from visdom import Visdom
from collections.abc import Iterable
from itertools import product

#from QHLF import QHFL
from HLFErrors import *

class Graphic(object):
	def __init__(self):
		self.PointsX = list()
		self.PointsY = list()
		self.PointsLabel = list()
		self.selectedPoints1X = list()
		self.selectedPoints1Y = list()
		self.selectedPoints2X = list()
		self.selectedPoints2Y = list()

		self.emptyEdges = list()
		self.edgeLabel = dict()
		self.connections = list() # list of [[x0,y0],[x1,y1]]

	def isInPoint(self, x, y):
		try :
			idxList = [i for i in range(len(self.PointsX)) if self.PointsX[i]==x]
			return any([self.PointsY[i]==y for i in idxList])
		except ValueError :
			return False

	def isInSelectedPoint(self, x, y, l):
		if not l in [1,2] : 
			raise ValueError
		selectedX = self.selectedPoints1X if l==1 else self.selectedPoints2X
		selectedY = self.selectedPoints1Y if l==1 else self.selectedPoints2Y
		try :
			idxList = [i for i in range(len(selectedX)) if selectedX[i]==x]
			return any([selectedY[i]==y for i in idxList])
		except ValueError :
			return False

	def isInConnections(self, conn):
		return conn in self.connections

	def isNeighbor(self, conn):
		x0 = conn[0][0]
		y0 = conn[0][1]
		x1 = conn[1][0]
		y1 = conn[1][1]
		return ((x0==x1) and (abs(y0-y1)==1)) or ((y0==y1) and (abs(x0-x1)==1))

	def clear(self):
		self.PointsX = list()
		self.PointsY = list()
		self.PointsLabel = list()
		self.selectedPoints1X = list()
		self.selectedPoints1Y = list()
		self.selectedPoints2X = list()
		self.selectedPoints2Y = list()

		self.emptyEdges = list()
		self.edgeLabel = dict()
		self.connections = list()
		#self.request()

	def addPointWrapper(func):
		def inner(self, x, y, autoLabel=True, label=list(), l=1):
			ret = False
			if isinstance(x, int) and isinstance(y, int):
				ret = func(self, x, y, l)
				if ret :
					if autoLabel and len(self.PointsLabel) > 0 :
						labelToAdd = max(self.PointsLabel)+1
					elif autoLabel :
						labelToAdd = 0
					else : #manual label
						labelToAdd = label
					self.PointsLabel.append(labelToAdd)
			elif isinstance(x, Iterable) and isinstance(y, Iterable):
				if len(x)!=len(y) : 
					raise invalidArgumentTypeException((x,y), "should be in same length")
				ret = all([func(self, x[i], y[i], l) for i in range(len(x))])
				if ret :
					if autoLabel and len(self.PointsLabel) > 0 :
						labelToAdd = list(range(max(self.PointsLabel)+1, max(self.PointsLabel)+1+len(x)))
					elif autoLabel :
						labelToAdd = list(range(len(x)))
					else :
						if len(x) != len(label):
							raise invalidArgumentTypeException((x,label), "should be in same length")
						labelToAdd = label
					self.PointsLabel = self.PointsLabel + labelToAdd
			#self.request()
			return ret
		return inner

	@addPointWrapper
	def addPoint(self, x, y, _):
		if self.isInPoint(x,y):
			print(f"({x}, {y}) already exists")
			return True
		self.PointsX.append(x)
		self.PointsY.append(y)
		return True

	@addPointWrapper
	def addSelectPoint(self, x, y, l):
		if not self.isInPoint(x,y):
			raise invalidPointException((x,y))
		if self.isInSelectedPoint(x,y,l):
			print(f"({x}, {y}) already exists")
			return True
		if l == 1 :
			self.selectedPoints1X.append(x)
			self.selectedPoints1Y.append(y)
		elif l == 2 :
			self.selectedPoints2X.append(x)
			self.selectedPoints2Y.append(y)
		else :
			raise ValueError
		return True

	def addPointAsGrid(self, M, N, autoLabel=True, label=list()):
		xList = np.arange(M)
		yList = np.arange(N)
		yxList = list(product(yList, xList))
		xList = [k[1] for k in yxList]
		yList = [k[0] for k in yxList]
		self.addPoint(xList, yList, autoLabel, label)

	def setEmptyEdges(self, edges, labels):
		#print("Called")
		#print(edges)
		self.emptyEdges = edges
		self.edgeLabel = labels


	def addConnections(self, conns):
		# conns : array of [[x0,y0], [x1,y1]]
		for con in conns : 
			if not self.isInPoint(con[0][0], con[0][1]):
				raise invalidPointException((con[0][0], con[0][1]))
			if not self.isInPoint(con[1][0], con[1][1]):
				raise invalidPointException((con[1][0], con[1][1]))
			if con in self.connections or con[::-1] in self.connections:
				print(f"{con} is already in the list.")
				continue
			if not self.isNeighbor(con) :
				raise notNeighborException(con[0], con[1])
			con = tuple((tuple(con[0]), tuple(con[1])))
			self.connections.append(con)
		#self.request()

	def removeSelectedPoint(self, x, y, l):
		if not l in [1,2] : 
			raise ValueError
		selectedX = self.selectedPoints1X if l==1 else self.selectedPoints2X
		selectedY = self.selectedPoints1Y if l==1 else self.selectedPoints2Y
		if not self.isInSelectedPoint(x,y,l):
			print(f"point ({x}, {y}) is not selected.")
			return True
		idxList = [i for i in range(len(selectedX)) if selectedX[i]==x]
		idx = [selectedY[i]==y for i in idxList][0]
		del selectedX[idx]
		del selectedY[idx]

	def removeConnection(self, conRm):
		pass

	def removePoint(self, x, y):
		pass

	def request(self):
		raise NotImplementedError

class matplotRequest(Graphic):
	def __init__(self):
		super().__init__()
		plt.ion()
		self.textBoxProp = dict(boxstyle='round', facecolor='white')
		self.textBoxPropSelected1 = dict(boxstyle='round', facecolor='cyan')
		self.textBoxPropSelected2 = dict(boxstyle='round', facecolor='greenyellow')
		self.textBoxPropSelected12 = dict(boxstyle='round', facecolor='orchid')
		self.emptyEdgeStyle = dict(linestyle='dotted', linewidth=1)
		self.connectedEdgeStyle = dict(linestyle='solid', linewidth=2)
		self.edgeColor = 'rbmc'

	def midPoint(self, p1, p2):
		return (0.5*(p1[0]+p2[0]), 0.5*(p1[1]+p2[1]))

	def request(self):
		#print(self.PointsLabel)
		#print(self.PointsX)
		#print(self.PointsY)
		#print(self.emptyEdges)
		#print(self.edgeLabel)
		#print(self.selectedPoints1X)
		#print(self.selectedPoints2X)
		#print(self.connections)
		plt.clf()
		if len(self.PointsX) > 0 and len(self.PointsY) > 0 :
			plt.xlim(min(self.PointsX)-0.5, max(self.PointsX)+0.5)
			plt.ylim(min(self.PointsY)-0.5, max(self.PointsY)+0.5)
		plt.axis('off')
		# Draw Edges
		for i in range(len(self.emptyEdges)) :
			mp = self.midPoint(self.emptyEdges[i][0], self.emptyEdges[i][1])
			txt = self.edgeLabel[i]
			cIdx = (ord(txt[0])-ord('a'))%4
			color = self.edgeColor[cIdx]
			if self.emptyEdges[i] in self.connections :
				prop = self.connectedEdgeStyle
			else :
				prop = self.emptyEdgeStyle
			plt.plot([self.emptyEdges[i][0][0], self.emptyEdges[i][1][0]], [self.emptyEdges[i][0][1], self.emptyEdges[i][1][1]], color=color, **prop)
			if txt[0] in ['a', 'c'] :
				plt.text(mp[0], mp[1]+0.1, txt)
			else :
				plt.text(mp[0]+0.05, mp[1], txt)
		# Draw nodes
		for i in range(len(self.PointsX)):
			x, y = self.PointsX[i], self.PointsY[i]
			sel1 = self.isInSelectedPoint(x,y,1)
			sel2 = self.isInSelectedPoint(x,y,2)
			if sel1 and sel2 :
				prop = self.textBoxPropSelected12
			elif sel1 :
				prop = self.textBoxPropSelected1
			elif sel2 :
				prop = self.textBoxPropSelected2
			else :
				prop = self.textBoxProp
			plt.text(self.PointsX[i], self.PointsY[i], str(self.PointsLabel[i]), bbox=prop)
		plt.show()
		plt.pause(0.01)

class visdomRequest(Graphic):
	def __init__(self):
		super().__init__()
		self.vis = Visdom()
		self.win = 'HLF'
		self.env = 'main'		
		self.globalLayout = {
			'xaxis': {'showgrid':False, 'showline':False, 'showticklabels':False},
			'yaxis': {'showgrid':False, 'showline':False, 'showticklabels':False},
		}
		self.PointsStyle = {
			'size' : 10,
			'symbol' : 'dot',
			'color' : '#e07b39',
		}
		self.selectedPointsStyle ={
			'size' : 17,
			'symbol' : 'dot',
			'color' : '#ff0000',
		}
		self.connectionsStyle = {
			'width': 3,
			'color': '#0e699e',
		}

	def request(self):
		dataPoints = {
			'x': self.PointsX,
			'y': self.PointsY,
			'marker': self.PointsStyle,
			'type': 'scatter',
			'mode': 'markers',
			'name': 'qubit',
			'showlegend' : False
		}
		dataSelectedPoints = {
			'x': self.selectedPoints2X,
			'y': self.selectedPoints2Y,
			'marker': self.selectedPointsStyle,
			'type': 'scatter',
			'mode': 'markers',
			'name': 'S_qubit',
			'showlegend' : False
		}
		dataConnections = list()
		for conn in self.connections : 
			dataSingleConn ={
				'x':[conn[0][0], conn[1][0]],
				'y':[conn[0][1], conn[1][1]],
				'line':self.connectionsStyle,
				'type':'line',
				'mode':'lines',
				'name': 'Conn',
				'showlegend' : False
			}
			dataConnections.append(dataSingleConn)
		data = [dataPoints, dataSelectedPoints] + dataConnections
		opts = {'showlegend' : False}
		self.vis._send({'data': data, 'win': self.win, 'eid': self.env, 'layout': self.globalLayout, 'opts': opts})


'''
class graphic(object):
	def __init__(self, bind):
		# bind : a QHLF object to track on.
		self.bindQHLF = bind
'''
