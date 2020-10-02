import matplotlib.pyplot as plt
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
		self.selectedPointsX = list()
		self.selectedPointsY = list()

		self.emptyEdges = list()
		self.EdgeLabel = list()
		self.connections = list() # list of [[x0,y0],[x1,y1]]

	def isInPoint(self, x, y):
		try :
			idxList = [i for i in range(len(self.PointsX)) if self.PointsX[i]==x]
			return any([self.PointsY[i]==y for i in idxList])
		except ValueError :
			return False

	def isInSelectedPoint(self, x, y):
		try :
			idxList = [i for i in range(len(self.selectedPointsX)) if self.selectedPointsX[i]==x]
			return any([self.selectedPointsY[i]==y for i in idxList])
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
		self.selectedPointsX = list()
		self.selectedPointsY = list()

		self.emptyEdges = list()
		self.EdgeLabel = list()
		self.connections = list()
		self.request()

	def addPointWrapper(func):
		def inner(self, x, y, autoLabel, label):
			ret = False
			if isinstance(x, int) and isinstance(y, int):
				ret = func(self, x, y)
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
				ret = all([func(self, x[i], y[i]) for i in range(len(x))])
				if ret :
					if autoLabel and len(self.PointsLabel) > 0 :
						labelToAdd = list(range(max(self.PointsLabel)+1, self.PointsLabel+1+len(x)))
					elif autoLabel :
						labelToAdd = list(range(len(x)))
					else :
						if len(x) != len(label):
							raise invalidArgumentTypeException((x,label), "should be in same length")
						labelToAdd = label
					self.PointsLabel = self.PointsLabel + labelToAdd
			self.request()
			return ret
		return inner

	@addPointWrapper
	def addPoint(self, x, y):
		if self.isInPoint(x,y):
			print(f"({x}, {y}) already exists")
			return True
		self.PointsX.append(x)
		self.PointsY.append(y)
		return True

	@addPointWrapper
	def addSelectPoint(self, x, y):
		if not self.isInPoint(x,y):
			raise invalidPointException((x,y))
		if self.isInSelectedPoint(x,y):
			print(f"({x}, {y}) already exists")
			return True
		self.selectedPointsX.append(x)
		self.selectedPointsY.append(y)
		return True

	def addPointAsGrid(self, M, N, autoLabel=True, label=list()):
		xListNotFlat = [[i]*N for i in range(M)]
		xList = list()
		for sub in xListNotFlat :
			xList = xList + sub
		yListNotFlat = [list(range(N))]*M
		yList = list()
		for sub in yListNotFlat :
			yList = yList + sub
		self.addPoint(xList, yList, autoLabel, label)

	def setEmptyEdges(self, edges, labels):
		self.emptyEdges = edges
		self.EdgeLabel = labels


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
			self.connections.append(con)
		self.request()

	def removeSelectedPoint(self, x, y):
		if not self.isInSelectedPoint(x,y):
			print(f"point ({x}, {y}) is not selected.")
			return True
		idxList = [i for i in range(len(self.selectedPointsX)) if self.selectedPointsX[i]==x]
		idx = [self.selectedPointsY[i]==y for i in idxList][0]
		del self.selectedPointsX[idx]
		del self.selectedPointsY[idx]

	def removeConnection(self, conRm):
		pass

	def removePoint(self, x, y):
		pass

	def request(self):
		raise NotImplementedError

class matplotRequest(Graphic):
	def __init__(self):
		super(self)
		plt.ion()
		self.textBoxProp = dict(boxstyle='round', facecolor='white')
		self.textBoxPropSelected = dict(boxstyle='round', facecolor='blue')
		self.emptyEdgeStyle = dict(linestyle='dashed', linewidth=4)
		self.connectedEdgeStyle = dict(linewidth=4)
	def request(self):
		plt.clf()
		# Draw Edges
		for i in range(len(self.emptyEdges)) :
			plt.plot([self.emptyEdges[i][0][0], self.emptyEdges[i][1][0]], [self.emptyEdges[i][0][1], self.emptyEdges[i][1][1]], **self.emptyEdgeStyle)
		for i in range(len(self.connections))
			plt.plot([self.connection[i][0][0], self.connection[i][1][0]], [self.connection[i][0][1], self.connection[i][1][1]], **self.connectedEdgeStyle)
		# Draw nodes
		for i in range(len(self.PointsX)):
			x, y = self.PointsX[i], self.PointsY[i]
			if self.isInSelectedPoint(x,y) :
				prop = self.textBoxPropSelected
			else :
				prop = self.textBoxProp
			plt.text(self.PointsX[i], self.PointsY[i], str(self.PointsLabel[i]), bbox=prop)

class visdomRequest(Graphic)
	def __init__(self):
		super(self)
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
			'x': self.selectedPointsX,
			'y': self.selectedPointsY,
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
