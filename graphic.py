import matplotlib.pyplot as plt
from visdom import Visdom
from collections.abc import Iterable

#from QHLF import QHFL
from HLFErrors import *

class visdomRequest(object):
	def __init__(self):
		self.vis = Visdom()

		self.win = 'HLF'
		self.env = 'main'		
		self.globalLayout = {
			'xaxis': {'showgrid':False, 'showline':False, 'showticklabels':False},
			'yaxis': {'showgrid':False, 'showline':False, 'showticklabels':False},
		}

		self.PointsX = list()
		self.PointsY = list()
		self.PointsStyle = {
			'size' : 10,
			'symbol' : 'dot',
			'color' : '#e07b39',
		}

		self.selectedPointsX = list()
		self.selectedPointsY = list()
		self.selectedPointsStyle ={
			'size' : 17,
			'symbol' : 'dot',
			'color' : '#ff0000',
		}

		self.connections = list() # list of [[x0,y0],[x1,y1]]
		self.connectionsStyle = {
			'width': 3,
			'color': '#0e699e',
		}

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

	def isNeighbor(self, conn):
		x0 = conn[0][0]
		y0 = conn[0][1]
		x1 = conn[1][0]
		y1 = conn[1][1]
		return ((x0==x1) and (abs(y0-y1)==1)) or ((y0==y1) and (abs(x0-x1)==1))

	def clear(self):
		self.PointsX = list()
		self.PointsY = list()
		self.selectedPointsX = list()
		self.selectedPointsY = list()
		self.connections = list()
		self.request()

	def addPointWrapper(func):
		def inner(self, x, y):
			ret = False
			if isinstance(x, int) and isinstance(y, int):
				ret = func(self, x, y)
			elif isinstance(x, Iterable) and isinstance(y, Iterable):
				if len(x)!=len(y) : 
					raise invalidArgumentTypeException((x,y), "should be in same length")
				ret = all([func(self, x[i], y[i]) for i in range(len(x))])
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

	def addPointAsGrid(self, M, N):
		xListNotFlat = [[i]*N for i in range(M)]
		xList = list()
		for sub in xListNotFlat :
			xList = xList + sub
		yListNotFlat = [list(range(N))]*M
		yList = list()
		for sub in yListNotFlat :
			yList = yList + sub
		self.addPoint(xList, yList)

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
		dataPoints = {
			'x': self.PointsX,
			'y': self.PointsY,
			'marker': self.PointsStyle,
			'type': 'scatter',
			'mode': 'markers',
			'name': 'qubit'
		}
		dataSelectedPoints = {
			'x': self.selectedPointsX,
			'y': self.selectedPointsY,
			'marker': self.selectedPointsStyle,
			'type': 'scatter',
			'mode': 'markers',
			'name': 'S_qubit'
		}
		dataConnections = list()
		for conn in self.connections : 
			dataSingleConn ={
				'x':[conn[0][0], conn[1][0]],
				'y':[conn[0][1], conn[1][1]],
				'line':self.connectionsStyle,
				'type':'line',
				'mode':'lines',
				'name': 'Conn'
			}
			dataConnections.append(dataSingleConn)
		data = [dataPoints, dataSelectedPoints] + dataConnections
		opts = {}
		self.vis._send({'data': data, 'win': self.win, 'eid': self.env, 'layout': self.globalLayout, 'opts': opts})
'''
class graphic(object):
	def __init__(self, bind):
		# bind : a QHLF object to track on.
		self.bindQHLF = bind
'''
