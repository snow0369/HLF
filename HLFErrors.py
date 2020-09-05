# Errors
class notSymmetricException(Exception):
	# Given matrix is not symmetric
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
		self.message = f"{n1}, {n2} has diff of {abs(n1-n2)}, required: {rD}"


# graphic error
class invalidPointException(Exception):
	def __init__(self, p):
		self.message = f"{p} is not in the point list"

class notNeighborException(Exception):
	def __init__(self, p0, p1):
		self.message = f"{p0} and {p1} is not neighboring."