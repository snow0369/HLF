from QHLF import QHLF

if __name__ == '__main__':
	## 1-1. Determine N
	N = '_'
	while not isinstance(N, int):
		try :
			N = input("[N=?]")
			N = int(N)
		except ValueError : 
			print("Input proper an integer value.")

	## 1-2. make QHLF object with grid N
	qhlf = QHLF(qubitDim=(N,N))

	## 2. Single qubit gate
	singleQubitGate = '_'
	singleQubitIdx = list()
	singleQubitGateList = {0:'H', 1:'X', 2:'Y', 3:'Z', 4:'T', 5:'S', 6:'S_DAG', 7:'T_DAG'}
	while not singleQubitGate in singleQubitGateList.values() :
		try :
			print("Single qubit gate : ")
			print(singleQubitGateList)
			singleQubitGate = input()
			
			print("qubit index (seperate with space) : ")
			idxStr = input()
			singleQubitIdx = [ int(x) for x in idxStr.split(" ") ]
		except :
			print("invalid input")

	## 3-1. two qubit singleQubitGate
	twoQubitGate = '_'
	twoQubitIdx = list()
	twoQubitGateList = {0:'H', 1:'X', 2:'Y', 3:'Z', 4:'T', 5:'S', 6:'S_DAG', 7:'T_DAG'}
	while not twoQubitGate in twoQubitGateList.values() :
		try :
			print("Single qubit gate : ")
			print(twoQubitGateList)
			twoQubitGate = input()
			
			print("qubit index (seperate with space) : ")
			idxStr = input()
			twoQubitIdx = [ int(x) for x in idxStr.split(" ") ]
		except :
			print("invalid input")
