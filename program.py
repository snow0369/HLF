from QHLF import QHLF
import sys
import numpy as np
from itertools import product

singleQubitGateList = {0:'H', 1:'X', 2:'Y', 3:'Z', 4:'T', 5:'S', 6:'S_DAG', 7:'T_DAG'}

twoQubitGateList = {0:'CX', 1:'CY', 2:'CZ', 3:'CRX', 4:'CRY', 5:'CRZ', 6:'CU1', 7:'CU3'}
oneParamGate = ['CRX', 'CRY', 'CRZ', 'CU1']
twoParamGate = []
threeParamGate = ['CU3']

def inputQuit(desc=""):
	inp = input('>> ')
	if inp == 'x' :
		quit()
		sys.exit(0)
	return inp

def inputSingleQubit(desc, N):
	singleQubitGate = '_'
	singleQubitIdx = list()
	while not singleQubitGate in singleQubitGateList.values() :
		try :
			print(f"Single qubit gate {desc}: ")
			print(singleQubitGateList)
			singleQubitGate = inputQuit()
			if singleQubitGate.isnumeric():
				singleQubitGate = singleQubitGateList[int(singleQubitGate)]
			else :
				singleQubitGate = singleQubitGate.upper()
			if not singleQubitGate in singleQubitGateList.values() :
				raise ValueError
			print("qubit index (seperate by spaces, ':' for all) : ")
			idxStr = inputQuit()
			if idxStr == ':' :
				singleQubitIdx = list(range(N*N))
			elif len(idxStr) == 0 :
				singleQubitIdx = list()
			else :
				singleQubitIdx = sorted(list(set([ int(x) for x in idxStr.split(" ") ])))
			if not all([ x in list(range(N*N)) for x in singleQubitIdx]) :
				raise ValueError
		except ValueError :
			singleQubitGate = '_'
			singleQubitIdx = list()
			print(f"invalid input SQ{desc}")
		except (SystemExit, KeyboardInterrupt) :
			print("terminate program")
			quit()
			sys.exit(0)
		except :
			print("Unexpected error:", sys.exc_info())
			quit()
			sys.exit(0)
	return singleQubitGate, singleQubitIdx

def inputTwoQubit(N):
	twoQubitGate = '_'
	twoQubitIdxA = list()
	twoQubitIdxB = list()
	twoQubitIdxC = list()
	twoQubitIdxD = list()
	params = list()
	maxEdgeIdx = (((N+1)//2) + ((N-1)//2)) * (N//2)

	while not twoQubitGate in twoQubitGateList.values() :
		try :
			print("Two qubit gate : ")
			print(twoQubitGateList)
			twoQubitGate = inputQuit()

			if twoQubitGate.isnumeric():
				twoQubitGate = twoQubitGateList[int(twoQubitGate)]
			else :
				twoQubitGate = twoQubitGate.upper()
			if not twoQubitGate in twoQubitGateList.values() :
				raise ValueError
			if twoQubitGate in oneParamGate :
				print("input params(1) : ")
				p = float(inputQuit())
				params = [p]
			elif twoQubitGate in twoParamGate :
				print("input params(2) : ")
				p = inputQuit()
				p = [ float(x) for x in p.split(" ") ]
				assert len(p)==2
				params = p
			elif twoQubitGate in threeParamGate :
				print("input params(3) : ")
				p = inputQuit()
				p = [ float(x) for x in p.split(" ") ]
				assert len(p)==3
				params = p
			else :
				params = list()
			
			print("Edge index (seperate with space) (a): ")
			idxStr = inputQuit()
			if len(idxStr) == 0 :
				twoQubitIdxA = list()
			else :
				twoQubitIdxA = sorted(list(set([ int(x) for x in idxStr.split(" ") ])))
			if not all([ x in list(range(maxEdgeIdx)) for x in twoQubitIdxA]) :
				raise ValueError

			print("Edge index (seperate with space) (b): ")
			idxStr = inputQuit()
			if len(idxStr) == 0 :
				twoQubitIdxB = list()
			else :
				twoQubitIdxB = sorted(list(set([ int(x) for x in idxStr.split(" ") ])))
			if not all([ x in list(range(maxEdgeIdx)) for x in twoQubitIdxB]) :
				raise ValueError

			print("Edge index (seperate with space) (c): ")
			idxStr = inputQuit()
			if len(idxStr) == 0 :
				twoQubitIdxC = list()
			else :
				twoQubitIdxC = sorted(list(set([ int(x) for x in idxStr.split(" ") ])))
			if not all([ x in list(range(maxEdgeIdx)) for x in twoQubitIdxC]) :
				raise ValueError

			print("Edge index (seperate with space) (d): ")
			idxStr = inputQuit()
			if len(idxStr) == 0 :
				twoQubitIdxD = list()
			else :
				twoQubitIdxD = sorted(list(set([ int(x) for x in idxStr.split(" ") ])))
			if not all([ x in list(range(maxEdgeIdx)) for x in twoQubitIdxD]) :
				raise ValueError
		except (ValueError, AssertionError) :
			twoQubitGate = '_'
			twoQubitIdxA = list()
			twoQubitIdxB = list()
			twoQubitIdxC = list()
			twoQubitIdxD = list()
			params = list()
			print("invalid input TQ")
		except (SystemExit, KeyboardInterrupt) :
			print("terminate program")
			quit()
			sys.exit(0)
		except :
			print("Unexpected error:", sys.exc_info()[0])
			quit()
			sys.exit(0)
	return twoQubitGate, params, twoQubitIdxA, twoQubitIdxB, twoQubitIdxC, twoQubitIdxD


if __name__ == '__main__':
	## 1-1. Determine N
	N = '_'
	print("Input 'x' whenever you want to leave.")
	while not isinstance(N, int):
		try :
			print("[N=?]")
			N = inputQuit()
			N = int(N)
		except ValueError : 
			print("Input an integer value.")
		except (SystemExit, KeyboardInterrupt) :
			print("terminate program")
			quit()
			sys.exit(0)
		except :
			print("Unexpected error:", sys.exc_info()[0])
			quit()
			sys.exit(0)
	## 1-2. make QHLF object with grid N
	qhlf = QHLF(qubitDim=(N,N), enableMatplot=True)
	qhlf.drawfromA()

	## 2. Single qubit gate
	singleQubitGate1, singleQubitIdx1 = inputSingleQubit("(1)", N)
	qhlf.addSelectPointsWithIndex(1, singleQubitGate1, singleQubitIdx1)

	## 3. two qubit singleQubitGate
	twoQubitGate, params, twoQubitIdxA, twoQubitIdxB, twoQubitIdxC, twoQubitIdxD = inputTwoQubit(N)
	qhlf.addConnectionsWithIndex(twoQubitGate, params, twoQubitIdxA, twoQubitIdxB, twoQubitIdxC, twoQubitIdxD)

	## 4. Single qubit gate
	singleQubitGate2, singleQubitIdx2 = inputSingleQubit("(2)", N)
	qhlf.addSelectPointsWithIndex(2, singleQubitGate2, singleQubitIdx2)

	## 5. Peform simulation
	ans = '_'
	while not ans in ['y', 'n']:
		print("remove unused qubits? [y/n]")
		ans = input(">> ")
	if ans == 'y':
		ans = True
	else :
		ans = False
	qhlf.circuitImplementation(ans, True)
	sv = qhlf.performStatevectorSim()
	p  = np.abs(sv)
	maxBasis = format(np.argmax(p), 'b').rjust(N*N, "0")[::-1]
	print("result")
	for i in range(N-1, -1, -1):
		print(" ".join(maxBasis[i*N:(i+1)*N]))

	## 6. Terminate
	_ = input("Press enter key to exit.")