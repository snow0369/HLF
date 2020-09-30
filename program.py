from QHLF import QHLF

if __name__ == '__main__':
	## Determine N
	N = '_'
	while not isinstance(N, int):
		try :
			N = input("[N=?]")
			N = int(N)
		except ValueError : 
			print("Input proper an integer value.")

	## make QHLF object with grid N
	qhlf = QHLF(qubitDim=(N,N))

