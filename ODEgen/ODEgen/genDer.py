import sys
import json
from sympy import *
import pprint
import re
import copy
from collections import deque
import random
import compiler

# from domainFuncs import *
from sympy.core.sympify import kernS

from eqGen import EquationTree, buildEq, isCorrect, putNodeInTree, functionOneInp, functionBothSides, intList
from neuralAlgonometry import readBlockJsonEquations, writeJson

import mxnet as mx


def loadData(path, splitRatio=0.8):
	# trainDepth =  [1,2,3,4,5,6,7]
	# testDepth =  [1,2,3,4,5,6,7]
	trainDepth =  [1,2,3,4]
	testDepth =  [1,2,3,4]

	[_,_,_,_,
	 _,_,_,_,
	 equations, variables, ranges, labels] \
	             = readBlockJsonEquations(path, trainDepth=trainDepth, testDepth=testDepth,
	                                      excludeFunc='', splitRatio=splitRatio, devSplit=0.9)
	for e in equations:
		if not e.isUnique(numList=[]):
			e.enumerize_queue()

	eqToChange = copy.deepcopy(equations[0])
	origNode = copy.deepcopy(eqToChange.findNode(106))
	child1 = EquationTree(func='Number', varname='1.1')
	rootNode = EquationTree(func='Add', args=[child1, origNode])
	rootNode = EquationTree(func='Derivative', args=[rootNode,buildEq(Symbol('var_0'),{})])
	changedEq = putNodeInTree(eqToChange, rootNode, 106)
	changedEq.args[0] = EquationTree(func='NumberDecoder', args=[changedEq.args[0]], varname='')
	changedEq.args[1] = EquationTree(func='NumberDecoder', args=[changedEq.args[1]], varname='')
	changedEq.enumerize_queue()

	equations[0] = copy.deepcopy(changedEq)	
	return [equations, variables, labels]

def genSingleDer():
	# Generating derivation axioms
	var_0 = Symbol('var_0')
	var_1 = Symbol('var_1')
	ders = []
	maxDepth = 0
	for func in functionOneInp:

		eq = buildEq(sympify(func+'(var_0)'),{})
		der = diff(sympify(eq),var_0)
		der = buildEq(der,{})
		# eq = EquationTree(func='Derivative', args=[eq,buildEq(Symbol('var_0'), {})]) # if we had ODE we needed these
		eq = EquationTree(func='Derivative', args=[eq,buildEq(Symbol('var_0'),{})])
		ode = EquationTree(func='Equality', args=[eq,der])
		ode.enumerize_queue()
		ders.append(ode)
		currDepth = ode.depth
		if currDepth > maxDepth:
			maxDepth = currDepth

	eq = buildEq(sympify('Equality(Add(1,1,evaluate=False),Derivative(Add(var_0,var_0,evaluate=False),var_0,evaluate=False),evaluate=False)'),{})
	eq.enumerize_queue()
	ders.append(eq)
	currDepth = eq.depth
	if currDepth > maxDepth:
		maxDepth = currDepth
	eq = buildEq(sympify('Equality(Add(Mul(1,var_0,evaluate=False),Mul(var_0*1,evaluate=Flase),evaluate=False),Derivative(Mul(var_0,var_0,evaluate=False),var_0,evaluate=False),evaluate=False)'),{})
	eq.enumerize_queue()
	ders.append(eq)
	currDepth = eq.depth
	if currDepth > maxDepth:
		maxDepth = currDepth

	for func in functionBothSides:
		eq = buildEq(sympify(func+'(var_0,var_0,evaluate=False)'),{})
		der = diff(sympify(eq),var_0)
		der = buildEq(der, {})
		eq = EquationTree(func='Derivative', args=[eq,buildEq(Symbol('var_0'), {})])
		ode = EquationTree(func='Equality', args=[eq,der])
		ode.enumerize_queue()
		ders.append(ode)
		currDepth = ode.depth
		if currDepth > maxDepth:
			maxDepth = currDepth

	# print len(ders)
	# for der in ders:
	# print sympify(der)
		# print dsolve(sympify(der.pretty()))
	# print max([der.depth for der in ders])
	return [ders, maxDepth]

def genPosODE(order, equations, variables, labels):
	coef = [random.choice(intList[1:]) for _ in range(order+1)]
	# print intList
	#raw_input('enter')
	eq = 'Mul('+str(coef[0])+',f(var_0))'
	for ind, c in enumerate(coef[1:]):
		varOrder = ''
		for iind in xrange(ind+1):
			varOrder = varOrder + 'var_0,'
		varOrder = varOrder[:-1]
		# print str(c)
		eq = 'Add('+eq+',Mul('+str(c)+',Derivative(f(var_0),'+varOrder+')))'
		# print 'eqIn:', eq
		# print sympify(eq)
		lhs = buildEq(sympify(eq),{})
		# print 'lhs:',lhs

	chooseRHS = random.choice([0,1])
	if chooseRHS:
		randomEq, randVar, randLab = random.choice(zip(equations, variables, labels))
		# print 'numpy:'
		# print randLab.asnumpy()
		# print 'mxnet:'
		# print randLab
		# if randLab.asnumpy()==[ 0.]:
		# 	print 'found mxnet array' 
		# print type(randLab)
		while randomEq.isNumeric() or len(randVar)==0 or randLab.asnumpy()==[ 0.]:
			randomEq, randVar, randLab = random.choice(zip(equations, variables, labels))
		rhs = random.choice(randomEq.args)
		rhs = rhs.pretty()
		# print rhs
	else:
		oneInp = 1 # random.choice([0,1]) # choosing only one input since sympy cannot solve twoInput
		if oneInp:
			# randomFunc = random.choice(functionOneInp)
			# while 'h' in randomFunc:
			# 	randomFunc = random.choice(functionOneInp)
			randomFunc = random.choice(['sin','cos','tan','cot','exp']) # sympy cannot solve (takes too long to solve) the other functions
			randomCoef = random.choice(intList[1:])
			# print randomFunc, randomCoef
			rhs = buildEq(sympify(randomFunc+'(Mul('+str(randomCoef)+',var_0))'),{})
			rhs = rhs.pretty()
			# print rhs
		else:
			randomFunc = random.choice(['Add','Mul'])
			randomFunc1 = random.choice(['sin','cos','tan','cot','exp'])
			randomFunc2 = random.choice(['sin','cos','tan','cot','exp'])
			rhs = buildEq(sympify(randomFunc+'('+randomFunc1+'(var_0)'+','+randomFunc2+'(var_0))'),{})
			rhs = rhs.pretty()
			# print rhs
	ode = 'Eq('+eq+','+rhs+')'
	return ode

def genNegODE(odeSeq):
	ode = buildEq(sympify(odeSeq),{'x':0, 'C1':1, 'C2':2, 'C3':3})
	ode.enumerize_queue()
	# print 'ode:', ode
	# print 'ode:', ode.pretty()
	try :
		odeStr = str(sympify(ode))
		# print 'Billy ode we are trying to change: ' + odeStr
		# print 'odeStr:', odeStr
		dummy = EquationTree(args=[ode])
		negOdeStr = odeStr
		while negOdeStr == odeStr:
			#print 'while'
			nodeFunc = 'Derivative'
			while nodeFunc == 'Derivative' or nodeFunc=='Symbol' or nodeFunc == 'Tuple': 
				# changing the symbol is meaningless here, since ODEs have only one 
				# variable and the other variables are anyways constants for all the
				# values of which the eq holds so there is no point replacing those
				# we also don't want to change diffs or the unknown functions.
				nodeNum = random.randrange(1, ode.numNodes)
				node = dummy.findNode(nodeNum)
				nodeFunc = node.func

			# print 'Billy node ' + node.func
			# raw_input('enter')
			# print 'node to change for ne gen:', node
			# node.preOrder()
			newNode = node.changeNode(ode.extractVars())
			# print 'node after changing:', newNode
			# newNode.preOrder()
			# traverse and replace the new node:
			negOde = putNodeInTree(ode, newNode, nodeNum)
			if not isinstance(newNode, EquationTree):
				# print newNode
				raise AssertionError("newNode is not of type EquationTree")
			# after insertion of the new nodes numberings have changed
			# negOde.preOrder()
			#print 'sympifying takes time'
			negOdeStr = str(sympify(negOde))
			# print 'oden:', negOde
			# raw_input('enter')
		# print negOde
		# negOde = buildEq(sympify(negOde),{'x':0, 'C1':1, 'C2':2, 'C3':3})
		# negOde.enumerize_queue()
	except :
		return False
	return negOde.pretty()

def genLaplaceODE(order, path, numberOfEqs, outPath):
	# import time
	# start = time.clock()
	nums = [round(i/100.00,2) for i in xrange(-314,315)]
	# stop = time.clock()
	# print stop - start

	equations, variables, labels = loadData(path=path)
	odeList = []
	negOdeList = []
	maxDepth = 0
	posOdeSet = set()
	negOdeSet = set()

	with open(outPath,'w') as outFile:
		outFile.write('here are {0} generated ODEs of order at most {1}\n'.format(numberOfEqs, order))

	# Generating ODEs:
	# ODE skeleton: \sum_i a_i f^{(i)}(t) = \psi(t) where a_i are numbers and \psi(t) is a function in t
	f = Function('f')
	for odeNum in xrange(numberOfEqs):
		print 'generating equation: {0}'.format(odeNum)
		for o in range(1,order+1):

			odeAcceptable = 0
			noOfTries = 0
			while not odeAcceptable and noOfTries <= 15:
				noOfTries += 1
				#print 'generating pos ode'
				ode = genPosODE(order=o,equations=equations,variables=variables,labels=labels)
				# print 'ode:', ode
				# print 'try to solve it'
				try:
					sol = dsolve(sympify(ode))
				except:
					continue
				# print 'sol before:', sol
				# print 'first of sol:', str(sol)[:12]
				if str(sol)[:12] != 'Eq(f(var_0),':
					continue
				sol = str(sol)[13:-1]
				#print 'sol after:', sol
				ode = ode.replace('f(var_0)',sol)
				#print 'cheking ode in pos set'
				if ode in posOdeSet:
					continue
				#print 'building eq tree'
				#print 'Billy ' + str(sympify(ode))
				odeTree = buildEq(sympify(ode),{'x':0, 'C1':1, 'C2':2, 'C3':3}) # we might need to add more Ci's as order grows
				#print 'Billy posOdeTree:', odeTree.pretty()
				#print 'enumerizing eq'
				odeTree.enumerize_queue()
				#print 'checking grammar compatibility'
				odeAcceptable = odeTree.isGrammarCompatible()
				#print 'odeAcceptable:', odeAcceptable
			if noOfTries <= 15:
				posOdeSet.add(ode)
				odeList.append(odeTree)
				currDepth = odeTree.depth
				if currDepth > maxDepth:
					maxDepth = currDepth
				with open(outPath,'a') as outFile:
					outFile.write('EquationP: {0}\n depth: {1}\n'.format(odeTree.pretty(),odeTree.depth))
					outFile.write('positive: {0}\n'.format(odeTree.pretty()))
					try:
						outFile.write('isCorrect: {0}\n'.format(isCorrect(odeTree)))
					except:
						outFile.write('isCorrect not possible, dropping equation and moving to the next equation\n')

			#print 'generating neg ODE'
			negOdeAcceptable = 0
			noOfTries = 0
			while not negOdeAcceptable and noOfTries <= 15:
				noOfTries += 1
				print 'generating neg Ode'
				negOde = genNegODE(ode)
				if(negOde == False or negOde in negOdeSet):
					continue
				
				#if (negOde not in negOdeSet and negOde != False and):
				#	negOdeAcceptable = 1
				# print 'befor:', negOde
				negOde = negOde.replace('f(var_0)',sol)
				negOdeTree = buildEq(sympify(negOde, evaluate=False),{'x':0, 'C1':1, 'C2':2, 'C3':3})
				# print 'negTree:', negOdeTree.pretty()
				negOdeTree.enumerize_queue()
				try:
					negOdeTree.extractVars()
				except:
					#print 'neg ODE with problems:', negOdeTree
					negOdeTree.preOrder()
					#raw_input('enter')
				negOdeAcceptable = negOdeTree.isGrammarCompatible()
				if(not negOdeAcceptable):
					continue
			#print 'done generating negODE'
			if noOfTries <= 15:
				negOdeSet.add(negOde)
			# print 'after:', negOde

				negOdeList.append(negOdeTree)
				currDepth = negOdeTree.depth
				if currDepth > maxDepth:
					maxDepth = currDepth
				with open(outPath,'a') as outFile:
					outFile.write('EquationN: {0}\n depth:{1}\n'.format(negOdeTree.pretty(),negOdeTree.depth))
					outFile.write('negative: {0}\n'.format(negOdeTree.pretty()))
					try:
						outFile.write('isCorrect: {0}\n'.format(isCorrect(negOdeTree)))
					except:
						outFile.write('isCorrect not possible, dropping equation and moving to the next equation\n')

				# if not(odeTree.hasFunc('Dummy') or odeTree.hasFunc('ComplexRootOf') or odeTree.hasFunc('Integral')):
				
				# if not(negOdeTree.hasFunc('Dummy') or negOdeTree.hasFunc('ComplexRootOf') or negOdeTree.hasFunc('Integral')):
				
	return [odeList, negOdeList, maxDepth]

def main():

	[singleDers, maxDepth] = genSingleDer()
	singleLabel = [mx.nd.array([1]) for _ in singleDers]
	singleVars = [sd.extractVars() for sd in singleDers]
	# print singleLabel[0]
	ranges = []
	numEq = 20000
	jsonPath = 'data/singleDers{0}.json'.format(numEq)
	writeJson(jsonPath, singleDers, ranges, singleVars, singleLabel, maxDepth)
	# print [isCorrect(sd) for sd in singleDers]
	path = 'data/data4000_orig_inpuOut_with_neg.json'
	maxOrder = 1
	outPath = 'data/outputODEs_{0}.txt'.format(numEq)
	jsonPath = 'data/outputODEs_{0}.json'.format(numEq)
	[posLaplaceODE, negLaplaceODE, maxDepth] = genLaplaceODE(order=maxOrder, path=path, numberOfEqs=numEq, outPath=outPath)
	posLabel = [mx.nd.array([1]) for _ in posLaplaceODE]
	negLabel = [mx.nd.array([0]) for _ in negLaplaceODE]
	posVars = [ode.extractVars() for ode in posLaplaceODE]
	negVars = [ode.extractVars() for ode in negLaplaceODE]
	ranges = []
	posLaplaceODE.extend(negLaplaceODE)
	posVars.extend(negVars)
	posLabel.extend(negLabel)
	writeJson(jsonPath, posLaplaceODE, ranges, posVars, posLabel, maxDepth)


if __name__=='__main__':
	# import time
	# random.seed(time.clock())
	random.seed(42)
	main()
