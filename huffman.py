import heapq
import os

class HeapNode:
	def __init__(self, char, freq):
		self.char = char
		self.freq = freq
		self.left = None
		self.right = None
		self.index= None

	def __lt__(self, other):    #less than, use min-heap
		if(other == None):
			return -1
		if(not isinstance(other, HeapNode)):
			return -1
		return self.freq < other.freq


class HuffmanCoding:
	def __init__(self):
		self.heap = []
		self.codes = {}
		self.reverse_mapping = {}

	def make_heap(self, frequency):
		for key in frequency: #frequency == frequency.keys()
			node = HeapNode(key, frequency[key]) #node == (key, value) ex) (0,10)
			heapq.heappush(self.heap, node) #add into self. heap list, heappush == append, values are sorted in this code, 
							#ascending order 

	def merge_nodes(self):
		count=0
		while(len(self.heap)>1):
			node1 = heapq.heappop(self.heap) #pop out and delete elements in self.heap 
			node2 = heapq.heappop(self.heap)
			merged = HeapNode(None, node1.freq + node2.freq)  # ascending order 
			merged.left = node1
			merged.right = node2
			merged.index = count # index of each node in binary tree
			count+=1

			heapq.heappush(self.heap, merged) #add new merged element in self.heap, eventually self.heap will contain one biggest value element
		return merged


	def make_codes_helper(self, root, current_code): #recursive
		if(root == None): #if there's no root finish this function
			return

		if(root.char != None): # if this function arrives end of the tree e.g.) (None,5) then end this code
			self.codes[root.char] = current_code #self.codes[600]='1010000111'
			self.reverse_mapping[current_code] = root.char
			return

		self.make_codes_helper(root.left, current_code + "0") #e.g.) first stage, ""+"0"="0"
		self.make_codes_helper(root.right, current_code + "1") #e.g.) first stage, ""+"1"="1"


	def make_codes(self):
		root = heapq.heappop(self.heap) # biggest value element 
		current_code = ""
		self.make_codes_helper(root, current_code)


	def build(self, frequency):
		self.make_heap(frequency)
		merge=self.merge_nodes() #return merged
		self.make_codes()

		return self.codes ,merge
