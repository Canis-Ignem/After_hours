
class node():
    
    def __init__(self, element = None, next_node = None ):

        #Atributes of node
        self.__element = element
        self.__next_node = next_node
    
    #Methods
    def add(self, node ):
        
        if self.get_element() == None:
            self.set_element( node.get_element() ) 
        
        else:   
            aux_node = self
            while aux_node.get_next() != None:
                aux_node = aux_node.get_next()
                aux_node.set_next(node)
            
    def get_element(self):
        return self.__element
    
    def set_element(self, el):
        self.__element = el
        
        
    def get_next(self):
        return self.__next_node
    
    def set_next(self, node):
        self.__next_node = node
        



    
# Creating the nodes
n1 = node(1)
n2 = node(2)
n3 = node(3)
n4 = node(4)

# Adding the nodes to the list
n1.add(n2)
n1.add(n3)
n1.add(n4)


names = ["Alessio", "Deniz", "marvin","Daniel","Thanh"]

head = node()
for i in range(len(names)):
    new_node = node(names[i])
    head.add(new_node)
    
looking_for = "Deniz"
aux_node = head
while aux_node.get_element() != looking_for:
    aux_node = aux_node.get_next()

