class TreeNode:
    def __init__(self, value,level=1):
        self.leftNode = None
        self.rightNode = None
        self.value = value
        self.level= level
    def addDescendant(self,value):
        if value <= self.value:
            self.addleftNode(value)
        else:
            self.addRightNode(value)

    def addleftNode(self, value):
        if self.leftNode is None:
            self.leftNode = TreeNode(value,level=self.level+1)
        else:
            self.leftNode.addDescendant(value)

    def addRightNode(self, value):
        if self.rightNode is None:
            self.rightNode = TreeNode(value,level=self.level+1)
        else:
            self.rightNode.addDescendant(value)
    def printNodes(self):
        print(self.value)
        if self.leftNode is not None:
            print('left of ',self.value,self.leftNode.level)
            self.leftNode.printNodes()
        if self.rightNode is not None:
            print('right of ',self.value,self.rightNode.level)
            self.rightNode.printNodes()

    def countOfNodes(self):
        count = 1
        if self.leftNode is not None:
            count = count + self.leftNode.countOfNodes()
        if self.rightNode is not None:
            count = count + self.rightNode.countOfNodes()
        return count
    def countOfNodesOnLevel(self,currentLevel,K):
        if currentLevel==K:
            return 1
        else:
            count = 0
            if self.leftNode is not None:
                count = count + self.leftNode.countOfNodesOnLevel(currentLevel+1,K)
            if self.rightNode is not None:
                count = count + self.rightNode.countOfNodesOnLevel(currentLevel+1,K)
            return count
    def countOnLevelCall(self,K):
        return self.countOfNodesOnLevel(1,K)

    def countOnLevelNonRec(self,K):
        count=0
        nodes=[self]
        while nodes:
            if nodes[0].level==K:
                count = count + 1
            else:
                if nodes[0].leftNode is not None:
                    nodes.append(nodes[0].leftNode)
                if nodes[0].rightNode is not None:
                    nodes.append(nodes[0].rightNode)
            nodes.pop(0)
        return count

a = [4,5,6,1,2,3,4,5,6,7,8,9,10]
#a=[2,3,1]
root = TreeNode(a[0])
for i in a[1:]:
    root.addDescendant(i)
#root.printNodes()
#print(root.countOfNodes())
print("Количество узло рекурсивно:",root.countOnLevelCall(4))
print("Количество узлов нерекурсивно:",root.countOnLevelNonRec(4))