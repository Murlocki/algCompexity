class TreeNode:
    def __init__(self, value):
        self.leftNode = None
        self.rightNode = None
        self.value = value
    def addDescendant(self,value):
        if value <= self.value:
            self.addleftNode(value)
        else:
            self.addRightNode(value)

    def addleftNode(self, value):
        if self.leftNode is None:
            self.leftNode = TreeNode(value)
        else:
            self.leftNode.addDescendant(value)

    def addRightNode(self, value):
        if self.rightNode is None:
            self.rightNode = TreeNode(value)
        else:
            self.rightNode.addDescendant(value)
    def printNodes(self):
        print(self.value)
        if self.leftNode is not None:
            print('left of ',self.value)
            self.leftNode.printNodes()
        if self.rightNode is not None:
            print('right of ',self.value)
            self.rightNode.printNodes()

    def countOfNodes(self):
        count = 1
        if self.leftNode is not None:
            count = count + self.leftNode.countOfNodes()
        if self.rightNode is not None:
            count = count + self.rightNode.countOfNodes()
        return count
a = [1,2,3,4,5,6,7,8,9,10]
root = TreeNode(a[0])
for i in a[1:]:
    root.addDescendant(i)
root.printNodes()
print(root.countOfNodes())

