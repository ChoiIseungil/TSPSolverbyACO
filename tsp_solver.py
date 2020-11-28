import sys
import csv
import numpy as np
import math
import pandas as pd
import pylab
import time

nodesDict = {}
nodesList = []
antColony = {}
routes = []

#external parameters
#CAUTION: parameter should not contaion .(floating point)
A = 1
B = 9
ANT = 0.005 #ant factor
PHE = 10000 #pheromone factor
THR = 0.005
RHO = 0.0
STEPS = 1

def main():
    #cost matrix
    class Cost():
        def __init__(self,length):
            self.matrix = np.zeros((length+1,length+1),float)

        def update(self,i,j):
            self.matrix[i][j] = distance(i,j)

        def mirror(self):
            self.matrix += self.matrix.transpose()

        #0- indexing이므로 하나씩 더해줘야함
        def get(self,i,j):
            return self.matrix[i,j]

    #pheromone matrix
    class Pheromone():
        #TODO: initialize pheromone array with 1 to all-zero?
        def __init__(self,length):
            self.matrix = np.ones((length+1,length+1),float)

        def leave(self,i,j,amount):
            newAmount = (1-RHO)*self.matrix[i][j]+amount  
            self.matrix[i][j] = newAmount
            self.matrix[j][i] = newAmount

        #0- indexing이므로 하나씩 더해줘야함
        def get(self,i,j):
            return self.matrix[i,j]

    #더 많은 개미를 쓰도록 발전?
    #ant (have to reset every step)
    class Ant():
        def __init__(self,availableList):
            self.nodes = availableList
            self.reset()

        #reset instance
        def reset(self):
            self.traversed = [1] #visited points
            self.position = 1 #무조건 point1에서 시작하는 것으로 가정
            self.route = 0
            self.available = (self.nodes).copy() #hard copy

        #move ant for a one step
        def move(self):
            #not self.available이면 다 돌았다는 것
            while (self.available):
                probList = makeProbList(self.position,self.available)
                probList /= sum(probList)
                # print("Current Position",self.position)
                dest = np.random.choice(self.available, p=probList) #갈 수 있는 모든 node에 대해 계산하고, 확률에 따라 선택
                # print("Destination",dest)
                self.route += cost.get(self.position,dest)
                self.position = dest
                self.traversed.append(dest)
                self.available.remove(dest)

            #last node (?->1)
            dest = 1
            self.route += cost.get(self.position,dest)
            self.position = dest
            self.traversed.append(dest)
            # self.available.remove(dest)
            
            print("I'M BACK")
            print("Route Traveled:",self.route)
            print()

        #leave pheromone on its route
        def leave(self):
            for i, j in zip(self.traversed, self.traversed[1:]):
                pheromone.leave(i,j,Q/self.route)

            #last node (?->1)
            # pheromone.leave(self.traversed[-1],1,Q/self.route)

    # Euclidan distance between i and j point
    def distance(i,j):
        (i_x,i_y) = nodesDict[i]
        (j_x,j_y) = nodesDict[j]
        return math.sqrt(((i_x-j_x)**2)+(i_y-j_y)**2)

    #generate list of probability
    def makeProbList(currentPos,availList):
        # probability of ant k choosing an edge ij (not normalised yet)
        def prob(j):
            return ((pheromone.get(currentPos,j))**A)*((1/(cost.get(currentPos,j)))**B)
        return list(map(prob,availList))

    FILENAME = sys.argv[-1]
    reader = csv.reader(open(FILENAME),delimiter=" ")

    #TODO: erase printing parameter settings
    for row in reader:
        if row[0]=='NODE_COORD_SECTION': break
        print(" ".join(row))

    #initialise dictionary of nodes and list of nodes
    length = 0
    for row in reader:
        if row[0]=='EOF': break
        length+=1
        nodesDict[int(row[0])] = (float(row[1]),float(row[2]))
    nodesList = list(nodesDict.keys())

    #initialise cost matrix
    cost = Cost(length)
    for i in range(length):
        for j in range(i):
            cost.update(i+1,j+1)
    cost.mirror()

    print(np.average(cost.matrix))

    Q = PHE*np.average(cost.matrix)

    pheromone = Pheromone(length)
    numAnts = math.ceil(length*ANT)
    
    #remove point 1 
    nodesList.remove(1)

    for i in range(numAnts):
        antColony["ant{0}".format(i)] = Ant(nodesList)

    # ant = Ant(nodesList)
    print(FILENAME.split('.')[0] + "  A:" + str(A) + "  B:" + str(B) + "  PHE:" + str(PHE) + "  RHO:" + str(RHO).split('.')[1]  + "  ANT:" + str(ANT).split('.')[1] + "  STEPS:" + str(STEPS))

    for i in range(STEPS):
        totalRoute = 0
        print("Step",i+1)
        for ant in antColony.values():
            ant.move()
            totalRoute+=ant.route

        avgRoute = totalRoute/numAnts
        routes.append(avgRoute)

        for ant in antColony.values():
            ant.leave()
            ant.reset()

    print("Final Average Route: ",avgRoute)

    #last step for get a solution.csv
    lastAnt = Ant(nodesList)
    lastAnt.move()
    solution = lastAnt.traversed #save solution for last step

    phr = pd.DataFrame(pheromone.matrix)
    phr.to_csv('pheromone.csv',index=True)

    cos = pd.DataFrame(cost.matrix)
    cos.to_csv('cost.csv',index=True)

    result = pd.DataFrame(solution)
    result.to_csv('./solution/' + FILENAME.split('.')[0] + "  A" + str(A) + "  B" + str(B) + "  PHE" + str(PHE) + "  RHO" + str(RHO).split('.')[1]  + "  ANT" + str(ANT).split('.')[1] + "  STEPS" + str(STEPS) + ".csv")

    steps = [i for i in range(STEPS)]
    pylab.plot(steps, routes, 'b')
    pylab.savefig("./graphs/" + FILENAME.split('.')[0] + "  A" + str(A) + "  B" + str(B) + "  PHE" + str(PHE) + "  RHO" + str(RHO).split('.')[1]  + "  ANT" + str(ANT).split('.')[1] + "  STEPS" + str(STEPS))

if __name__=="__main__":
    tic = time.time()
    main()
    toc = time.time()
    print("Elapsed Time",toc-tic)