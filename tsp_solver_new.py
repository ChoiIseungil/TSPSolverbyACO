import csv
import argparse
import numpy as np
import math
import pandas as pd
import pylab
import time

nodesDict = {}
nodesList = []
antColony = {}
routes = []

def main():
    #cost matrix
    class Cost():
        def __init__(self,length):
            self.matrix = np.zeros((length+1,length+1),float)
            self.recip = np.zeros((length+1,length+1),float)

        def update(self,i,j):
            self.matrix[i][j] = distance(i,j)

        def inverse(self):
            self.recip = np.reciprocal(self.matrix)

        def mirror(self):
            self.matrix += self.matrix.transpose()

        #0- indexing이므로 하나씩 더해줘야함
        def get(self,i,j):
            return self.matrix[i,j]

        def getRecip(self,i,j):
            return self.recip[i,j]

    #pheromone matrix
    class Pheromone():
        #TODO: initialize pheromone array with 1 to all-zero?
        def __init__(self,length):
            self.matrix = np.random.rand(length+1,length+1)

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
            return ((pheromone.get(currentPos,j))**A)*((cost.getRecip(currentPos,j))**B)
        return list(map(prob,availList))

    #input flags
    parser = argparse.ArgumentParser()
    # parser.add_argument('-a', nargs = '?', default = 1, help = "parameter A",)
    # parser.add_argument('-b', nargs = '?', default = 9,  help = "parameter B")
    parser.add_argument('-phe', nargs = '?', default = 5000, help = "total amount of pheromone")
    parser.add_argument('-rho', nargs = '?', default = 0.0, type = float,help = "evaporation rate")
    parser.add_argument('-ant', nargs = '?', default = 0.005, type = float ,help = "number of ants")
    parser.add_argument('-f', nargs = '?', default = 1000000, type = int, help = "totla number of fitness evaluation")
    parser.add_argument('filename', help = "name of tsp file")

    args = parser.parse_args()

    A = 1
    B = 9
    PHE = args.phe
    RHO = args.rho
    ANT = args.ant
    ITER = args.f
    FILENAME = args.filename
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

    Q = PHE*np.average(cost.matrix)  #평균의 반

    cost.mirror()
    cost.inverse()

    pheromone = Pheromone(length)
    numAnts = math.ceil(length*ANT)
    STEPS = math.floor(ITER/numAnts)

    #remove point 1 
    nodesList.remove(1)

    for i in range(numAnts):
        antColony["ant{0}".format(i)] = Ant(nodesList)

    # ant = Ant(nodesList)
    print(FILENAME.split('.')[0] + "  A:" + str(A) + "  B:" + str(B) + "  PHE:" + str(PHE) + "  RHO:" + str(RHO).split('.')[1]  + "  ANT:" + str(ANT).split('.')[1] + "  Fitness:" + str(ITER))

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

    # phr = pd.DataFrame(pheromone.matrix)
    # phr.to_csv('pheromone.csv',index=True)

    cos = pd.DataFrame(cost.matrix)
    cos.to_csv('cost.csv',index=True)

    rec = pd.DataFrame(cost.recip)
    rec.to_csv('recip.csv',index=True)

    result = pd.DataFrame(solution)
    result.to_csv('./solution/' + FILENAME.split('.')[0] + "  A" + str(A) + "  B" + str(B) + "  PHE" + str(PHE) + "  RHO" + str(RHO).split('.')[1]  + "  ANT" + str(ANT).split('.')[1] + "  Fitness" + str(ITER) + ".csv")

    steps = [i for i in range(STEPS)]
    pylab.plot(steps, routes, 'b')
    pylab.savefig("./graphs/" + FILENAME.split('.')[0] + "  A" + str(A) + "  B" + str(B) + "  PHE" + str(PHE) + "  RHO" + str(RHO).split('.')[1]  + "  ANT" + str(ANT).split('.')[1] + "  Fitness" + str(ITER))

if __name__=="__main__":
    tic = time.time()
    np.seterr(divide='ignore')
    main()
    toc = time.time()
    print("Elapsed Time",toc-tic)