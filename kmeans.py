import csv
import math
import random
from typing import List

class KMeansClusterClassifier:
    centerL = []
    labelL = []
    data_list=[];
    def __init__(self, n_cluster: int):
        self.k_cluster = n_cluster
        pass

    def fit(self, X: List[List[float]], y: List[int]):
        for index in range(0, len(X)):
            self.data_list.append([float(X[index][0]), float(X[index][1]), float(X[index][3])])
        
        for i in range(self.k_cluster):
            self.centerL.append(self.data_list[i])
            
        for dot in self.data_list:
            EL = []
            for i in self.centerL:
                r = math.sqrt(math.pow(dot[0] - i[0], 2) + math.pow(dot[1] - i[1], 2) + math.pow(dot[2] - i[2], 2))
                EL.append(r)
            minValue = min(EL)
            minIndex = EL.index(minValue)
            self.labelL.append(minIndex)
        while 1:
            ex_centerL = self.centerL
            self.centerL= self.determine_new_center()
            self.labelL=self.determine_new_labels()
            
            if self.centerL == ex_centerL:
                break
     
        

    def determine_new_center(self):
        total_labels = 0
        total_data_list = 0
        totalAll = 0
        newcenterL = []
        c = 0
        for element in range(self.k_cluster):
                for index in range(len(self.labelL)):
                    if self.labelL[index] == element:
                        total_data_list = total_data_list + self.data_list[index][0]
                        total_labels = total_labels + self.data_list[index][1]
                        totalAll = totalAll + self.data_list[index][2]
                        c = c + 1
                newcenterL.append([total_data_list/c, total_labels/c, totalAll/c])
                total_data_list = 0
                total_labels = 0
                totalAll = 0
                c = 0
        return newcenterL;
    
    def predict(self, X: List[List[float]]):
        data_list2=[]
        for index in range(0, len(X)):
           data_list2.append([X[index][0], X[index][1], X[index][3]])
        self.data_list=data_list2
        return self.determine_new_labels()
       
    def determine_new_labels(self):
        newlabelL = []
        for dot in self.data_list:
            EL = []
            for c in self.centerL:
                    r = math.sqrt(math.pow(dot[0] - c[0], 2) + math.pow(dot[1] - c[1], 2) + math.pow(dot[2] - c[2], 2))
                    EL.append(r)
            minValue = min(EL)
            minIndex = EL.index(minValue)
            newlabelL.append(minIndex)
        return newlabelL;
    
    
        
   

# if __name__ == '__main__':  
    # X, y = ...
    # X_train, X_test, y_train, y_test = ...

    # clf = DecisionTreeClassifier(max_depth=5)
    # clf.fit(X_train, y_train)
    # yhat = clf.predict(X_test)    
    






















