from typing import List

class Node:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None



class DecisionTreeClassifier:
    Dataset =[];
    def __init__(self, max_depth: int):
        self.max_depth = max_depth
        pass

    def fit(self, X: List[List[float]], y: List[int]):
        self.n_classes_ = len(set(y));
        self.n_features_ = 4;
        self.tree_ = self._grow_tree(X, y)
        pass
    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class=[0,0,0];
        for i in range(len(y)):
            num_samples_per_class[y[i]]=num_samples_per_class[y[i]]+1;
        predicted_class= num_samples_per_class.index(max(num_samples_per_class));
        node = Node(predicted_class=predicted_class)
        
        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                X_left=[];
                y_left=[];
                X_right=[];
                y_right=[];
                for i in range(len(X)):
                    if(X[i][idx]<thr):
                        X_left.append(X[i]);
                        y_left.append(y[i]);
                    else:
                       X_right.append(X[i]);
                       y_right.append(y[i]);
   
                    node.feature_index = idx
                    node.threshold = thr
                    node.left = self._grow_tree(X_left, y_left, depth + 1)
                    node.right = self._grow_tree(X_right, y_right, depth + 1)
       
        return node
    def _best_split(self, X, y):
        m = len(y);
        if m <= 1:
            return None, None
        num_parent=[0,0,0];
        for i in y:
            num_parent[i]=num_parent[i]+1;
        total=[1,1,1];
        for i in range(len(num_parent)):
            total [i]=(num_parent[i]/m)*(num_parent[i]/m);
        best_gini=1.0-sum(total);
 
        best_idx, best_thr = None, None
        for idx in range(self.n_features_):
            th=[];
            for row in X:
                th.append(row[idx]);
            thresholds, classes = zip(*sorted(zip(th, y)))
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy();
            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(self.n_classes_)
                )
                gini_right = 1.0 - sum((num_right[h] / (m - i)) ** 2 for h in range(self.n_classes_))
                gini = (i * gini_left + (m - i) * gini_right) / m
            
                if thresholds[i] == thresholds[i - 1]:
                    continue
            
                if gini < best_gini:
                    # print("girdi" );
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2;
        
        return best_idx, best_thr
    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class
    def predict(self, X: List[List[float]]):
        return [self._predict(inputs) for inputs in X]
        
   

# if __name__ == '__main__':  
    # X, y = ...
    # X_train, X_test, y_train, y_test = ...

    # clf = DecisionTreeClassifier(max_depth=5)
    # clf.fit(X_train, y_train)
    # yhat = clf.predict(X_test)    
    