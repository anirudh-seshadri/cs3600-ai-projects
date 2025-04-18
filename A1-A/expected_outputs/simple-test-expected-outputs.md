# Simple Test Case Solutions

The following are the expected outputs for the simple test cases. These are the environments that are already loaded in the assignment notebook. You can use these to compare your outputs to the expected outputs. The testing suite in the notebook will compare your outputs to the expected outputs for each of the 20 provided environments. 

As a reminder, we will be testing your code on further environments as well, so make sure your code does not hardcode any solutions (e.g. paths to goals). These hidden environments will not test any edge cases (i.e., there will always be a solution).

## Coin (numLocations=5,includeDoors=1,numDistractorItems=0)

### BFS
The plan: ['open door to south', 'move south', 'open door to east', 'move east', 'take coin']
Number of visited states: 30

### DFS
The plan: ['open door to west', 'open door to south', 'close door to west', 'move south', 'open door to east', 'move north', 'open door to west', 'move south', 'close door to north', 'move east', 'take coin']
Number of visited states: 12

### A-Star
The plan: ['open door to south', 'move south', 'open door to east', 'move east', 'take coin']
Number of visited states: 16

## Map Reader (numLocations=5,maxDistanceApart=3,includeDoors=0,maxDistractorItemsPerLocation=0)

### BFS
The plan: ['move west', 'move north', 'take coin', 'move south', 'move east', 'put coin in box']
Number of visited states: 19


### DFS
The plan: ['put map in box', 'move north', 'move west', 'take coin', 'move south', 'move east', 'put coin in box']
Number of visited states: 11


### A-Star
The plan: ['move north', 'move west', 'take coin', 'move east', 'move south', 'put coin in box']
Number of visited states: 16
