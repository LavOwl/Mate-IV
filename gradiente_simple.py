import math

def derivativeByX(x:float) -> float:
    return 4*(x+2)*((x+2)**2 - 3) + 1

def calculateGradiant(x:float):
    return [derivativeByX(x)]

def updatePoint(x:float, alpha:float):
    gradiant = calculateGradiant(x)
    print(gradiant)
    return [x - gradiant[0]*alpha]

def belowTolerance(x:float, tolerance:float) -> bool:
    gradiant = calculateGradiant(x)
    return (gradiant[0] < tolerance)

def main():
    startingPoint = [0]
    alpha = 0.001
    tolerance = 0.0001
    iterations = 0
    while(not belowTolerance(startingPoint[0], tolerance)):
        iterations += 1
        print("Iteration number: " + str(iterations))
        print("Current point at coordinates (" + str(startingPoint[0]) + ")")
        startingPoint = updatePoint(startingPoint[0], alpha)
    print("The calculated minimum is at the coordinates (" + str(startingPoint[0]) + ")")

main()