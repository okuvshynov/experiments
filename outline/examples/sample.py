# Sample Python file for testing

# Constants
PI = 3.14159
GRAVITY = 9.8

# Variable
counter = 0

# Function to add two numbers
def add(a, b):
    return a + b

# Class definition
class Calculator:
    def __init__(self, precision=2):
        self.precision = precision
    
    def add(self, a, b):
        return round(a + b, self.precision)
    
    def subtract(self, a, b):
        return round(a - b, self.precision)
    
    def multiply(self, a, b):
        return round(a * b, self.precision)
    
    def divide(self, a, b):
        if b == 0:
            raise ValueError("Division by zero")
        return round(a / b, self.precision)

# Class with inheritance
class ScientificCalculator(Calculator):
    def __init__(self, precision=4):
        super().__init__(precision)
    
    def square_root(self, x):
        if x < 0:
            raise ValueError("Cannot calculate square root of negative number")
        return round(x ** 0.5, self.precision)
    
    def power(self, base, exponent):
        return round(base ** exponent, self.precision)

# Lambda function
square = lambda x: x * x

# If statement to test the code
if __name__ == "__main__":
    calc = Calculator()
    print(f"Addition: {calc.add(5, 3)}")
    
    sci_calc = ScientificCalculator()
    print(f"Square root of 16: {sci_calc.square_root(16)}")
    print(f"2 raised to power 3: {sci_calc.power(2, 3)}")