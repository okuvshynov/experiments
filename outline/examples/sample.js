// Sample JavaScript file for testing

const PI = 3.14159;
let counter = 0;

function add(a, b) {
  return a + b;
}

class Calculator {
  constructor(precision = 2) {
    this.precision = precision;
  }
  
  add(a, b) {
    return +(a + b).toFixed(this.precision);
  }
  
  subtract(a, b) {
    return +(a - b).toFixed(this.precision);
  }
  
  multiply(a, b) {
    return +(a * b).toFixed(this.precision);
  }
  
  divide(a, b) {
    if (b === 0) {
      throw new Error('Division by zero');
    }
    return +(a / b).toFixed(this.precision);
  }
}

// Arrow function
const square = (x) => x * x;

// Export
export { Calculator, add, square, PI };