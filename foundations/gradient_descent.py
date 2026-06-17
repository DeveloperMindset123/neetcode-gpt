class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        '''
        the same mathematical formula, is to be applied every iteration until iteration reaches 0
        '''
        # objective_function = pow(init,2)
        # derivative = 2 * init

        # def (x_old, x_new, objective_function, derivative)

        # we can also implement using a for loop as well.
        while iterations > 0:
            init = init - learning_rate * (2 * init)        # this is the implementation of the formula
            iterations -= 1      # decrement iteration by 1

        return round(init, 5)       # because, we have been asked to round to 5 decimal places



