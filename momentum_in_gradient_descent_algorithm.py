#The classical gradient descent algorithm may get stuck at a local
#minimum for the loss function. To avoid that, we can take into account
#the "speed" at which the gradient has dropped.

def get_current_parameters(weight,cost_func_val,grad):
    return weight,cost_func_val,grad
#initialize parameters
w,loss,gradient=0,0,0
past_velocity = 0.

#the fraction of the previous speed which is kept in memory (10%)
momentum = 0.1

#while we have not reached a satisfying minimum
while loss > 0.01:
    w, loss, gradient = get_current_parameters(weight,cost_func_val,grad)
              #first * : acquired speed | second * : 'gravity' component
    velocity = past_velocity * momentum + learning_rate * gradient
    w = w + momentum * velocity - learning_rate * gradient
    past_velocity = velocity
#update_parameter(w)
