
import numpy as np
import matplotlib.pyplot as plt

# SIR model function
def sir_model(beta, gamma, population_size, initial_infected, time_steps):
    """
    Simulates the SIR model.

    Args:
        beta: Infection rate.
        gamma: Recovery rate.
        population_size: Total population size.
        initial_infected: Initial number of infected individuals.
        time_steps: Number of time steps to simulate.

    Returns:
        A tuple containing arrays of susceptible, infected, and recovered individuals over time.
    """
    S = np.zeros(time_steps)
    I = np.zeros(time_steps)
    R = np.zeros(time_steps)

    S[0] = population_size - initial_infected
    I[0] = initial_infected
    R[0] = 0

    for t in range(1, time_steps):
        dSdt = -beta * S[t-1] * I[t-1] / population_size
        dIdt = beta * S[t-1] * I[t-1] / population_size - gamma * I[t-1]
        dRdt = gamma * I[t-1]

        S[t] = S[t-1] + dSdt
        I[t] = I[t-1] + dIdt
        R[t] = R[t-1] + dRdt

    return S, I, R


# Parameters for the simulation
population_size = 1000
initial_infected = 10
time_steps = 180

# Low-income population parameters (higher infection rate, lower recovery rate)
beta_low = 0.3  # Higher infection rate due to factors like crowded living conditions
gamma_low = 0.05 # Lower recovery rate due to limited access to healthcare

# High-income population parameters (lower infection rate, higher recovery rate)
beta_high = 0.15 # Lower infection rate due to better hygiene and social distancing
gamma_high = 0.1 # Higher recovery rate due to better access to healthcare


# Run the SIR model for both populations
S_low, I_low, R_low = sir_model(beta_low, gamma_low, population_size, initial_infected, time_steps)
S_high, I_high, R_high = sir_model(beta_high, gamma_high, population_size, initial_infected, time_steps)

# Plot the results
time = np.arange(time_steps)

plt.figure(figsize=(12, 6))
plt.plot(time, I_low, label='Low-Income (Infected)', color='red')
plt.plot(time, I_high, label='High-Income (Infected)', color='blue')
plt.xlabel('Time (Days)')
plt.ylabel('Number of Infected Individuals')
plt.title('SIR Model: Low-Income vs High-Income Populations')
plt.legend()
plt.grid(True)
plt.show()