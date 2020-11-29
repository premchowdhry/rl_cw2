import time
import numpy as np

from random_environment import Environment
from agent_wandb import Agent

import wandb

# Main entry point
if __name__ == "__main__":
    hyperparameter_defaults = dict(
        epsilon=0.5,
        decay=0.95,
        gamma=0.9,
        lr=0.001,
        batch_size=32,
        update_target_freq=50,
        episode_length=5000,
        alpha=0.6
    )
    wandb.init(config=hyperparameter_defaults, project='Coursework', entity='premchowdhry')
    config = wandb.config

    # This determines whether the environment will be displayed on each each step.
    # When we train your code for the 10 minute period, we will not display the environment.
    display_on = False

    # Create a random seed, which will define the environment
    # random_seed = int(time.time())
    random_seed = 12345
    np.random.seed(random_seed)

    # Create a random environment
    environment = Environment(magnification=500)

    # Create an agent
    agent = Agent(
        config.epsilon,
        config.decay,
        config.gamma,
        config.lr,
        config.batch_size,
        config.update_target_freq,
        config.episode_length,
        config.alpha
    )

    # Get the initial state
    state = environment.init_state

    # Train the agent, until the time is up
    for _ in range(25 * config.episode_length):
        # If the action is to start a new episode, then reset the state
        if agent.has_finished_episode():
            state = environment.init_state
        # Get the state and action from the agent
        action = agent.get_next_action(state)
        # Get the next state and the distance to the goal
        next_state, distance_to_goal = environment.step(state, action)
        # Return this to the agent
        agent.set_next_state_and_distance(next_state, distance_to_goal)
        # Set what the new state is
        state = next_state
        # Optionally, show the environment
        if display_on:
            environment.show(state)

    # Test the agent for 100 steps, using its greedy policy
    state = environment.init_state
    has_reached_goal = False
    for step_num in range(100):
        action = agent.get_greedy_action(state)
        next_state, distance_to_goal = environment.step(state, action)

        metrics = {'distance_to_goal': distance_to_goal}
        wandb.log(metrics)
        # The agent must achieve a maximum distance of 0.03 for use to consider it "reaching the goal"
        if distance_to_goal < 0.03:
            has_reached_goal = True
            break
        state = next_state

    # Print out the result
    if has_reached_goal:
        print('Reached goal in ' + str(step_num) + ' steps.')
    else:
        print('Did not reach goal. Final distance = ' + str(distance_to_goal))
