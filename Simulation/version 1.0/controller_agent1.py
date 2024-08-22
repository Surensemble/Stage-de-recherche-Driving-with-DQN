########################################################################
#============================== libraries =============================#
########################################################################
from controller import Supervisor , TouchSensor
from vehicle import Driver , Car
from controller import Camera
import numpy as np
import random as rd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import os
import cv2 as cv
import matplotlib.pyplot as plt

episode=[]
rewardCurve=[] 
driver=Driver()

########################################################################
#============== interaction with the environment ======================#
########################################################################
class AutonomousCar(Supervisor):
    def __init__(self):
        super(AutonomousCar, self).__init__()
        self.driver = driver
        self.driver.setSteeringAngle(0)
        self.timestep = int(self.driver.getBasicTimeStep())
        self.max_steps = 10000
        
        # Camera initialization
        self.camera = self.driver.getDevice('camera')
        self.camera.enable(self.timestep)
        
        # TouchSensor
        self.touch_sensor = self.driver.getDevice('touch sensor')
        self.touch_sensor.enable(self.timestep)
        
        # Save initial position and orientation
        self.translation_field = self.getFromDef("CAR").getField("translation")
        self.rotation_field = self.getFromDef("CAR").getField("rotation")
        self.initial_position = self.translation_field.getSFVec3f()
        self.initial_rotation = self.rotation_field.getSFRotation()
        
        
        # Initialize state
        self.steps = 0
        self.episode_reward = 0
        self.done = False

    def reset(self):
        # Reset the simulation
        
        # Reset the position and orientation
        self.translation_field.setSFVec3f(self.initial_position)
        self.rotation_field.setSFRotation(self.initial_rotation)
        
        self.simulationResetPhysics()
        super(AutonomousCar, self).step(int(self.getBasicTimeStep()))
        self.steps = 0
        self.episode_reward = 0
        self.done = False
        
        # Get initial state
        image = self.camera.getImageArray()
        state = self.process_image(image)
        return state 

    def process_image(self, image):
        # Convert the image from Webots to a NumPy array
        image = np.array(image, dtype=np.uint8)
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        # Preprocess the image to feed into the network
        image = cv.resize(image, (64, 64))
        image = np.transpose(image, (2, 0, 1))  # Change shape to (3, 64, 64)
        return image

    def step(self, action):
        # Apply action
        self.steps += 1
        self.driver.setCruisingSpeed(15)
        if action == 0:
           self.driver.setSteeringAngle(0.5)  
        elif action == 1:
            self.driver.setSteeringAngle(-0.5)  
        elif action == 2:
            self.driver.setSteeringAngle(0)  
                  
        super(AutonomousCar, self).step(self.timestep)
        
        # Get next state
        image = self.camera.getImageArray()
        next_state = self.process_image(image)
        
        # Calculate reward
        # à modifier pour mieux evaluer les actions
        reward = 0
        if self.touch_sensor.getValue() > 0:
            self.done = True
            reward = -100  # Negative reward for collision
            
        elif self.steps >= self.max_steps:
            self.done = True
            reward = 1
        else:
            reward = 0.5  

        return next_state, reward, self.done, {}

    def render(self):
        pass

    def close(self):
        pass

########################################################################
#============================== replay memory =========================#
########################################################################
class ReplayMemory:
      def __init__(self,capacity):
          self.capacity = capacity
          self.states = []
          self.actions = []
          self.next_states = []
          self.rewards = []
          self.dones = []
          
          self.idx = 0
          
          
      def store(self, states, actions, next_states, rewards, dones):
          if len(self.states) < self.capacity:
             self.states.append(states)
             self.actions.append(actions)
             self.next_states.append(next_states)
             self.rewards.append(rewards)
             self.dones.append(dones)
          else: 
             self.states[self.idx] = states
             self.actions[self.idx] = actions
             self.next_states[self.idx] = next_states
             self.rewards[self.idx] = rewards
             self.dones[self.idx] = dones
                  
          self.idx = (self.idx + 1)%self.capacity
       
       
      def sample(self, batchsize, device):
          indices_to_sample = rd.sample(range(len(self.states)), k=batchsize) 
          
          states = torch.from_numpy(np.array(self.states)[indices_to_sample]).float().to(device)    
          actions = torch.from_numpy(np.array(self.actions)[indices_to_sample]).to(device) 
          next_states = torch.from_numpy(np.array(self.next_states)[indices_to_sample]).float().to(device)   
          rewards = torch.from_numpy(np.array(self.rewards)[indices_to_sample]).float().to(device)         
          dones = torch.from_numpy(np.array(self.dones)[indices_to_sample]).float().to(device)

          return states, actions, next_states, rewards, dones


      def __len__(self):
          return len(self.states) 
        
########################################################################
#============================== deep Q network ========================#
########################################################################       

class DQNnet(nn.Module):

    def __init__(self, state_shape, action_shape, lr=1e-3):

        super(DQNnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        self.fc1 = nn.Linear(self.feature_size(state_shape), 512)
        self.fc2 = nn.Linear(512, action_shape)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def feature_size(self, state_shape):
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, *state_shape)))).view(1, -1).size(1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x    

########################################################################
#============================== DQN agent =============================#
########################################################################                          
                        
class DQNAgent:

    def __init__(self, observation_space, action_space, device, epsilon_max, epsilon_min, epsilon_decay, memory_capacity, discount=0.99, lr=1e-3):

        self.observation_space = observation_space
        self.action_space = action_space
        self.discount = discount
        self.device = device

        self.epsilon = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.loss = 5000.
        self.replay_memory = ReplayMemory(memory_capacity)

        self.online_network = DQNnet(self.observation_space.shape, self.action_space.n, lr).to(self.device)

        self.target_network = DQNnet(self.observation_space.shape, self.action_space.n, lr).to(self.device)

        self.target_network.eval()
        self.update_target()


    def update_target(self):
        self.target_network.load_state_dict(self.online_network.state_dict())  

    def select_action(self, state):
        if rd.random() < self.epsilon:
            action = self.action_space.sample()
            #print(action)
            return action
            #return self.action_space.sample()

        if not torch.is_tensor(state):
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        with torch.no_grad():
            action = torch.argmax(self.online_network(state))
           # print(action)
        return action.item() 
    
    def select_action_test(self, state):
        if not torch.is_tensor(state):
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        with torch.no_grad():
            action = torch.argmax(self.online_network(state))
           # print(action)
        return action.item() 

    def update_epsilon(self) :
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)  

    def learn(self, batchsize):
        if len(self.replay_memory) < batchsize:
            return

        states, actions, next_states, rewards, dones = self.replay_memory.sample(batchsize, self.device)  
        actions = actions.reshape((-1,1))       
        rewards = rewards.reshape((-1,1))
        dones = dones.reshape((-1,1))

        predicted_qs = self.online_network(states)
        predicted_qs = predicted_qs.gather(1, actions)

        target_qs = self.target_network(next_states)
        target_qs = torch.max(target_qs, dim=1).values
        target_qs = target_qs.reshape(-1,1)
        dones = dones.bool()
        target_qs[dones] = 0.0
        y_js = rewards + (self.discount*target_qs)

        loss = F.mse_loss(predicted_qs, y_js)
        self.loss=loss
        self.online_network.optimizer.zero_grad()
        loss.backward()
        self.online_network.optimizer.step()


    def save(self, filename):
        torch.save(self.online_network.state_dict(), filename) 

    def load(self, filename):
        self.online_network.load_state_dict(torch.load(filename))
        self.online_network.eval()

########################################################################
#============================== Environnement =========================#
######################################################################## 
class Environment(gym.Env):

    def __init__(self):
        super(Environment, self).__init__()
        # initialize the env class
        self.max_steps = 10000
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8)
        self.car = AutonomousCar()

    def reset(self):
        return self.car.reset()
 
    def step(self, action): 
        return self.car.step(action)

    def render(self):
        self.car.render()

    def close(self):
        self.car.close()

########################################################################
#============================== Main ==================================#
######################################################################## 

def fill_memory(env, agent, memory_fill_eps):
    for _ in range(memory_fill_eps):
        state = env.reset()                     
        done = False
        while not done:  
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.replay_memory.store(state, action, next_state, reward, done)
            state = next_state

def train(env, agent, train_eps, memory_fill_eps, batchesize, update_freq, model_filename):
    
    
    fill_memory(env, agent, memory_fill_eps)
    print('Samples in memory: ', len(agent.replay_memory))

    step_cnt = 0
    best_score = -np.inf
    reward_history = []
    #loss_history= []
    for ep_cnt in range(train_eps):
        state = env.reset()
        done = False
        ep_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.replay_memory.store(state, action, next_state, reward, done)
            agent.learn(batchesize)

            if step_cnt % update_freq == 0:
                agent.update_target()

            state = next_state
            ep_reward += reward
            step_cnt += 1
        #stock
        agent.update_epsilon()
        reward_history.append(ep_reward)
        #loss_history.append(agent.loss)
        current_avg_score = np.mean(reward_history[-100:])
        episode.append(ep_cnt)
        rewardCurve.append(current_avg_score)

        print('EP: {}, Total Steps: {}, EP Score: {}, Avg score: {}; Updated Epsilon: {}'.format(ep_cnt, step_cnt, ep_reward, current_avg_score, agent.epsilon))  
  
        if current_avg_score >= best_score:
            agent.save(model_filename)
            best_score = current_avg_score

    """loss_np = np.zeros((2,1))
    for i in range(2):
        loss_np[i] = loss_history[i].detach().cpu().numpy()        
    print(loss_np) """
     
       
def test(env, agent, test_eps):
    for ep_cnt in range(test_eps):
        state = env.reset()
        done = False
        ep_reward = 0

        while not done:
            action = agent.select_action_test(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
            ep_reward += reward

        print('Ep: {} '.format(ep_cnt))    

"""def set_seed(env, seed_value):
    os.environ['PYTHONHASHSEED']=str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    env.seed(seed_value)
    env.action_space.np_random.seed(seed_value)"""

if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_mode = False  # True for training mode and False for test model
    
    env = Environment()
    model_filename = 'model1'

    if train_mode:
         #set_seed(env,0)
         
        
         agent = DQNAgent(observation_space=env.observation_space, 
                          action_space=env.action_space,
                          device=device,
                          epsilon_max=1.0,
                          epsilon_min=0.01,
                          epsilon_decay=0.995,
                          memory_capacity=10000,
                          discount=0.99,
                          lr=1e-3)
        
         train(env=env,
                 agent=agent,
                 train_eps=250,    #2000
                 memory_fill_eps=50,
                 batchesize=64,
                 update_freq=1000,
                 model_filename=model_filename)
         
         #display of the reward curve
         while driver.step()!=-1:          
             print(episode, rewardCurve)              
             plt.plot(episode,rewardCurve)
             plt.xlabel("episodes")
             plt.ylabel("Average Reward")
             plt.grid(1) 
             plt.show()
             print("Training step completed.")    

    else:
        #set_seed(env, 10)

        agent = DQNAgent(observation_space=env.observation_space,
                           action_space=env.action_space,
                           device=device,
                           epsilon_max=0.0,
                           epsilon_min=0.0,
                           epsilon_decay=0.0,
                           memory_capacity=10000,
                           discount=0.99,
                           lr=1e-3)
        agent.load(model_filename)

        test(env=env, agent=agent, test_eps=100)
