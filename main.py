import numpy as np
import matplotlib.pyplot as plt
import math
from DQN import Agent

class Enviroment():
    def __init__(self):
        self.x_list = list()
        self.y_list = list()
        self.alpha_list = list()
        self.ctrl_list = list()
        self.time_list = list()
        self.x0 = 0
        self.xk = 0
        self.y0 = 0
        self.yk = 1000
        self.alpha0 = 0
        self.agent = Agent()
        self.gain = np.pi * 0.1
        self.water_speed = 1
        self.boat_speed = 5
        self.delta_time = 0.2
    def modeling(self):
        x = self.x0
        y = self.y0
        alpha = self.alpha0
        self.x_list.clear()
        self.y_list.clear()
        self.alpha_list.clear()
        self.ctrl_list.clear()
        self.time_list.clear()
        T = 0
        Vw = self.water_speed
        V = self.boat_speed
        h = self.delta_time
        while True:
            if y < 0:
                break
            if y > 1000:
                break
            if abs(x) > 1000:
                break
            self.x_list.append(x)
            self.y_list.append(y)
            self.alpha_list.append(alpha)
            self.time_list.append(T)
            state = [x,y,alpha]

            action = self.agent.select_action(state)

            dx = V*np.sin(alpha)-Vw
            dy = V*np.cos(alpha)
            dalpha = action*self.gain

            x += dx*h
            y += dy*h
            alpha += dalpha*h
            T += h
        print("X: ", x)
        print("Y: ", y)


    def replay(self):
        x = self.x0
        y = self.y0
        alpha = self.alpha0
        T = 0
        Vw = self.water_speed
        V = self.boat_speed
        h = self.delta_time
        fata = False
        while True:

            if y < 0:
                break
            if y > 1000:
                break

            if abs(x) > 1000:
                break

            state = [x,y,alpha]
            action = self.agent.select_action(state)
            dx = V*np.sin(alpha)-Vw
            dy = V*np.cos(alpha)
            dalpha = action*self.gain

            x += dx*h
            y += dy*h
            alpha += dalpha*h

            reward = 100*dy-abs(x)+y

            next_state = [x, y, alpha]


            self.agent.add_to_replay([state, action, reward, next_state])

            if y>10:
                self.agent.train()

    def getX(self):
        return self.x_list

    def getY(self):
        return self.y_list
    def get_time(self):
        return self.time_list
    def get_alpha(self):
        return self.alpha_list

env = Enviroment()

env.modeling()
for i in range(100):
    print("____________")
    env.replay()
    env.modeling()


plt.plot([-500, 500],[1000, 1000], color = "black")
plt.plot([-500, 500],[0, 0], color = "black")
plt.plot([0, 0],[0, 1000], color = "red",linestyle = 'dashed')
plt.plot(env.getX(), env.getY())
plt.show()


plt.plot(env.get_time(), env.get_alpha())
plt.show()