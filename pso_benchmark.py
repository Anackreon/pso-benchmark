import random
import math
import numpy as np
import pandas as pd


"""
Configuraion parameters for PSO Intelligent Parameter Tuning
"""
SWARM_SIZE = 20
DIMENTIONS = 20
ITERATIONS = 1000
PSO_RUNS = 30
ENABLE_EXEC_LOG = True



class PSO:
    """
    Particle Swarm Optimization implementation with star topology
    """
    def __init__(self, w, c1, c2, bounds, obj_func):
        """
        POS random initialization of particle positions and velocities
        """
        self.inertia = w
        self.cognitive = c1
        self.social = c2
        self.obj_func = obj_func
        # establish the swarm
        swarm=[]
        for i in range(SWARM_SIZE):
            start_position = [random.uniform(bounds[0], bounds[1]) for i in range(DIMENTIONS)]
            swarm.append(Particle(start_position))
        self.swarm = swarm
        self.bounds = bounds

    def run(self):
        err_best_g = -1                   # best error for group
        pos_best_g = []                   # best position for group
        swarm = self.swarm

        # begin optimization loop
        for i in range(ITERATIONS):
            # cycle through particles in swarm and evaluate fitness
            for j in range(SWARM_SIZE):
                swarm[j].evaluate(self.obj_func)

                # determine if current particle is the best (globally)
                if swarm[j].particle_err < err_best_g or err_best_g == -1:
                    pos_best_g=list(swarm[j].particle_pos)
                    err_best_g=float(swarm[j].particle_err)

            # cycle through swarm and update velocities and position
            for j in range(SWARM_SIZE):
                swarm[j].update_velocity(pos_best_g, self.inertia, self.cognitive, self.social)
                swarm[j].update_position(self.bounds)

        return err_best_g



class Particle:
    """
    Represents a particle of the swarm
    """
    def __init__(self, start):
        self.particle_pos = []          # particle position
        self.particle_vel = []          # particle velocity
        self.best_pos = []              # best position individual
        self.best_err = -1              # best result individual
        self.particle_err = -1          # result individual
        
        for i in range(DIMENTIONS):
            self.particle_vel.append(random.uniform(-1,1))
            self.particle_pos.append(start[i])

    def evaluate(self, costFunc):
        """
        Evaluate current fitness
        """
        self.particle_err = costFunc(self.particle_pos)

        # check to see if the current position is an individual best
        if self.particle_err < self.best_err or self.best_err == -1:
            self.best_pos = self.particle_pos
            self.best_err = self.particle_err

    def update_velocity(self, pos_best_g, w, c1, c2):
        """
        Updates new particle velocity
        """
        for i in range(DIMENTIONS):
            r1 = random.random()
            r2 = random.random()

            vel_cognitive = c1 * r1 * (self.best_pos[i] - self.particle_pos[i])
            vel_social = c2 * r2 * (pos_best_g[i] - self.particle_pos[i])
            self.particle_vel[i] = w * self.particle_vel[i] + vel_cognitive + vel_social

    def update_position(self, bounds):
        """
        Updates the particle position based off new velocity updates
        """
        for i in range(DIMENTIONS):
            self.particle_pos[i] = self.particle_pos[i] + self.particle_vel[i]

            # adjust maximum position if necessary
            if self.particle_pos[i] > bounds[1]:
                self.particle_pos[i] = bounds[1]

            # adjust minimum position if neseccary
            if self.particle_pos[i] < bounds[0]:
                self.particle_pos[i] = bounds[0]



class ObjFn:
    """
    Objective functions for optimization with PSO
    """
    spherical_bounds = (-5.12, 5.12)
    ackley_bounds = (-32.768, 32.768)
    michalewicz_bounds = (np.finfo(np.float64).tiny, math.pi)
    katsuura_bounds = (-100, 100)

    def spherical(x):
        result = 0
        for i in range(len(x)):
            result = result + x[i]**2
        return result

    def ackley(x):
        n = len(x)
        res_cos = 0
        for i in range(n):
            res_cos = res_cos + math.cos(2*(math.pi)*x[i])
        #returns the point value of the given coordinate
        result = -20 * math.exp(-0.2*math.sqrt((1/n)*ObjFn.spherical(x)))
        result = result - math.exp((1/n)*res_cos) + 20 + math.exp(1)
        return result

    def michalewicz(x):
        m=10
        result = 0
        for i in range(len(x)):
            result -= math.sin(x[i]) * (math.sin((i*x[i]**2)/(math.pi)))**(2*m)
        return result

    def katsuura(x):
        prod = 1
        d = len(x)
        for i in range(0, d):
            sum = 0
            two_k = 1
            for k in range(1, 33):
                two_k = two_k * 2
                sum += abs(two_k * x[i] - round(two_k * x[i])) / two_k
            prod *= (1 + (i+1) * sum)**(10/(d**1.2))
        return (10/(d**2)) * (prod - 1)



class BenchmarkPSO:
    """
    Implements the logical steps of the PSO Intelligent Parameter Tuning
    """
    def __init__(self):
        next

    def run_benchmark():
        if ENABLE_EXEC_LOG:
            print("******** RUNNING PSO BENCHMARKS ********")
            print("\nBelow are logged sample optimization results:")
            print("\tW\tC1\tC2\tSpherical\t\tAckley\t\tMichalewicz\t\tKatsuura")
        data = []
        for w in np.arange(0.8, 0.85, 0.1):
            for c in np.arange(0.05, 2.525, 0.05):
                s_res = BenchmarkPSO.run_pso_batch(w, c, c, ObjFn.spherical, ObjFn.spherical_bounds)
                a_res = BenchmarkPSO.run_pso_batch(w, c, c, ObjFn.ackley, ObjFn.ackley_bounds)
                m_res = BenchmarkPSO.run_pso_batch(w, c, c, ObjFn.michalewicz, ObjFn.michalewicz_bounds)
                k_res = BenchmarkPSO.run_pso_batch(w, c, c, ObjFn.katsuura, ObjFn.katsuura_bounds)
                data.append([w, c, c, s_res, a_res, m_res, k_res])
                # if ENABLE_EXEC_LOG and math.floor(c * 100) % 25 == 0:
                #     print("\t{:.1f}\t{:.2f}\t{:.2f}\t{:.10f}\t{:.10f}\t\t{:.10f}\t{:.10f}".format(w, c, c, s_res, a_res, m_res, k_res))
                print("\t{:.1f}\t{:.2f}\t{:.2f}\t{:.10f}\t\t{:.10f}\t{:.10f}".format(w, c, c, a_res, m_res, k_res))
        df = pd.DataFrame(data, columns=['inertia', 'cognitive', 'social', 'spherical', 'ackley', 'michalewicz', 'katsuura'])
        if ENABLE_EXEC_LOG:
            print("PSO Benchmarking completed")
        return df
        
    def run_pso_batch(w, c1, c2, func, bounds):
        total = 0
        for i in range(PSO_RUNS):
            pso = PSO(w = w, c1 = c1, c2 = c2, obj_func = func, bounds = bounds)
            total += pso.run()
        return total / PSO_RUNS



class TunePSO:
    """
    Implements the logical steps of the PSO Intelligent Parameter Tuning
    using the Tabu Search algorithm
    """




def main():
    df = BenchmarkPSO.run_benchmark()
    df.to_csv('pso_benchmark.csv')


if __name__ == "__main__":
    # main(sys.argv)
    main()